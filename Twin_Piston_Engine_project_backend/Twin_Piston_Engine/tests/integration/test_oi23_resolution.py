"""
OI-23 resolution (Prompt 18): flight phase derived in the pipeline, the
WebSocket streams every record, and RUL plausibility is checked.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.core.config import get_settings
from src.core.provenance import FaultClass, FlightPhase
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import (
    DEFAULT_MISSION_WAYPOINTS,
    PipelineReplayAdapter,
    ReplayEngine,
    RunningPipeline,
    mission_profile_from_waypoints,
)

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"


# ---------------------------------------------------------------------------
# Flight phase
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _mission_steps():
    """The default mission: 300 s climb power from sea level to 1500 m, 60 s
    power reduction, cruise at 1500 m (420 s at 1 Hz)."""
    prof = mission_profile_from_waypoints(DEFAULT_MISSION_WAYPOINTS)
    recs, _, _ = ScenarioRunner(seed=3).run_scenario(duration_s=420.0, dt_s=1.0, operating_profile=prof)
    return tuple(PipelineReplayAdapter().process_sequence(recs))


def test_flight_phase_is_derived_and_never_ground_in_flight() -> None:
    """Measured: TAKEOFF (climb power below 500 m), CLIMB, then CRUISE after
    the power reduction; GROUND never appears."""
    phases = [s.mission_state.flight_phase for s in _mission_steps()]
    assert FlightPhase.GROUND not in phases
    assert phases[0] == FlightPhase.TAKEOFF
    assert FlightPhase.CLIMB in phases[:300]
    assert all(p == FlightPhase.CRUISE for p in phases[-30:])
    # order: takeoff -> climb -> cruise, no return to an earlier phase
    first = {p: phases.index(p) for p in (FlightPhase.TAKEOFF, FlightPhase.CLIMB, FlightPhase.CRUISE)}
    assert first[FlightPhase.TAKEOFF] < first[FlightPhase.CLIMB] < first[FlightPhase.CRUISE]


def test_phase_not_derivable_keeps_last_phase_with_quality_zero() -> None:
    """rpm invalid (crank period missing) mid-flight: the classifier returns
    UNKNOWN, which MissionState would render as GROUND; the pipeline keeps the
    last derived phase and marks its quality 0 instead."""
    recs, _, _ = ScenarioRunner(seed=5).run_scenario(duration_s=40.0, dt_s=1.0,
                                                      operating_profile=[{"rpm": 4000.0, "map_pa": 110000.0}])
    recs = [r.model_copy(update={"crank_period_us": None, "crank_period_burst_us": ()}) if 20 <= i < 30 else r
            for i, r in enumerate(recs)]
    steps = PipelineReplayAdapter().process_sequence(recs)
    assert steps[19].mission_state.flight_phase == FlightPhase.CRUISE
    lost = steps[20:30]
    assert all(s.mission_state.flight_phase == FlightPhase.CRUISE for s in lost)
    assert all(s.mission_state.phase_quality == 0.0 for s in lost)


def test_mission_risk_api_reports_the_derived_phase() -> None:
    app = create_app()
    replay = ReplayEngine(scenario_id="phase")
    recs, gts, _ = ScenarioRunner(seed=6).run_scenario(duration_s=20.0, dt_s=1.0,
                                                        operating_profile=[{"rpm": 4000.0, "map_pa": 110000.0}])
    replay.load_scenario("phase", recs, gts)
    pipe = RunningPipeline(PipelineReplayAdapter())
    app.dependency_overrides[get_replay_engine] = lambda: replay
    app.dependency_overrides[get_running_pipeline] = lambda: pipe
    body = TestClient(app).get("/api/v1/mission/risk").json()
    assert body["flight_phase"] == "CRUISE"


# ---------------------------------------------------------------------------
# WebSocket: every record, not the first 5
# ---------------------------------------------------------------------------

def test_websocket_streams_every_record() -> None:
    app = create_app()
    replay = ReplayEngine(scenario_id="ws")
    recs, gts, _ = ScenarioRunner(seed=7).run_scenario(duration_s=12.0, dt_s=1.0)
    replay.load_scenario("ws", recs, gts)
    app.dependency_overrides[get_replay_engine] = lambda: replay
    import src.api.v1.websocket as ws_mod
    orig = ws_mod.get_replay_engine
    ws_mod.get_replay_engine = lambda: replay
    try:
        with TestClient(app) as client, client.websocket_connect("/api/v1/ws/engine") as ws:
            assert ws.receive_json()["payload"]["status"] == "CONNECTED"
            telemetry, health, done = 0, 0, None
            while done is None:
                msg = ws.receive_json()
                if msg["event_type"] == "telemetry_update":
                    telemetry += 1
                elif msg["event_type"] == "health_update":
                    health += 1
                elif msg["event_type"] == "system_status":
                    done = msg["payload"]
    finally:
        ws_mod.get_replay_engine = orig
    assert telemetry == health == len(recs) == 12
    assert done == {"status": "STREAM_COMPLETE", "records_streamed": 12, "records_available": 12}


# ---------------------------------------------------------------------------
# RUL plausibility
# ---------------------------------------------------------------------------

def test_baseline_rul_on_a_short_fault_run_is_marked_unvalidated() -> None:
    """The baseline RUL extrapolates the health-index slope of a 120 s run into
    hours (measured 0.113 h on OIL_DEGRADATION in Prompt 15). Such a number is
    not a plausible engine RUL: it must never be presented as validated, and
    its interval must be labelled not calibrated. It stays within [0, horizon]."""
    recs, _, _ = ScenarioRunner(seed=42).run_scenario(
        duration_s=120.0, dt_s=1.0, operating_profile=[{"rpm": 4000.0, "map_pa": 110000.0}],
        fault_scenarios=[FaultScenarioConfig(fault_class=FaultClass.OIL_DEGRADATION, severity=1.0,
                                             onset_time_s=30.0, duration_s=1e4)])
    steps = PipelineReplayAdapter().process_sequence(recs)
    horizon = get_settings().rul.max_prediction_horizon_hours
    for s in steps:
        r = s.rul_state
        assert r.validated is False and r.interval_calibrated is False
        assert 0.0 <= r.hours_remaining <= horizon

    app = create_app()
    replay = ReplayEngine(scenario_id="oil")
    replay.load_scenario("oil", recs, [None] * len(recs))
    app.dependency_overrides[get_replay_engine] = lambda: replay
    app.dependency_overrides[get_running_pipeline] = lambda: RunningPipeline(PipelineReplayAdapter())
    body = TestClient(app).get("/api/v1/engine/rul").json()
    assert body["status"] in ("UNVALIDATED", "INSUFFICIENT_HISTORY") and body["interval_calibrated"] is False


@pytest.mark.skipif(not (MODEL_ROOT / "life1" / "rul_lifetime.joblib").exists(),
                    reason="lifetime RUL bundle not generated (git-ignored; scripts/train_rul.py)")
def test_lifetime_rul_is_plausible_along_a_degradation_history() -> None:
    """Lifetime RUL from a severity history: no number before degradation; once
    ESTIMATED, RUL lies in [0, 400] h, the interval brackets it, and RUL falls
    as the severity keeps rising."""
    import joblib

    from src.l3_ml import lifetime_rul as L
    b = joblib.load(MODEL_ROOT / "life1" / "rul_lifetime.joblib")
    healthy = L.history_features(np.zeros(40))
    assert set(b["rul_model"].predict(healthy)["status"]) == {"NO_DEGRADATION"}
    sev = np.concatenate([np.zeros(20), np.linspace(0.12, 0.9, 60)])     # onset, then steady growth
    p = b["rul_model"].predict(L.history_features(sev))
    est = p["status"] == "ESTIMATED"
    assert est.sum() >= 30
    rul, lo, hi = p["rul_h"][est], p["rul_lo_h"][est], p["rul_hi_h"][est]
    assert np.all((rul >= 0.0) & (rul <= L.RUL_CAP_H)) and np.all((lo <= rul) & (rul <= hi))
    assert rul[-5:].mean() < rul[:5].mean()


def test_websocket_stops_when_the_client_disconnects() -> None:
    """With every record streamed, a client that leaves early must end the
    stream (the handler watches for the disconnect) instead of the server
    running through the whole replay."""
    import time

    import src.api.v1.websocket as ws_mod
    app = create_app()
    replay = ReplayEngine(scenario_id="long")
    recs, gts, _ = ScenarioRunner(seed=8).run_scenario(duration_s=600.0, dt_s=1.0)
    replay.load_scenario("long", recs, gts)
    orig = ws_mod.get_replay_engine
    ws_mod.get_replay_engine = lambda: replay
    try:
        t0 = time.monotonic()
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/engine") as ws:
                ws.receive_json()
                ws.receive_json()
        elapsed = time.monotonic() - t0
    finally:
        ws_mod.get_replay_engine = orig
    assert elapsed < 30.0, elapsed          # streaming all 600 records takes > 30 s (0.05 s pacing alone)
    assert not ws_mod.manager.active_connections
