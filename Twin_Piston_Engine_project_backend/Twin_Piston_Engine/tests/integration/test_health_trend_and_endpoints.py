"""
L3 health trend (windowed Theil-Sen, shared with the Prompt 14 overheating
trends) and /rul, /mission/risk, /diagnostics/anomaly, /diagnostics/fault
served from the running pipeline (OI-20 follow-up).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.core.config import get_settings
from src.core.provenance import Provenance
from src.core.schemas import OperatingPoint, ResidualState, make_tagged
from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter, ReplayEngine, RunningPipeline
from src.l3_ml.health_supervision import HealthSupervisionEngine
from tests.scientific.physics_audit_harness import SCENARIOS, simulate

BY_NAME = {s.name: s for s in SCENARIOS}
T0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
OP = OperatingPoint(rpm=4000.0, map_pressure_pa=110000.0, altitude_m=0.0, ambient_temp_k=288.15,
                    ambient_pressure_pa=101325.0)


@lru_cache(maxsize=None)
def _records(name: str):
    return tuple(simulate(BY_NAME[name]))


@lru_cache(maxsize=None)
def _pipeline(name: str):
    return tuple(PipelineReplayAdapter().process_sequence(list(_records(name))))


# ---------------------------------------------------------------------------
# Health trend
# ---------------------------------------------------------------------------

def test_nominal_120_s_ends_stable() -> None:
    """Measured (seed 42, 1 Hz): slope +0.00138 HI/min, 95 % CI +0.00072 .. +0.00206.
    The CI excludes zero but the slope is below trend_min_rate_per_min (0.006),
    so the trend is STABLE."""
    h = _pipeline("NOMINAL")[-1].health_state
    info = h.evidence["trend"]
    assert h.trend == "STABLE"
    assert abs(info["slope_per_min"]) < get_settings().health.trend_min_rate_per_min
    lo, hi = info["ci_per_min"]
    assert lo <= info["slope_per_min"] <= hi
    assert info["samples"] == 120


def test_oil_degradation_ends_degradation() -> None:
    """Measured: slope -0.0153 HI/min, 95 % CI -0.0214 .. -0.0114."""
    h = _pipeline("OIL_DEGRADATION 1.0")[-1].health_state
    lo, hi = h.evidence["trend"]["ci_per_min"]
    assert h.trend == "DEGRADATION"
    assert hi < 0.0 and h.evidence["trend"]["slope_per_min"] <= -get_settings().health.trend_min_rate_per_min
    assert h.health_rate < 0.0


def test_insufficient_data_until_the_window_is_full() -> None:
    steps = _pipeline("NOMINAL")
    window = get_settings().health.trend_window_s
    assert all(s.health_state.trend == "INSUFFICIENT_DATA" for s in steps[:int(window) - 1])
    assert steps[0].health_state.health_rate is None


def _healthy_res(oil_residual_pa: float = 0.0, op: OperatingPoint = OP) -> ResidualState:
    return ResidualState(operating_point=op, residuals={"oil_pressure": make_tagged(oil_residual_pa, Provenance.DERIVED)})


def test_single_sample_spike_does_not_change_the_trend() -> None:
    engine = HealthSupervisionEngine()
    trends = []
    for k in range(200):
        spike = k == 150
        h = engine.evaluate_health(residual_state=_healthy_res(-100000.0 if spike else 0.0),
                                   timestamp=T0 + timedelta(seconds=k))
        if spike:
            assert h.health_index.value < 0.95  # the spike is real in the HI itself
        trends.append(h.trend)
    assert set(trends[130:]) == {"STABLE"}, set(trends[130:])


def test_operating_point_step_restarts_the_window() -> None:
    engine = HealthSupervisionEngine()
    for k in range(130):
        h = engine.evaluate_health(residual_state=_healthy_res(), timestamp=T0 + timedelta(seconds=k))
    assert h.trend == "STABLE"
    stepped = OP.model_copy(update={"rpm": 5500.0})
    h = engine.evaluate_health(residual_state=_healthy_res(op=stepped), timestamp=T0 + timedelta(seconds=130))
    assert h.trend == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Endpoints from the running pipeline
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _api(name: str) -> dict[str, dict]:
    engine = ReplayEngine("t")
    engine.load_scenario("t", list(_records(name)), [])
    pipeline = RunningPipeline(PipelineReplayAdapter())
    app = create_app()
    app.dependency_overrides[get_replay_engine] = lambda: engine
    app.dependency_overrides[get_running_pipeline] = lambda: pipeline
    out = {}
    with TestClient(app) as client:
        for ep in ("/api/v1/engine/rul", "/api/v1/mission/risk", "/api/v1/diagnostics/anomaly",
                   "/api/v1/diagnostics/fault"):
            r = client.get(ep)
            assert r.status_code == 200, (ep, r.text)
            out[ep] = r.json()
    return out


def test_rul_from_pipeline_state() -> None:
    api = _api("OIL_DEGRADATION 1.0")["/api/v1/engine/rul"]
    full = _pipeline("OIL_DEGRADATION 1.0")[-1].rul_state
    one = PipelineReplayAdapter().process_sequence([_records("OIL_DEGRADATION 1.0")[-1]])[-1].rul_state
    assert api["records_in_window"] == len(_records("OIL_DEGRADATION 1.0"))
    assert api["hours_remaining"] == pytest.approx(full.hours_remaining)
    assert (api["trend"], api["inference_status"]) == (full.trend, full.status.value) == ("DEGRADATION", "SUCCESS")
    # Prompt 17: the baseline estimate is not validated, so it is never presented as SUCCESS
    assert api["status"] == "UNVALIDATED" and api["validated"] is False and api["validation_note"]
    # the latest record alone has no history: not what the endpoint returns
    assert one.status.value == "INSUFFICIENT_HISTORY" and api["inference_status"] != one.status.value


def test_mission_risk_from_pipeline_state() -> None:
    api = _api("OIL_DEGRADATION 1.0")["/api/v1/mission/risk"]
    full = _pipeline("OIL_DEGRADATION 1.0")[-1].mission_state
    one = PipelineReplayAdapter().process_sequence([_records("OIL_DEGRADATION 1.0")[-1]])[-1].mission_state
    assert api["records_in_window"] == len(_records("OIL_DEGRADATION 1.0"))
    assert api["risk_score"] == pytest.approx(full.risk_score)
    assert api["risk_score"] != pytest.approx(one.risk_score)
    assert api["flight_phase"] == full.flight_phase.value and api["risk_trend"] == full.risk_trend


@pytest.mark.parametrize("ep", ["/api/v1/diagnostics/anomaly", "/api/v1/diagnostics/fault"])
def test_ml_endpoints_from_pipeline_state_and_no_substitutes(ep: str) -> None:
    """Both L3 models are MODEL_UNAVAILABLE (OI-16): the endpoints report that
    status with None values (they returned SUCCESS / confidence 1.0 before)."""
    api = _api("OIL_DEGRADATION 1.0")[ep]
    last = _pipeline("OIL_DEGRADATION 1.0")[-1]
    result = last.anomaly_result if ep.endswith("anomaly") else last.fault_result
    assert api["records_in_window"] == len(_records("OIL_DEGRADATION 1.0"))
    assert api["status"] == result.status.value == "MODEL_UNAVAILABLE"
    if ep.endswith("anomaly"):
        assert api["anomaly_score"] is None and api["is_anomaly"] is None
    else:
        assert api["confidence"] is None and api["probabilities"] is None
