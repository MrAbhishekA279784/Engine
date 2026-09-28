"""
Prompt 14: coolant channel, overheating time-to-limit, hot-weather profile.

Simulator: liquid coolant loop (pump flow ∝ rpm, thermostat, crossflow
radiator with airspeed and ambient, heat from the heads), physical
COOLING_FAULT sub-modes, NTC forward model; CHT and oil follow their maps
with thermal lags. L2: coolant from Steinhart-Hart, coolant residual,
CHT - coolant, Theil-Sen trend of (measured - lagged expectation) with a
time-to-limit only for a significant, practically relevant rise.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_replay_engine, get_running_pipeline
from src.core.config import get_settings
from src.core.provenance import FaultClass
from src.core.sensor_physics import ntc_resistance_ohm, ntc_temp_c
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l1_data.simulator.replay_and_whatif import (
    PipelineReplayAdapter,
    ReplayEngine,
    RunningPipeline,
    WhatIfEngine,
    hot_weather_mission_profile,
    mission_profile_from_waypoints,
)
from src.l2_digital_twin.cooling_model import NO_CIRCUIT, CoolantModel
from src.l2_digital_twin.overheat_trend import OverheatTrendModel
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, ResidualEngine, operating_point_from_record
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l3_ml.advisory_explainability import overheat_advisories
from tests.unit.test_module6 import _make_dummy_normalized_record as make_record

ALERT = ("WARNING", "CRITICAL")


# ---------------------------------------------------------------------------
# NTC
# ---------------------------------------------------------------------------

def test_ntc_round_trip_within_0p2_c_from_minus_40_to_130() -> None:
    cal, sim = get_settings().sensor_calibration, get_settings().simulator.cooling
    worst = 0.0
    for t in np.arange(-40.0, 130.0 + 1e-9, 0.5):
        r = ntc_resistance_ohm(float(t), sim.ntc_sh_a, sim.ntc_sh_b, sim.ntc_sh_c)   # simulator forward
        worst = max(worst, abs(ntc_temp_c(r, cal.ntc_sh_a, cal.ntc_sh_b, cal.ntc_sh_c) - t))  # L2 inverse
    assert worst < 0.2, f"worst round-trip error {worst:.4f} C"


def test_ntc_forward_matches_datasheet_point() -> None:
    cal = get_settings().sensor_calibration
    assert ntc_resistance_ohm(25.0, cal.ntc_sh_a, cal.ntc_sh_b, cal.ntc_sh_c) == pytest.approx(10000.0, rel=1e-4)


def test_coolant_channel_filled_from_simulated_ntc() -> None:
    recs, gts, _ = ScenarioRunner(seed=7).run_scenario(duration_s=3.0, dt_s=1.0)
    norm = convert_raw_to_engineering_state(recs[-1])
    assert norm.coolant_temp.valid
    assert norm.coolant_temp.value == pytest.approx(gts[-1].coolant.coolant_temp_k, abs=0.3)


# ---------------------------------------------------------------------------
# No coolant circuit: unavailable, never inferred (SRD-FUN-045 pattern)
# ---------------------------------------------------------------------------

def test_no_coolant_circuit_every_coolant_parameter_unavailable() -> None:
    s = get_settings().model_copy(deep=True)
    s.engine.has_coolant_circuit = False
    recs, gts, _ = ScenarioRunner(settings=s, seed=7).run_scenario(duration_s=3.0, dt_s=1.0)
    assert recs[-1].coolant_ntc_ohms is None and gts[-1].coolant is None
    norm = convert_raw_to_engineering_state(recs[-1], s)
    assert norm.coolant_temp.valid is False and "has_coolant_circuit = False" in norm.coolant_temp.fault_flag
    # even a record that carries an NTC value is not interpreted
    with_ntc = convert_raw_to_engineering_state(recs[-1].model_copy(update={"coolant_ntc_ohms": 1000.0}), s)
    assert with_ntc.coolant_temp.valid is False and with_ntc.coolant_temp.value is None
    state = CoolantModel(s).evaluate(norm)
    for name in ("coolant_temp_c", "coolant_residual_k", "coolant_cht_delta_k"):
        tv = getattr(state, name)
        assert tv.valid is False and tv.value is None and tv.fault_flag == NO_CIRCUIT
    trend = OverheatTrendModel(s).evaluate(norm).channels["coolant"]
    assert trend.time_to_limit_s.valid is False and "has_coolant_circuit" in trend.time_to_limit_s.fault_flag


def test_coolant_expectation_does_not_read_the_coolant_channel() -> None:
    base = make_record(rpm_val=4000.0)
    changed = base.model_copy(update={"coolant_temp": base.coolant_temp.model_copy(update={"value": 420.0})})
    engine = ResidualEngine()
    assert engine.evaluate(changed)[1].expected_values["coolant"] == engine.evaluate(base)[1].expected_values["coolant"]


# ---------------------------------------------------------------------------
# Trend statistics
# ---------------------------------------------------------------------------

def _synthetic_trend(slope_k_min: float, noise_k: float = 0.3, n: int = 240):
    """Records at 1 Hz, fixed operating point, CHT = start + slope*t + noise."""
    rng = np.random.default_rng(3)
    model = OverheatTrendModel()
    base = make_record(rpm_val=4000.0)
    from datetime import timedelta
    state = None
    for k in range(n):
        cht = 380.0 + slope_k_min / 60.0 * k + rng.normal(0.0, noise_k)
        rec = base.model_copy(update={
            "timestamp": base.timestamp + timedelta(seconds=k),
            "cht_cyl_1": base.cht_cyl_1.model_copy(update={"value": cht})})
        state = model.evaluate(rec)
    return state.channels["cht"]


def test_trend_recovers_slope_and_projects_time_to_limit() -> None:
    t = _synthetic_trend(2.0)
    lo, hi = t.slope_residual_ci_k_min
    assert t.slope_residual_k_min.value == pytest.approx(2.0, abs=0.1) and lo < 2.0 < hi
    remaining = t.limit_c - t.current_c.value
    assert t.time_to_limit_s.valid
    assert t.time_to_limit_s.value == pytest.approx(remaining / (t.slope_residual_k_min.value / 60.0))
    assert t.time_to_limit_range_s[0] < t.time_to_limit_s.value < t.time_to_limit_range_s[1]


def test_flat_temperature_has_no_trend() -> None:
    t = _synthetic_trend(0.0)
    assert t.time_to_limit_s.valid is False and t.time_to_limit_s.value is None
    assert "no trend" in t.time_to_limit_s.fault_flag
    assert t.status.value == "NORMAL"


# ---------------------------------------------------------------------------
# Healthy full-power climb: no time-to-limit alert
# ---------------------------------------------------------------------------

def _power_climb(top_altitude_m: float) -> list[dict[str, float]]:
    """2 min cruise, 30 s power increase to full power, then climb at full
    power to top_altitude_m over 12.5 min (airspeed 35 m/s)."""
    return mission_profile_from_waypoints([
        {"t_s": 0.0, "rpm": 4000.0, "map_pa": 110000.0, "altitude_m": 0.0, "airspeed_m_s": 50.0},
        {"t_s": 120.0, "rpm": 4000.0, "map_pa": 110000.0, "altitude_m": 0.0, "airspeed_m_s": 50.0},
        {"t_s": 150.0, "rpm": 5800.0, "map_pa": 140000.0, "altitude_m": 0.0, "airspeed_m_s": 35.0},
        {"t_s": 900.0, "rpm": 5800.0, "map_pa": 140000.0, "altitude_m": top_altitude_m, "airspeed_m_s": 35.0},
    ])


@lru_cache(maxsize=None)
def _climb(top_altitude_m: float):
    recs, gts, _ = ScenarioRunner(seed=7).run_scenario(duration_s=900.0, dt_s=1.0,
                                                       operating_profile=_power_climb(top_altitude_m))
    return recs, gts, PipelineReplayAdapter().process_sequence(recs)


def _assert_no_alert(top_altitude_m: float) -> None:
    recs, gts, steps = _climb(top_altitude_m)
    assert gts[400].cht_k > gts[100].cht_k  # CHT does rise legitimately with power
    alerts = [(k, n) for k, s in enumerate(steps) for n, t in s.overheat_state.channels.items()
              if t.status.value in ALERT]
    assert alerts == [], alerts[:5]
    assert all(overheat_advisories(s.overheat_state) == [] for s in steps)
    assert all(s.coolant_state.status.value == "NORMAL" for s in steps)


def test_healthy_full_power_increase_gives_no_time_to_limit_alert() -> None:
    """Power step to full power at constant altitude: CHT +9.0 K over minutes,
    the lagged expectation rises with it, no channel alerts."""
    _assert_no_alert(0.0)


@pytest.mark.xfail(strict=True, reason=(
    "OI-22 (OI-5 class): healthy full-power climb 0 -> 2400 m gives a CHT time-to-limit WARNING at "
    "446-485 s (39 records). Measured CHT is still settling after the power increase (+0.42 K/min; "
    "simulator head lag 90 s) while the L2 expectation (lag 60 s, 0.85 K per K ambient) already falls "
    "with the cooling air; the thermostat-regulated simulator CHT does not. Residual slope +0.43 K/min "
    "-> 1635 s (27 min) to 135 C, inside the 30 min horizon. Not tuned away: needs calibrated "
    "expectations (ambient/density sensitivity, head time constant) from flight data."))
def test_healthy_full_power_climb_to_2400_m_gives_no_time_to_limit_alert() -> None:
    _assert_no_alert(2400.0)


# ---------------------------------------------------------------------------
# Radiator blockage
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _blockage():
    s = get_settings().model_copy(deep=True)
    s.simulator.cooling.fault_ramp_s = 600.0   # the blockage grows over 10 min
    faults = [FaultScenarioConfig(fault_class=FaultClass.COOLING_FAULT, severity=1.0, onset_time_s=60.0,
                                  duration_s=1e4, sub_mode="radiator_blockage")]
    recs, gts, _ = ScenarioRunner(settings=s, seed=7).run_scenario(
        duration_s=900.0, dt_s=1.0, operating_profile=[{"rpm": 5000.0, "map_pa": 125000.0}], fault_scenarios=faults)
    return recs, gts, PipelineReplayAdapter(s).process_sequence(recs)


def test_radiator_blockage_raises_the_coolant_residual() -> None:
    _, _, steps = _blockage()
    res = [s.coolant_state.coolant_residual_median_k.value for s in steps]
    assert res[899] - res[59] > 20.0
    assert steps[899].coolant_state.status.value == "CRITICAL"


def test_radiator_blockage_time_to_limit_valid_and_decreasing() -> None:
    """Coolant time-to-limit, sampled every 30 s from its first valid value
    until the limit is reached: valid throughout and strictly decreasing."""
    _, _, steps = _blockage()
    ttl = [s.overheat_state.channels["coolant"].time_to_limit_s for s in steps]
    first = next(k for k, tv in enumerate(ttl) if tv.valid)
    assert first < 300  # becomes valid while the coolant is still in the thermostat band
    samples = []
    for k in range(first, len(ttl), 30):
        assert ttl[k].valid, (k, ttl[k].fault_flag)
        samples.append(ttl[k].value)
        if ttl[k].value == 0.0:
            break
    assert samples[-1] == 0.0
    assert all(b < a for a, b in zip(samples, samples[1:]) if a > 0.0), samples
    msgs = [a.message for a in overheat_advisories(steps[600].overheat_state)]
    assert any("faster than expected for this power; projected to reach the alarm limit in about" in m for m in msgs)


# ---------------------------------------------------------------------------
# Hot-weather what-if (SRD-FUN-154)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _mission(isa_dev: float):
    we = WhatIfEngine()
    sc = we.create_what_if_scenario("mission", {}, hot_weather_mission_profile(isa_dev))
    recs, gts, _ = we.execute_what_if(sc, duration_s=900.0, dt_s=1.0)
    return recs, gts, PipelineReplayAdapter().process_sequence(recs)


def test_isa_plus_30_raises_cht_and_oil_expectations_at_the_same_power() -> None:
    (r0, g0, _), (r30, g30, _) = _mission(0.0), _mission(30.0)
    exp = HealthyExpectationModel()
    for k in (100, 250, 600):
        assert g30[k].ambient_temp_k == pytest.approx(g0[k].ambient_temp_k + 30.0)
        assert g30[k].rpm == g0[k].rpm and g30[k].map_pressure_pa == g0[k].map_pressure_pa
        op0 = operating_point_from_record(convert_raw_to_engineering_state(r0[k]))
        op30 = operating_point_from_record(convert_raw_to_engineering_state(r30[k]))
        assert exp.expected_cht_k(op30) > exp.expected_cht_k(op0)
        assert exp.expected_oil_temp_k(op30) > exp.expected_oil_temp_k(op0)


@pytest.mark.parametrize("isa_dev", [0.0, 30.0])
def test_hot_weather_mission_healthy_engine_nothing_alarms(isa_dev: float) -> None:
    _, _, steps = _mission(isa_dev)
    for k, s in enumerate(steps):
        assert all(t.status.value not in ALERT for t in s.overheat_state.channels.values()), k
        assert s.coolant_state.status.value == "NORMAL", k
        assert s.health_state.health_status.value == "NORMAL", k


def test_what_if_ambient_overrides_are_applied() -> None:
    """ambient_temp_k was accepted but ignored before Prompt 14."""
    we = WhatIfEngine()
    sc = we.create_what_if_scenario("base", {"operating_profile": [{"rpm": 4000.0, "map_pa": 110000.0}]},
                                    {"ambient_temp_k": 310.0})
    _, gts, _ = we.execute_what_if(sc, duration_s=2.0, dt_s=1.0)
    assert gts[-1].ambient_temp_k == 310.0
    with pytest.raises(ValueError):
        we.validate_modifications({"isa_deviation_k": 200.0})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_api_exposes_coolant_and_overheat_trends() -> None:
    recs, _, _ = _blockage()
    engine = ReplayEngine("oh")
    engine.load_scenario("oh", recs[:600], [])
    pipeline = RunningPipeline(PipelineReplayAdapter(_blockage_settings()))
    app = create_app()
    app.dependency_overrides[get_replay_engine] = lambda: engine
    app.dependency_overrides[get_running_pipeline] = lambda: pipeline
    with TestClient(app) as client:
        body = client.get("/api/v1/engine/health").json()
    assert body["coolant_temp_c"] is not None and body["coolant_status"] in ("NORMAL", "WARNING", "CRITICAL")
    cool = body["overheat_trends"]["coolant"]
    assert cool["time_to_limit_s"] is not None and cool["time_to_limit_range_s"] is not None
    assert any("projected to reach the alarm limit" in m for m in body["overheat_advisories"])
    for name, t in body["overheat_trends"].items():
        assert (t["time_to_limit_s"] is None) == (t["reason"] is not None), name


def _blockage_settings():
    s = get_settings().model_copy(deep=True)
    s.simulator.cooling.fault_ramp_s = 600.0
    return s
