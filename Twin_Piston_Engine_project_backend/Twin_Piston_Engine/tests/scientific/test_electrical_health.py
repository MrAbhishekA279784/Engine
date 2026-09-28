"""
Prompt 12: battery and alternator health from bus voltage, alternator current
and battery current (simulator electrical.py -> L2 electrical_model.py -> L3).

The simulator uses its own constants (simulator.electrical); the L2 model uses
electrical.* expectations. Faults are physical parameter changes in the
simulator (SRD-FUN-152): setpoint drift, output collapse, open rectifier diode,
battery R_int growth / capacity fade.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.config import _validate_ripple_band, get_settings
from src.core.exceptions import ConfigurationError
from src.core.sensor_physics.M05_nyquist_guard import band_is_representable
from src.l1_data.simulator.electrical import ElectricalSimulator
from src.l1_data.simulator.fault_injection import FaultMode
from src.l1_data.simulator.forward_simulator import FaultScenarioConfig, ScenarioRunner
from src.l2_digital_twin.electrical_model import NOT_OBSERVABLE_REGULATED, ElectricalModel
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l3_ml.advisory_explainability import electrical_advisories
from src.l3_ml.health_supervision import HealthSupervisionEngine

DT = 0.1
IDLE = [{"start_time_s": 0.0, "end_time_s": 1e6, "rpm": 1800.0, "map_pa": 60000.0}]


def _run(duration_s: float, faults=None, profile=None, settings=None, seed: int = 7):
    settings = settings or get_settings()
    recs, gts, _ = ScenarioRunner(settings=settings, seed=seed).run_scenario(
        duration_s=duration_s, dt_s=DT, fault_scenarios=faults, operating_profile=profile)
    model = ElectricalModel(settings)
    states = [model.evaluate(convert_raw_to_engineering_state(r)) for r in recs]
    return recs, gts, states


def _charging(sub_mode: str, onset: float = 30.0) -> list[FaultScenarioConfig]:
    return [FaultScenarioConfig(fault_class=FaultMode.CHARGING_FAULT, severity=1.0, onset_time_s=onset,
                                duration_s=1e4, sub_mode=sub_mode)]


# ---------------------------------------------------------------------------
# Healthy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nominal():
    return _run(120.0)


def test_healthy_residual_within_0p2_v_above_cut_in(nominal) -> None:
    _, gts, states = nominal
    assert all(g.rpm >= get_settings().electrical.cut_in_rpm for g in gts)
    res = [s.charging_residual_v.value for s in states]
    assert all(s.charging_residual_v.valid for s in states)
    assert max(abs(r) for r in res) < 0.2, f"max |residual| {max(abs(r) for r in res):.3f} V"
    assert all(s.charging_status.value == "NORMAL" for s in states)


def test_healthy_ripple_below_healthy_band(nominal) -> None:
    _, _, states = nominal
    rip = [s.voltage_ripple_pct.value for s in states]
    assert all(s.voltage_ripple_pct.valid for s in states)
    assert max(rip) < get_settings().electrical.ripple_healthy_pct, f"max ripple {max(rip):.3f} %"


def test_nominal_flight_has_no_observable_r_int_window(nominal) -> None:
    """Reported figure: a 120 s NOMINAL flight (4000 rpm, regulator in control)
    gives 0 observable R_int windows, so R_int stays invalid with the reason."""
    _, _, states = nominal
    assert all(s.r_int_step_count == 0 for s in states)
    assert all(not s.battery_resistance_mohm.valid for s in states)
    assert states[-1].battery_resistance_mohm.value is None
    assert states[-1].battery_resistance_mohm.fault_flag == NOT_OBSERVABLE_REGULATED


def test_healthy_ehi_normal_and_no_electrical_advisory(nominal) -> None:
    _, _, states = nominal
    s = states[-1]
    assert s.ehi.valid and s.ehi_band == "NORMAL"
    assert electrical_advisories(s) == []


# ---------------------------------------------------------------------------
# CHARGING_FAULT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_mode", ["setpoint_drift", "output_collapse"])
def test_charging_fault_residual_beyond_warning_within_60_s(sub_mode: str) -> None:
    onset = 30.0
    _, _, states = _run(120.0, _charging(sub_mode, onset))
    warn = get_settings().electrical.charging_warning_v
    flagged = [i * DT for i, s in enumerate(states)
               if s.charging_residual_median_v.valid and abs(s.charging_residual_median_v.value) > warn]
    assert flagged, "charging residual never left the warning band"
    assert all(t >= onset for t in flagged), "flagged before onset"
    assert flagged[0] - onset <= 60.0, f"detected {flagged[0] - onset:.1f} s after onset"
    msgs = [a.message for a in electrical_advisories(states[-1])]
    assert any("below expectation for this rpm, consistent with regulator or alternator drive failure" in m
               and "Inspect the regulator and the alternator drive" in m for m in msgs), msgs


def test_open_diode_ripple_rises_while_mean_voltage_in_band() -> None:
    onset = 30.0
    _, _, states = _run(120.0, _charging("open_diode", onset))
    cfg = get_settings().electrical
    n0 = int(onset / DT)
    healthy = np.median([s.voltage_ripple_pct.value for s in states[:n0]])
    faulty = np.median([s.voltage_ripple_pct.value for s in states[n0 + 5:]])
    assert faulty >= 3.0 * healthy, f"ripple {faulty:.3f} % vs healthy {healthy:.3f} %"
    res = [s.charging_residual_v.value for s in states[n0 + 5:]]
    assert abs(np.mean(res)) < cfg.charging_warning_v, f"mean residual {np.mean(res):+.3f} V"
    s = states[-1]
    assert s.ripple_status.value in ("WARNING", "CRITICAL")
    # Shaft-locked: integer ripple order; frequency not attributable without the pole count
    assert s.ripple_order.valid and abs(s.ripple_order.value - round(s.ripple_order.value)) < 0.25
    assert not s.ripple_dominant_frequency_hz.valid
    assert "generator pole count unknown" in s.ripple_dominant_frequency_hz.fault_flag
    assert any("open rectifier diode" in a.message for a in electrical_advisories(s))


# ---------------------------------------------------------------------------
# BATTERY_DEGRADATION and R_int observability
# ---------------------------------------------------------------------------

def test_battery_degradation_r_int_within_20_percent_after_5_steps() -> None:
    """Below cut-in (1800 rpm) the battery carries every load step, so R_int is
    observable. Degradation ramps over 60 s; compare after the ramp."""
    s = get_settings().model_copy(deep=True)
    s.simulator.electrical.battery_degradation_ramp_s = 60.0
    faults = [FaultScenarioConfig(fault_class=FaultMode.BATTERY_DEGRADATION, severity=1.0, onset_time_s=0.0,
                                  duration_s=1e4)]
    _, gts, states = _run(240.0, faults, IDLE, settings=s)
    steps_after_ramp = sum(
        1 for i, st in enumerate(states) if i * DT > 60.0 for e in st.ehi_evidence if e.startswith("load step"))
    assert steps_after_ramp >= 5, f"only {steps_after_ramp} observable load steps after the ramp"
    r_est = states[-1].battery_resistance_mohm
    r_true = gts[-1].electrical.battery_r_int_ohm * 1000.0
    assert r_est.valid
    assert abs(r_est.value - r_true) / r_true < 0.20, f"R_int {r_est.value:.1f} vs simulated {r_true:.1f} mOhm"
    assert states[-1].battery_status.value == "CRITICAL"  # 4x nominal > alarm ratio 2.5
    assert any("Battery internal resistance" in a.message for a in electrical_advisories(states[-1]))


def test_no_load_steps_r_int_invalid_not_a_number() -> None:
    s = get_settings().model_copy(deep=True)
    s.simulator.electrical.load_step_mean_interval_s = 0.0
    _, _, states = _run(60.0, None, IDLE, settings=s)
    r = states[-1].battery_resistance_mohm
    assert r.valid is False and r.value is None and r.quality == 0.0
    assert "insufficient load steps" in r.fault_flag
    assert all(st.r_int_step_count == 0 for st in states)


# ---------------------------------------------------------------------------
# Not instrumented -> excluded from health
# ---------------------------------------------------------------------------

def test_electrical_absent_is_invalid_and_excluded_from_health(nominal) -> None:
    recs, _, _ = nominal
    stripped = recs[-1].model_copy(update={
        "bus_v_counts": None, "alt_i_counts": None, "batt_i_counts": None,
        "bus_v_burst_counts": None, "bus_v_burst_fs_hz": None})
    norm = convert_raw_to_engineering_state(stripped)
    for ch in ("voltage", "current", "battery_current"):
        assert getattr(norm, ch).valid is False and "not instrumented" in getattr(norm, ch).fault_flag
    state = ElectricalModel().evaluate(norm)
    assert state.ehi.valid is False and state.ehi.value is None and state.ehi_coverage == 0.0
    for name in ("charging_residual_v", "battery_resistance_mohm", "voltage_ripple_pct"):
        assert getattr(state, name).valid is False and getattr(state, name).value is None

    engine = HealthSupervisionEngine()
    hs = engine.evaluate_health(elec_state=state)
    assert "electrical_health" not in hs.component_health
    cov = hs.evidence["coverage"]
    assert cov["excluded"] == ["electrical_health"]
    assert cov["coverage"] == pytest.approx(1.0 - get_settings().health.weight_electrical)


def test_electrical_present_enters_health(nominal) -> None:
    _, _, states = nominal
    hs = HealthSupervisionEngine().evaluate_health(elec_state=states[-1])
    assert hs.component_health["electrical_health"] == pytest.approx(states[-1].ehi.value)
    assert hs.evidence["coverage"]["excluded"] == []


def test_health_weights_sum_to_one() -> None:
    h = get_settings().health
    weights = [h.weight_thermal, h.weight_lubrication, h.weight_vibration, h.weight_combustion,
               h.weight_performance, h.weight_anomaly_fault, h.weight_electrical]
    assert math.isclose(sum(weights), 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Ripple frequency scaling and M-05
# ---------------------------------------------------------------------------

def test_simulated_ripple_frequency_below_burst_nyquist_over_rpm_range() -> None:
    """Reported table: ripple fundamental 6 x f_e with the SIMULATION placeholder
    pole count; fs must be >= 2.5 x the maximum (Prompt 12 note)."""
    sim = ElectricalSimulator()
    f = {rpm: sim.ripple_fundamental_hz(rpm) for rpm in (2000.0, 3000.0, 4000.0, 5000.0, 5800.0)}
    assert f[2000.0] == pytest.approx(1200.0) and f[5800.0] == pytest.approx(3480.0)
    fs = sim.cfg.bus_v_burst_fs_hz
    assert fs >= 2.5 * max(f.values())
    assert band_is_representable(max(f.values()), fs)
    assert get_settings().electrical.bus_v_burst_fs_nominal_hz == fs


@pytest.mark.parametrize("rpm", [2500.0, 4000.0, 5800.0])
def test_ripple_order_constant_across_rpm(rpm: float) -> None:
    profile = [{"start_time_s": 0.0, "end_time_s": 1e6, "rpm": rpm, "map_pa": 100000.0}]
    _, _, states = _run(3.0, None, profile)
    s = states[-1]
    poles = get_settings().simulator.electrical.generator_poles_placeholder
    assert s.ripple_order.valid
    assert s.ripple_order.value == pytest.approx(3 * poles, abs=0.25)  # 2 x 3 phases x pole pairs


def test_dominant_frequency_valid_only_with_known_poles() -> None:
    s = get_settings().model_copy(deep=True)
    s.engine.generator_poles = s.simulator.electrical.generator_poles_placeholder  # test-only assumption
    _, _, states = _run(3.0, None, None, settings=s)
    f = states[-1].ripple_dominant_frequency_hz
    assert f.valid
    assert f.value == pytest.approx(6 * (12 / 2) * 4000.0 / 60.0, abs=10.0)


def test_ripple_invalid_when_band_exceeds_burst_nyquist(nominal) -> None:
    recs, _, _ = nominal
    low_fs = recs[-1].model_copy(update={"bus_v_burst_fs_hz": 5120.0})  # Nyquist 2560 < band 4000
    state = ElectricalModel().evaluate(convert_raw_to_engineering_state(low_fs))
    assert state.voltage_ripple_pct.valid is False and state.voltage_ripple_pct.value is None
    assert "Nyquist" in state.voltage_ripple_pct.fault_flag
    assert state.ripple_dominant_frequency_hz.valid is False


def test_config_rejects_ripple_band_above_nyquist() -> None:
    elec = get_settings().electrical.model_copy(update={"ripple_band_high_hz": 6000.0})
    with pytest.raises(ConfigurationError):
        _validate_ripple_band(elec, ConfigurationError)
    _validate_ripple_band(get_settings().electrical, ConfigurationError)


# ---------------------------------------------------------------------------
# batt_i_counts transport
# ---------------------------------------------------------------------------

async def test_batt_i_counts_on_can_0x103_fourth_word() -> None:
    from src.l1_data.adapters.can_transport import MockCANTransport, encode_can_frame_0x103
    from src.l1_data.adapters.live_telemetry_adapter import LiveTelemetryAdapter

    transport = MockCANTransport()
    adapter = LiveTelemetryAdapter(transport=transport)
    await adapter.connect()
    await transport.send_frame(encode_can_frame_0x103(2100, 2800, 1500, 2150))
    rec = await adapter.read_next()
    await adapter.disconnect()
    assert (rec.fuel_press_counts, rec.bus_v_counts, rec.alt_i_counts, rec.batt_i_counts) == (2100, 2800, 1500, 2150)


def test_simulated_record_carries_electrical_raw_counts_only(nominal) -> None:
    recs, _, _ = nominal
    r = recs[-1]
    assert all(isinstance(getattr(r, f), int) for f in ("bus_v_counts", "alt_i_counts", "batt_i_counts"))
    assert len(r.bus_v_burst_counts) == 2048 and all(isinstance(c, int) for c in r.bus_v_burst_counts[:8])
    assert r.bus_v_burst_fs_hz == 10240.0


def test_csv_replay_reads_batt_i_counts_and_missing_as_none(tmp_path) -> None:
    import csv

    from src.l1_data.adapters.csv_replay_adapter import CSVReplayAdapter

    path = tmp_path / "p12.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_number", "bus_v_counts", "alt_i_counts", "batt_i_counts"])
        w.writerow([1, 2800, 1500, 2150])
        w.writerow([2, 2800, 1500, ""])
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    adapter = CSVReplayAdapter(path)
    assert adapter._row_to_raw_record(rows[0], 1).batt_i_counts == 2150
    assert adapter._row_to_raw_record(rows[1], 2).batt_i_counts is None


# ---------------------------------------------------------------------------
# Pipeline and API wiring
# ---------------------------------------------------------------------------

def test_pipeline_carries_electrical_state_into_api_summary() -> None:
    from src.api.v1.engine_health import ELECTRICAL_FIELDS, electrical_summary
    from src.l1_data.simulator.replay_and_whatif import PipelineReplayAdapter

    recs, _, _ = _run(3.0, _charging("output_collapse", onset=0.0))
    results = PipelineReplayAdapter().process_sequence(recs)
    accepted = [r for r in results if r.accepted]
    assert accepted and all(r.electrical_state is not None for r in accepted)
    summary = electrical_summary(accepted[-1].electrical_state)
    assert set(ELECTRICAL_FIELDS) <= set(summary)
    assert summary["bus_voltage_v"] is not None
    # Output collapse above cut-in: alternator current ~0, battery supplies the load
    assert summary["alternator_current_a"] < 1.0 and summary["battery_current_a"] > 0.0
    # Probabilistic wording (SRD-FUN-142), not a diagnosis
    assert any("consistent with" in m for m in summary["electrical_advisories"])
    # Invalid values are None with a reason, never a number
    for name, value in summary.items():
        if name in ELECTRICAL_FIELDS and value is None:
            assert summary["electrical_invalid_reasons"][name]
    assert electrical_summary(None)["electrical_status"] == "INVALID"
