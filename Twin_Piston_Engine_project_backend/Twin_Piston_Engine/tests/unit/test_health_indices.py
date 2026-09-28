"""
Health indices: CSI (M-09), LHI and VHI (M-10).

    CSI = EGT std/mean + crank CoV + |CHT slope - expected CHT slope| / ref
    LHI = expectation-normalised pressure x over-temp derate (viscosity
          excluded: no measured channel; weights re-normalised over valid
          components)
    VHI = (RMS/ref) x (crest/ref) x (envelope/ref), rpm-dependent references

Strict xfails record measured values where the simulator or the current
processing does not meet the criterion (OI-5, OI-9, OI-10). Thresholds are
not tuned.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.core.provenance import FaultClass
from src.l1_data.signal_record import SampleBurst
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
)
from src.l2_digital_twin.lubrication_model import VISCOSITY_NOT_MEASURED, LubricationModel
from src.l2_digital_twin.misfire_classifier import MisfireDetector
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, operating_point_from_record
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import evaluate_digital_twin
from src.l2_digital_twin.egt_diagnostics import evaluate_egt_diagnostics
from src.l2_digital_twin.vibration_processor import VibrationProcessor, evaluate_vibration_processor
from tests.unit.test_module6 import _make_dummy_normalized_record as make_record

T0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
GRID_RPM = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 5800.0)
GRID_MAP = (80000.0, 110000.0, 135000.0)


def _sim(rpm: float, fault: FaultClass | None = None, severity: float = 0.0, map_pa: float = 110000.0):
    scenario = None
    if fault is not None:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0, duration_s=10.0)
    gt = ForwardPhysicsModel(seed=7).compute_ground_truth(time_s=1.0, rpm=rpm, map_pa=map_pa, fault_scenario=scenario)
    return convert_raw_to_engineering_state(SensorForwardModel(seed=7).convert_to_raw_record(gt, scenario))


# ==========================================================================
# CSI (M-09)
# ==========================================================================

def _steady_crank_burst(rpm: float) -> SampleBurst:
    period = 60.0e6 / rpm
    return SampleBurst(samples=tuple(period * (1.0 + 1e-4 * ((-1) ** i)) for i in range(32)),
                       unit="us", valid=True, quality=1.0)


def _synthetic_sequence(n: int, rpm, map_pa, cht_fn):
    """Records at 1 Hz; cht_fn(k, record) gives the CHT for step k."""
    records = []
    for k in range(n):
        rec = make_record(rpm_val=rpm(k), map_pa=map_pa(k), timestamp=T0 + timedelta(seconds=k))
        rec = rec.model_copy(update={"crank_period_burst": _steady_crank_burst(rpm(k))})
        cht = cht_fn(k, rec)
        records.append(rec.model_copy(update={"cht_cyl_1": rec.cht_cyl_1.model_copy(update={"value": cht})}))
    return records


def _run_csi(records):
    det = MisfireDetector()
    for rec in records:
        state, res = det.evaluate(rec, evaluate_digital_twin(rec), None, evaluate_vibration_processor(rec)[1])
    return state, res


CLIMB_S = 30
EXPECT = HealthyExpectationModel()


def _climb_records():
    """Power ramp 4000->5800 rpm, 80->135 kPa over 30 s; CHT follows its
    healthy expectation (a healthy engine that matches the expectation)."""
    rpm = lambda k: 4000.0 + 1800.0 * k / CLIMB_S
    map_pa = lambda k: 80000.0 + 55000.0 * k / CLIMB_S
    return _synthetic_sequence(CLIMB_S + 1, rpm, map_pa,
                               lambda k, rec: EXPECT.expected_cht_k(operating_point_from_record(rec)))


def test_nominal_climb_csi_normal() -> None:
    """During the power transient the thermal term is excluded (invalid,
    coverage 2/3), not zero; the expected CHT (uncalibrated, OI-5) is unused."""
    state, res = _run_csi(_climb_records())
    assert res.csi_terms["cht_slope_k_s"] > 0.4  # CHT really is rising
    assert "thermal_slope" not in res.csi_terms
    assert res.csi_terms["coverage"] == pytest.approx(2.0 / 3.0)
    assert "transient" in res.csi_reason
    assert state.csi_value.valid
    assert state.csi_value.value < 0.15
    assert state.csi_band == "NORMAL"


def test_same_climb_with_raw_slope_would_alarm() -> None:
    """Why the slope residual is needed: on this 30 s climb the raw-slope form
    gives 2.82 ALARM (the simulator's 20 s climb gives 1.15), vs < 0.15 above."""
    _, res = _run_csi(_climb_records())
    t = res.csi_terms
    raw_csi = t["egt_dispersion"] + t["speed_variability"] + abs(t["cht_slope_k_s"]) / 0.5
    assert raw_csi > 0.35  # ALARM band


def test_cht_runaway_at_constant_power_csi_alarm() -> None:
    # 35 records: the 30 s CHT window must be full before the thermal term counts (OI-13)
    records = _synthetic_sequence(35, lambda k: 5000.0, lambda k: 110000.0,
                                  lambda k, rec: 400.0 + 0.5 * k)  # +0.5 K/s, power constant
    state, res = _run_csi(records)
    assert state.csi_value.value > 0.35
    assert state.csi_band == "ALARM"
    assert state.csi_dominant_term == "thermal_slope"
    assert res.csi_terms["coverage"] == pytest.approx(1.0)  # steady: thermal term active


def test_csi_invalid_until_window_fills_and_without_crank_burst() -> None:
    rec = _climb_records()[0]
    state, res = MisfireDetector().evaluate(rec, None, None, evaluate_vibration_processor(rec)[1])
    assert state.csi_value.valid is False and state.csi_band == "UNKNOWN"
    no_burst = [r.model_copy(update={"crank_period_burst": SampleBurst()}) for r in _climb_records()]
    state, res = _run_csi(no_burst)
    assert state.csi_value.valid is False and "crank_cov" in res.csi_reason


def _sim_sequence_csi(duration_s: int, rpm0, rpm1, map0, map1, fault=None, severity=0.0):
    det, sim, sfm = MisfireDetector(), ForwardPhysicsModel(seed=7), SensorForwardModel(seed=7)
    scenario = None
    if fault is not None:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0, duration_s=1e4)
    for k in range(duration_s + 1):
        f = k / duration_s
        gt = sim.compute_ground_truth(float(k), sequence_number=k + 1, rpm=rpm0 + (rpm1 - rpm0) * f,
                                      map_pa=map0 + (map1 - map0) * f, timestamp=T0 + timedelta(seconds=k),
                                      fault_scenario=scenario)
        norm = convert_raw_to_engineering_state(sfm.convert_to_raw_record(gt, scenario))
        state, res = det.evaluate(norm, evaluate_digital_twin(norm), evaluate_egt_diagnostics(norm)[1],
                                  evaluate_vibration_processor(norm)[1])
    return state, res


def test_simulator_climb_csi_normal() -> None:
    """Was a strict xfail (CSI 2.09 with the expected-CHT slope residual).
    40 s ramp, window full: thermal term excluded as a transient (dP/dt
    1.37 kW/s > 0.5): CSI 0.0075, coverage 2/3."""
    state, res = _sim_sequence_csi(40, 5000.0, 5800.0, 80000.0, 135000.0)
    assert state.csi_band not in ("WARNING", "ALARM")
    assert "thermal_slope" not in res.csi_terms and "transient" in res.csi_reason


def test_simulator_steady_csi_normal() -> None:
    """Was a strict xfail (CSI 0.198 with a 10 s window). 30 s least-squares
    window: noise-only slope std 0.0709 -> 0.0145 K/s; measured CSI 0.011."""
    state, res = _sim_sequence_csi(40, 5500.0, 5500.0, 110000.0, 110000.0)
    assert res.csi_terms["coverage"] == pytest.approx(1.0)
    assert state.csi_value.value < 0.15


def test_misfire_raises_csi_via_crank_term() -> None:
    """MISFIRE 1.0, steady 4000 rpm: CSI 0.0107 -> 0.0207 (+0.010, the crank
    CoV term); it stays NORMAL because M-09's speed term is CoV/100."""
    _, nominal = _sim_sequence_csi(40, 4000.0, 4000.0, 110000.0, 110000.0)
    _, misfire = _sim_sequence_csi(40, 4000.0, 4000.0, 110000.0, 110000.0, FaultClass.MISFIRE, 1.0)
    assert misfire.csi_value > nominal.csi_value
    assert misfire.csi_terms["speed_variability"] > 10 * nominal.csi_terms["speed_variability"]
    assert misfire.csi_value - nominal.csi_value == pytest.approx(
        misfire.csi_terms["speed_variability"] - nominal.csi_terms["speed_variability"], abs=2e-3)


# ==========================================================================
# LHI (M-10)
# ==========================================================================

def _healthy_oil_record(temp_c: float, rpm: float = 4000.0):
    """Oil pressure on the healthy hydrodynamic law at this temperature and rpm.
    (The only healthy oil-pressure model in the repo is the one the expectation
    uses, so the pressure factor is 1 by construction; the point is that it
    stays 1 across oil temperature, unlike a fixed-nominal normalisation.)"""
    rec = make_record(rpm_val=rpm)
    rec = rec.model_copy(update={"oil_temp": rec.oil_temp.model_copy(update={"value": temp_c + 273.15})})
    p = EXPECT.expected_oil_pressure_pa(operating_point_from_record(rec), temp_c + 273.15)
    return rec.model_copy(update={"oil_pressure": rec.oil_pressure.model_copy(update={"value": p})}), p


@pytest.mark.parametrize("temp_c", [35.0, 85.0, 115.0])
def test_healthy_oil_lhi_near_one(temp_c: float) -> None:
    rec, p = _healthy_oil_record(temp_c)
    state, res = LubricationModel().evaluate(rec)
    assert 0.95 <= state.lhi.value <= 1.05
    assert state.lhi_band == "NORMAL"
    # A fixed-nominal (4 bar) normalisation would not be near 1 at cold/hot oil.
    if temp_c != 85.0:
        assert abs(p / 400000.0 - 1.0) > 0.05


def test_viscosity_excluded_and_reported() -> None:
    rec, _ = _healthy_oil_record(85.0)
    state, res = LubricationModel().evaluate(rec)
    visc = res.lhi_components["viscosity"]
    assert visc["valid"] is False and visc["weight"] == 0.0
    assert visc["reason"] == VISCOSITY_NOT_MEASURED
    assert visc["derived_value_pa_s"] > 0.0
    assert state.lhi_coverage == pytest.approx(2.0 / 3.0)
    assert any("viscosity excluded" in e for e in state.lhi_evidence)


def test_lhi_weights_renormalised_over_valid_components() -> None:
    rec, p = _healthy_oil_record(85.0)
    rec = rec.model_copy(update={"oil_pressure": rec.oil_pressure.model_copy(update={"value": 0.8 * p})})
    state, _ = LubricationModel().evaluate(rec)
    # Pressure factor 0.8, over-temp 1.0; W_all/W_valid = 3/2 -> 0.8 ** 1.5
    assert state.lhi.value == pytest.approx(0.8 ** 1.5, rel=1e-9)


def test_over_temperature_derate_only_above_130c() -> None:
    at_limit, _ = _healthy_oil_record(130.0)
    above, _ = _healthy_oil_record(140.0)
    assert LubricationModel().evaluate(at_limit)[0].lhi.value == pytest.approx(1.0)
    assert LubricationModel().evaluate(above)[0].lhi.value == pytest.approx(0.5 ** 1.5)


@pytest.mark.parametrize("rpm", [2500.0, 4000.0, 5500.0])
def test_oil_degradation_lhi_below_0_8(rpm: float) -> None:
    """Full-severity OIL_DEGRADATION: measured 0.225 / 0.372 / 0.517
    (severity 0.5: 0.604 / 0.699 / 0.816)."""
    state, _ = LubricationModel().evaluate(_sim(rpm, FaultClass.OIL_DEGRADATION, 1.0))
    assert state.lhi.value < 0.8


# ==========================================================================
# VHI (M-10)
# ==========================================================================

def test_healthy_vhi_within_band_across_grid() -> None:
    """Was a strict xfail (0.609-1.877, M-04 leakage). With the 650-1000 Hz
    tapered envelope band: 0.995-1.003 over 2000-5800 rpm x 80-135 kPa."""
    vp = VibrationProcessor()
    values = {(r, m): vp.process_record(_sim(r, map_pa=m))[0].vibration_health_index.value
              for r in GRID_RPM for m in GRID_MAP}
    assert all(0.8 <= v <= 1.2 for v in values.values()), values


def test_healthy_vhi_at_reference_table_points() -> None:
    vp = VibrationProcessor()
    for rpm in (2000.0, 3000.0, 4000.0, 5000.0, 5800.0):
        assert vp.process_record(_sim(rpm))[0].vibration_health_index.value == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("rpm", [2500.0, 4000.0, 5500.0])
def test_bearing_wear_vhi_alarm_while_rms_normal(rpm: float) -> None:
    """Early BEARING_WEAR (0.03): VHI 32.0 / 21.1 / 27.1 while RMS/ref is
    1.14 / 1.12 / 1.10 (NORMAL < 1.2). At 0.05 RMS/ref reaches 1.17-1.24."""
    vp = VibrationProcessor()
    state, res = vp.process_record(_sim(rpm, FaultClass.BEARING_WEAR, 0.03))
    rms_only = state.overall_rms_m_s2.value / vp.vhi_references(rpm)[0]
    assert state.vibration_health_index.value > 1.8
    assert rms_only < 1.2  # overall RMS alone stays in its NORMAL band
    assert state.vhi_dominant_factor == "envelope"


def test_misfire_confirmed_by_gate_and_raises_csi() -> None:
    """MISFIRE 1.0: the gate CONFIRMS it and CSI rises (crank term). VHI is not
    required to move: it has no half-order or kurtosis term."""
    _, nominal = _sim_sequence_csi(40, 4000.0, 4000.0, 110000.0, 110000.0)
    state, misfire = _sim_sequence_csi(40, 4000.0, 4000.0, 110000.0, 110000.0, FaultClass.MISFIRE, 1.0)
    assert misfire.gate_verdict == "CONFIRMED"
    assert any(state.misfire_detected)
    assert misfire.csi_value > nominal.csi_value


def test_vhi_invalid_without_burst() -> None:
    state, _ = VibrationProcessor().process_record(make_record())
    assert state.vibration_health_index.valid is False
    assert state.vhi_band == "UNKNOWN"
