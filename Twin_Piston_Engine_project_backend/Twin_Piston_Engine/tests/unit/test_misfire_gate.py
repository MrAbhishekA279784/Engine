"""
Dual-channel misfire AND gate (M-06) in MisfireDetector.

    Channel A (torsional):  CoV of the per-revolution crank periods in the burst.
    Channel B (structural): half-order (0.5X) energy fraction from vibration.

CONFIRMED needs BOTH at/above threshold; one channel -> UNCONFIRMED (and the
channel is named); a missing input -> INVALID, never NORMAL.
"""

import pytest

from src.core.provenance import DiagnosticStatus, FaultClass
from src.l1_data.signal_record import SampleBurst
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
)
from src.l2_digital_twin.egt_diagnostics import evaluate_egt_diagnostics
from src.l2_digital_twin.misfire_classifier import MisfireDetector
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import evaluate_digital_twin
from src.l2_digital_twin.vibration_processor import evaluate_vibration_processor

RPM = 4000.0
PERIOD_US = 60.0e6 / RPM


def _norm(fault: FaultClass | None = None, severity: float = 0.0):
    scenario = None
    if fault is not None:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0, duration_s=10.0)
    gt = ForwardPhysicsModel(seed=7).compute_ground_truth(time_s=1.0, rpm=RPM, map_pa=110000.0, fault_scenario=scenario)
    return convert_raw_to_engineering_state(SensorForwardModel(seed=7).convert_to_raw_record(gt, scenario))


def _evaluate(norm, vib=None):
    vib_res = evaluate_vibration_processor(norm)[1] if vib is None else vib
    egt_res = evaluate_egt_diagnostics(norm)[1]
    return MisfireDetector().evaluate(norm, evaluate_digital_twin(norm), egt_res, vib_res)


def _with_crank_burst(norm, alternation: float):
    """Replace the crank burst: every second revolution slower by `alternation`."""
    periods = tuple(PERIOD_US * (1.0 + alternation if i % 2 == 0 else 1.0) for i in range(32))
    burst = SampleBurst(samples=periods, unit="us", valid=True, quality=1.0)
    return norm.model_copy(update={"crank_period_burst": burst})


def _with_half_order(vib_res, fraction: float):
    return vib_res.model_copy(update={"half_order_fraction": fraction})


def _with_egt_drop(norm, cylinder: int, drop_k: float):
    field = f"egt_cyl_{cylinder}"
    ch = getattr(norm, field)
    return norm.model_copy(update={field: ch.model_copy(update={"value": ch.value - drop_k})})


# --------------------------------------------------------------------------
# Simulator scenarios
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("severity", "detected"), [
    (0.3, False),   # CoV 0.306 %: CONFIRMED, but EGT evidence too weak to localise a cylinder
    (0.5, True),    # CoV 0.508 %
    (0.8, True),    # CoV 0.808 %
    (1.0, True),    # CoV 1.008 %
])
def test_simulator_misfire_is_confirmed(severity: float, detected: bool) -> None:
    """Channel A limit = max(0.30 % floor, 3 x healthy CoV); no healthy CoV
    configured, so 0.30 % applies (OI-7). Half-order 0.41-0.66 >= 0.03."""
    state, res = _evaluate(_norm(FaultClass.MISFIRE, severity))
    assert res.gate_verdict == "CONFIRMED"
    assert res.crank_cov_limit_pct == pytest.approx(0.30)
    assert state.misfire_detected[0] is detected
    assert state.misfire_detected[1:] == [False, False, False]


def test_bearing_wear_is_not_confirmed() -> None:
    """High vibration RMS but no 0.5X energy: the false positive the gate prevents."""
    state, res = _evaluate(_norm(FaultClass.BEARING_WEAR, 0.8))
    assert state.misfire_verdict in ("NORMAL", "UNCONFIRMED")
    assert state.misfire_verdict != "CONFIRMED"
    assert state.misfire_detected == [False, False, False, False]
    assert state.half_order_fraction.value < 0.03


def test_nominal_is_normal() -> None:
    state, res = _evaluate(_norm())
    assert state.misfire_verdict == "NORMAL"
    assert state.overall_combustion_status == DiagnosticStatus.NORMAL


# --------------------------------------------------------------------------
# Gate logic
# --------------------------------------------------------------------------

def test_crank_cov_high_half_order_normal_is_unconfirmed_channel_a() -> None:
    norm = _with_crank_burst(_norm(), alternation=0.06)  # CoV ~3 %
    state, res = _evaluate(norm)
    assert res.crank_cov_pct >= 2.0 and res.half_order_fraction < 0.03
    assert state.misfire_verdict == "UNCONFIRMED"
    assert state.misfire_gate_channel == "crank_speed_variability"
    assert "crank_speed_variability" in state.misfire_gate_evidence
    assert state.misfire_detected == [False, False, False, False]
    assert state.overall_combustion_status == DiagnosticStatus.WARNING


def test_and_gate_not_weighted_sum() -> None:
    """A huge half-order fraction cannot compensate for a quiet crank channel."""
    norm = _norm()
    vib = _with_half_order(evaluate_vibration_processor(norm)[1], 0.95)
    state, res = _evaluate(norm, vib)
    assert res.gate_verdict == "UNCONFIRMED"
    assert res.gate_triggering_channel == "half_order_energy"
    assert not any(state.misfire_detected)


@pytest.mark.parametrize(("alternation", "half", "status"), [
    (0.05, 0.05, DiagnosticStatus.WARNING),    # half-order 0.05 < 2 x 0.03: not both >= 2x
    (0.10, 0.20, DiagnosticStatus.CRITICAL),   # CoV ~4.8 %, 0.20: both >= 2x limit
])
def test_both_channels_confirm_and_existing_logic_localises(alternation, half, status) -> None:
    norm = _with_egt_drop(_with_crank_burst(_norm(), alternation), cylinder=3, drop_k=120.0)
    vib = _with_half_order(evaluate_vibration_processor(norm)[1], half)
    state, res = _evaluate(norm, vib)
    assert res.gate_verdict == "CONFIRMED"
    assert state.misfire_detected == [False, False, True, False]
    assert state.misfire_status[2] == status
    assert state.overall_combustion_status == status


def test_missing_crank_burst_is_invalid_not_normal() -> None:
    norm = _norm().model_copy(update={"crank_period_burst": SampleBurst()})
    state, res = _evaluate(norm)
    assert state.misfire_verdict == "INVALID"
    assert state.overall_combustion_status != DiagnosticStatus.NORMAL
    assert state.crank_cov_pct.valid is False and state.crank_cov_pct.value is None
    assert "channel A" in state.misfire_gate_evidence
    assert state.misfire_detected == [False, False, False, False]


def test_missing_vibration_result_is_invalid_not_normal() -> None:
    norm = _norm()
    state, _ = MisfireDetector().evaluate(norm, evaluate_digital_twin(norm), evaluate_egt_diagnostics(norm)[1], None)
    assert state.misfire_verdict == "INVALID"
    assert state.overall_combustion_status != DiagnosticStatus.NORMAL
    assert "channel B" in state.misfire_gate_evidence
