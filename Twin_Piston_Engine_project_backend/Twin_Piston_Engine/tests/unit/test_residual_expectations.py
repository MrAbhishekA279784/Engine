"""
CHT, fuel-flow and per-cylinder EGT healthy expectations (M-11) and the
guard that no expectation reads the channel it predicts.

HealthyBaselineConfig holds the model defaults and is NOT calibrated to the
simulator. The two disagree (docs/OPEN_ITEMS.md, OI-5), so the nominal-residual
tests are strict xfails that record the measured residuals; they will XPASS,
and must be revisited, once the expectations are fitted to healthy-flight data.
"""

import inspect

import pytest

from src.core.provenance import FaultClass
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
)
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, ResidualEngine
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import evaluate_digital_twin
from tests.unit.test_module6 import _make_dummy_normalized_record as make_record

OPERATING_POINTS = [(2500.0, 90000.0), (4000.0, 110000.0), (5500.0, 135000.0)]


def _xfail_points(measured: dict[tuple[float, float], str]) -> list:
    """One strict xfail per operating point, carrying its measured residuals."""
    return [
        pytest.param(rpm, map_pa, marks=pytest.mark.xfail(
            strict=True,
            reason=f"OI-5: expectations disagree with the simulator at {rpm:.0f} rpm / "
                   f"{map_pa / 1000:.0f} kPa; measured {measured[(rpm, map_pa)]}",
        ))
        for rpm, map_pa in OPERATING_POINTS
    ]


# Measured with sensor noise on (seed 7), HealthyBaselineConfig defaults.
MEASURED_CHT_FUEL = {
    (2500.0, 90000.0): "CHT -35.7 K (-8.6 %), fuel -32.7 %",
    (4000.0, 110000.0): "CHT -47.5 K (-11.0 %), fuel -30.9 %",
    (5500.0, 135000.0): "CHT -65.5 K (-14.3 %), fuel -29.0 %",
}
MEASURED_EGT = {
    (2500.0, 90000.0): "EGT cyl1-4 -37.4 / -42.0 / -49.7 / -53.4 K",
    (4000.0, 110000.0): "EGT cyl1-4 -134.9 / -139.6 / -147.3 / -151.0 K",
    (5500.0, 135000.0): "EGT cyl1-4 -239.9 / -244.6 / -252.3 / -256.0 K",
}
EGT_KEYS = ("egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4")

# Each expectation method -> tokens naming the channel it predicts. A method
# may take the operating point and OTHER channels, never its own.
PREDICTED_CHANNEL_TOKENS = {
    "expected_egt_k": ("egt",),
    "expected_egt_per_cylinder_k": ("egt",),
    "expected_cht_k": ("cht",),
    "expected_fuel_flow_kg_s": ("fuel", "power"),  # measured power must not be an input either
    "expected_oil_pressure_pa": ("oil_p", "oil_pressure"),
    "expected_oil_temp_k": ("oil_temp", "oil_t"),
    "expected_vibration_rms_m_s2": ("vib",),
    "expected_brake_power_kw": ("power", "fuel"),
    # Prompt 13: the rail expectation may take the injector demand, never the rail pressure
    "expected_fuel_rail_dp_pa": ("rail", "fuel_press", "pressure"),
    "expected_fuel_delivery_ratio": ("delivery", "fuel_flow"),
    "expected_injector_flow_ratio": ("flow_ratio", "egt"),
    "expected_coolant_temp_k": ("coolant",),  # Prompt 14
}


def _simulate(rpm: float, map_pa: float, fault: FaultClass | None = None, severity: float = 0.0):
    scenario = None
    if fault is not None:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0, duration_s=10.0)
    gt = ForwardPhysicsModel(seed=7).compute_ground_truth(time_s=1.0, rpm=rpm, map_pa=map_pa, fault_scenario=scenario)
    norm = convert_raw_to_engineering_state(SensorForwardModel(seed=7).convert_to_raw_record(gt, scenario))
    return ResidualEngine().evaluate(norm, evaluate_digital_twin(norm))


# --------------------------------------------------------------------------
# Residuals on the simulator
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("rpm", "map_pa"), _xfail_points(MEASURED_CHT_FUEL))
def test_nominal_cht_and_fuel_residuals_below_5_percent(rpm: float, map_pa: float) -> None:
    """Criterion: |CHT| and |fuel| residual each below 5 % on the nominal simulator."""
    _, exp = _simulate(rpm, map_pa)
    for key in ("cht", "fuel_flow"):
        res = exp.residuals[key]
        assert res.valid
        assert abs(res.raw_residual) / res.expected < 0.05, (key, res)


@pytest.mark.parametrize(("rpm", "map_pa"), _xfail_points(MEASURED_EGT))
def test_nominal_per_cylinder_egt_residuals_below_10_k(rpm: float, map_pa: float) -> None:
    """Criterion: each per-cylinder EGT residual below 10 K on the nominal simulator."""
    _, exp = _simulate(rpm, map_pa)
    for key in EGT_KEYS:
        assert exp.residuals[key].valid
        assert abs(exp.residuals[key].raw_residual) < 10.0, (key, exp.residuals[key])


@pytest.mark.xfail(strict=True, reason=(
    "OI-22: COOLING_FAULT is a physical radiator blockage since Prompt 14 (was a +35 K CHT offset). "
    "At 4000 rpm / 110 kPa, severity 1.0 (80 % of the air path blocked), steady state: CHT +14.4 K "
    "(residual -47.5 -> -33.1 K) because the thermostat opens fully and absorbs part of the loss."))
def test_cooling_fault_raises_cht_residual_by_more_than_20_k() -> None:
    """Fault minus nominal, same operating point. With the uncalibrated baseline
    the absolute residual stays negative (nominal -47.5 K), so an absolute > 20 K
    check would only pass with a baseline fitted to the simulator."""
    nominal = _simulate(4000.0, 110000.0)[1].residuals["cht"].raw_residual
    fault = _simulate(4000.0, 110000.0, FaultClass.COOLING_FAULT, 1.0)[1].residuals["cht"].raw_residual
    assert fault - nominal > 20.0


def test_residual_state_carries_new_channels() -> None:
    state, exp = _simulate(4000.0, 110000.0)
    for key in ("cht", "fuel_flow", *EGT_KEYS):
        assert key in state.residuals and state.residuals[key].valid
        assert state.residuals[key].provenance.value == "DERIVED"


def test_per_cylinder_expectations_use_offsets() -> None:
    model = HealthyExpectationModel()
    op = _simulate(4000.0, 110000.0)[1].operating_point
    common = model.expected_egt_k(op)
    per_cyl = model.expected_egt_per_cylinder_k(op)
    assert per_cyl == pytest.approx(tuple(common + o for o in (-8.0, -3.0, 4.0, 7.0)))


def test_invalid_cht_channel_gives_invalid_residual_not_zero() -> None:
    base = make_record(rpm_val=4000.0)
    rec = base.model_copy(update={"cht_cyl_1": base.cht_cyl_1.model_copy(update={"valid": False, "quality": 0.0})})
    state, _ = ResidualEngine().evaluate(rec)
    assert state.residuals["cht"].valid is False
    assert state.residuals["cht"].value is None


# --------------------------------------------------------------------------
# Guard: no expectation may read the channel it predicts
# --------------------------------------------------------------------------

def test_every_expectation_method_is_registered() -> None:
    methods = {n for n, _ in inspect.getmembers(HealthyExpectationModel, inspect.isfunction) if n.startswith("expected_")}
    assert methods == set(PREDICTED_CHANNEL_TOKENS), "register new expected_* methods in PREDICTED_CHANNEL_TOKENS"


@pytest.mark.parametrize("method_name", sorted(PREDICTED_CHANNEL_TOKENS))
def test_expectation_signature_excludes_predicted_channel(method_name: str) -> None:
    params = [p for p in inspect.signature(getattr(HealthyExpectationModel, method_name)).parameters if p != "self"]
    assert params[0] == "op", "the operating point must be the first input"
    for param in params:
        for token in PREDICTED_CHANNEL_TOKENS[method_name]:
            assert token not in param.lower(), f"{method_name} takes '{param}', which names its own channel"


# (expected_values key, NormalizedSignalRecord field, new measured value)
MEASURED_CHANNELS = [
    ("cht", "cht_cyl_1", 520.0),
    ("fuel_flow", "fuel_flow", 0.02),
    ("egt_cyl1", "egt_cyl_1", 1300.0),
    ("egt_cyl4", "egt_cyl_4", 700.0),
    ("oil_temp", "oil_temp", 400.0),
    ("oil_pressure", "oil_pressure", 150000.0),
    ("vibration_rms", "vibration_rms", 40.0),
]


@pytest.mark.parametrize(("key", "field", "new_value"), MEASURED_CHANNELS)
def test_changing_measured_channel_leaves_its_expectation_unchanged(key: str, field: str, new_value: float) -> None:
    base = make_record(rpm_val=4000.0)
    changed = base.model_copy(update={field: getattr(base, field).model_copy(update={"value": new_value})})
    engine = ResidualEngine()
    exp_base = engine.evaluate(base)[1]
    exp_changed = engine.evaluate(changed)[1]
    assert exp_changed.expected_values[key] == exp_base.expected_values[key]
    assert exp_changed.residuals[key].observed == pytest.approx(new_value)


def test_fuel_expectation_ignores_measured_brake_power() -> None:
    """The degradation case: measured power falls, fuel expectation must not."""
    rec = make_record(rpm_val=4000.0)
    derived = evaluate_digital_twin(rec)
    degraded = derived.model_copy(update={
        "brake_power_kw": derived.brake_power_kw.model_copy(update={"value": derived.brake_power_kw.value * 0.6})
    })
    engine = ResidualEngine()
    assert engine.evaluate(rec, degraded)[1].expected_values["fuel_flow"] == \
        engine.evaluate(rec, derived)[1].expected_values["fuel_flow"]
    assert engine.evaluate(rec, degraded)[1].expected_values["brake_power_kw"] == \
        engine.evaluate(rec, derived)[1].expected_values["brake_power_kw"]
