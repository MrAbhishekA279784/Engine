"""
D-04 speed-density air flow and D-03 lambda derived from mass flows.

    - MAP drives air mass flow; fuel flow never does.
    - lambda = m_dot_air / (m_dot_fuel * AFR_stoich); AFR = lambda * AFR_stoich.
    - Ring-wear signature: same MAP/rpm, more fuel -> lambda and combustion
      efficiency fall.
    - Undefined lambda (zero fuel) is valid=False with value None.
    - sensor_inverse no longer invents a lambda channel.
"""

import math

import pytest

from src.core.provenance import Provenance
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import (
    CHARGE_TEMP_RISE_K,
    EngineGeometry,
    ThermodynamicMechanicalTwin,
    ThermodynamicTwin,
)
from tests.unit.test_module5 import valid_raw_record  # noqa: F401  (fixture)
from tests.unit.test_module6 import _make_dummy_normalized_record as make_record

CRUISE = dict(rpm_val=5000.0, map_pa=120000.0, amb_press_pa=101325.0, amb_temp_k=288.15)


@pytest.fixture
def twin() -> ThermodynamicTwin:
    return ThermodynamicTwin(EngineGeometry())


def test_map_drives_air_mass_flow(twin: ThermodynamicTwin) -> None:
    """MAP 70 -> 140 kPa at fixed rpm raises air flow 1.9-2.2x (old code: exactly 1.0x)."""
    lo = twin.compute(make_record(rpm_val=5000.0, map_pa=70000.0)).air_mass_flow_kg_s
    hi = twin.compute(make_record(rpm_val=5000.0, map_pa=140000.0)).air_mass_flow_kg_s
    assert lo.valid and hi.valid
    assert 1.9 <= hi.value / lo.value <= 2.2


def test_air_mass_flow_independent_of_fuel_flow(twin: ThermodynamicTwin) -> None:
    flows = [
        twin.compute(make_record(**CRUISE, fuel_kg_s=fuel)).air_mass_flow_kg_s.value
        for fuel in (0.002, 0.004, 0.006, 0.010)
    ]
    assert all(f == flows[0] for f in flows)
    invalid_fuel = twin.compute(make_record(**CRUISE, fuel_valid=False)).air_mass_flow_kg_s
    assert invalid_fuel.valid and invalid_fuel.value == flows[0]


def test_speed_density_uses_four_stroke_rpm_over_120(twin: ThermodynamicTwin) -> None:
    res = twin.compute(make_record(**CRUISE))
    rho = 120000.0 / (287.0528 * (288.15 + CHARGE_TEMP_RISE_K))
    expected = res.eta_volumetric.value * rho * 1.352e-3 * 5000.0 / 120.0
    assert res.air_mass_flow_kg_s.value == pytest.approx(expected, rel=1e-9)
    assert res.air_density_kg_m3.value == pytest.approx(rho, rel=1e-9)


def test_eta_volumetric_is_tagged_derived_model_value(twin: ThermodynamicTwin) -> None:
    res = twin.compute(make_record(**CRUISE))
    assert res.eta_volumetric.valid
    assert res.eta_volumetric.provenance == Provenance.DERIVED
    assert 0.5 < res.eta_volumetric.value < 1.2


def test_lambda_afr_and_combustion_efficiency_from_mass_flows(twin: ThermodynamicTwin) -> None:
    res = twin.compute(make_record(**CRUISE, fuel_kg_s=0.00422))
    air = res.air_mass_flow_kg_s.value
    assert res.lambda_derived.valid
    assert res.lambda_derived.provenance == Provenance.DERIVED
    assert res.lambda_derived.value == pytest.approx(air / (0.00422 * 14.7), rel=1e-9)
    assert res.afr.value == pytest.approx(res.lambda_derived.value * 14.7, rel=1e-9)
    assert res.combustion_efficiency.valid


def test_ring_wear_signature_lambda_and_combustion_efficiency_fall(twin: ThermodynamicTwin) -> None:
    """Same MAP and rpm, fuel +25 %: lambda ~1.05 -> ~0.84, combustion efficiency falls."""
    healthy = twin.compute(make_record(**CRUISE, fuel_kg_s=0.00422))
    worn = twin.compute(make_record(**CRUISE, fuel_kg_s=0.00422 * 1.25))

    assert healthy.lambda_derived.value == pytest.approx(1.05, abs=0.01)
    assert worn.lambda_derived.value == pytest.approx(0.84, abs=0.01)
    assert worn.lambda_derived.value == pytest.approx(healthy.lambda_derived.value / 1.25, rel=1e-9)
    assert worn.combustion_efficiency.value < healthy.combustion_efficiency.value
    # Air is unchanged: the deviation is visible only through lambda.
    assert worn.air_mass_flow_kg_s.value == healthy.air_mass_flow_kg_s.value


@pytest.mark.parametrize("fuel_kwargs", [{"fuel_kg_s": 0.0}, {"fuel_valid": False}])
def test_lambda_undefined_without_fuel_flow(twin: ThermodynamicTwin, fuel_kwargs: dict) -> None:
    """Idle cut-off (fuel 0) or invalid fuel: lambda is valid=False, never inf or 0."""
    res = twin.compute(make_record(**CRUISE, **fuel_kwargs))
    for tv in (res.lambda_derived, res.afr, res.combustion_efficiency):
        assert tv.valid is False
        assert tv.value is None
        assert tv.quality == 0.0
    assert res.air_mass_flow_kg_s.valid  # air path is unaffected


def test_derived_engine_state_carries_lambda_and_combustion_efficiency() -> None:
    state = ThermodynamicMechanicalTwin().evaluate(make_record(**CRUISE, fuel_kg_s=0.00422))
    assert state.lambda_derived.valid
    assert state.lambda_derived.value == pytest.approx(1.05, abs=0.01)
    assert state.combustion_efficiency.valid
    assert state.afr.value == pytest.approx(state.lambda_derived.value * 14.7, rel=1e-9)


def test_lambda_sensor_input_is_ignored(twin: ThermodynamicTwin) -> None:
    """A bogus lambda_sensor channel must not change the derived lambda."""
    a = twin.compute(make_record(**CRUISE, fuel_kg_s=0.00422, lambda_val=1.0))
    b = twin.compute(make_record(**CRUISE, fuel_kg_s=0.00422, lambda_val=0.5))
    assert a.lambda_derived.value == b.lambda_derived.value
    assert a.afr.value == b.afr.value


def test_sensor_inverse_does_not_invent_lambda(valid_raw_record) -> None:  # noqa: F811
    eng = convert_raw_to_engineering_state(valid_raw_record)
    assert eng.lambda_sensor.valid is False
    assert eng.lambda_sensor.quality == 0.0
    assert math.isnan(eng.lambda_sensor.value)
    assert eng.lambda_sensor.fault_flag
