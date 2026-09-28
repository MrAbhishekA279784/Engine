"""
SFC (M-07), normalised engine load (M-08) and mechanical-efficiency plausibility.

    - SFC = m_dot_fuel * 3600 / P_brake; undefined (valid=False) at idle.
    - load = P_brake / P_rated, clipped at 1.2.
    - eta_mech = (IMEP - FMEP) / IMEP; outside 0.65-0.92 the value is kept,
      quality drops to 0.3 and a warning is logged.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.l1_data.simulator.forward_simulator import ForwardPhysicsModel, SensorForwardModel
from src.l2_digital_twin.physics.M07_specific_fuel_consumption import specific_fuel_consumption_kg_kwh
from src.l2_digital_twin.physics.M08_normalised_engine_load import normalised_engine_load
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import (
    ETA_MECH_IMPLAUSIBLE_QUALITY,
    EngineGeometry,
    MechanicalTwin,
    evaluate_digital_twin,
)
from tests.unit.test_module6 import _make_dummy_normalized_record as make_record

RATED_KW = 104.0  # Prompt 18: peak power per flyrotax.com/products/915-is-a-isc-a (was 105.0)


# --------------------------------------------------------------------------
# M-07 SFC
# --------------------------------------------------------------------------

def test_sfc_hand_computed() -> None:
    # 0.0055 kg/s * 3600 s/h / 64 kW = 0.309375 kg/kWh
    assert specific_fuel_consumption_kg_kwh(0.0055, 64.0) == pytest.approx(0.309, abs=5e-4)


def test_twin_sfc_matches_fuel_over_power() -> None:
    res = MechanicalTwin(EngineGeometry()).compute(make_record(rpm_val=5000.0, fuel_kg_s=0.0055))
    assert res.brake_power_kw.valid and res.sfc_kg_kwh.valid
    assert res.sfc_kg_kwh.value == pytest.approx(0.0055 * 3600.0 / res.brake_power_kw.value, rel=1e-12)
    assert res.sfc_band != "UNKNOWN"
    assert res.sfc_degradation_pct.valid
    assert res.sfc_degradation_pct.value == pytest.approx(100.0 * (res.sfc_kg_kwh.value - 0.312) / 0.312)


@pytest.mark.parametrize("fuel_kg_s", [0.0, 2e-5])
def test_sfc_invalid_at_idle(fuel_kg_s: float) -> None:
    """Fuel cut-off, or idle fuelling that yields < 1 kW brake power."""
    res = MechanicalTwin(EngineGeometry()).compute(make_record(rpm_val=800.0, fuel_kg_s=fuel_kg_s))
    assert res.sfc_kg_kwh.valid is False
    assert res.sfc_kg_kwh.value is None
    assert res.sfc_band == "UNKNOWN"
    assert res.sfc_degradation_pct.valid is False


def test_sfc_invalid_without_measured_fuel() -> None:
    """Fallback power uses stoichiometric fuel estimated from air: SFC would be circular."""
    res = MechanicalTwin(EngineGeometry()).compute(make_record(fuel_valid=False))
    assert res.brake_power_kw.valid
    assert res.sfc_kg_kwh.valid is False


# --------------------------------------------------------------------------
# M-08 load
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("power_kw", "load"), [(52.0, 0.5), (140.0, 1.2)])  # 52.0 = half of 104 kW (was 52.5 of 105)
def test_normalised_load_points(power_kw: float, load: float) -> None:
    assert normalised_engine_load(power_kw, RATED_KW) == pytest.approx(load)


def test_twin_load_is_power_over_rated_with_band() -> None:
    res = MechanicalTwin(EngineGeometry()).compute(make_record(rpm_val=5000.0, fuel_kg_s=0.004))
    assert res.engine_load.valid
    assert res.engine_load.value == pytest.approx(min(res.brake_power_kw.value / RATED_KW, 1.2))
    assert res.load_band in ("IDLE", "LIGHT", "CRUISE", "HIGH", "ABOVE_RATED")


def test_load_invalid_without_power() -> None:
    res = MechanicalTwin(EngineGeometry()).compute(make_record(rpm_val=0.0, rpm_valid=False))
    assert res.engine_load.valid is False and res.engine_load.value is None
    assert res.load_band == "UNKNOWN"


# --------------------------------------------------------------------------
# eta_mech plausibility
# --------------------------------------------------------------------------

def test_eta_mech_is_imep_minus_fmep_over_imep() -> None:
    state = evaluate_digital_twin(make_record(rpm_val=5000.0, fuel_kg_s=0.006))
    expected = (state.imep_pa.value - state.fmep_pa.value) / state.imep_pa.value
    assert state.eta_mechanical.value == pytest.approx(expected, rel=1e-9)


def test_implausible_eta_mech_kept_with_low_quality_and_warning(caplog) -> None:
    """High fuel at low rpm -> IMEP >> FMEP -> eta_mech > 0.92."""
    with caplog.at_level(logging.WARNING):
        res = MechanicalTwin(EngineGeometry()).compute(make_record(rpm_val=2000.0, fuel_kg_s=0.006))
    assert res.eta_mech.valid
    assert res.eta_mech.value > 0.92
    assert res.eta_mech.quality == ETA_MECH_IMPLAUSIBLE_QUALITY
    assert any("Implausible mechanical efficiency" in r.getMessage() for r in caplog.records)


def test_plausible_eta_mech_keeps_power_quality() -> None:
    """Cruise-to-rated fuel at high rpm -> eta_mech inside the window -> normal quality."""
    res = MechanicalTwin(EngineGeometry()).compute(make_record(rpm_val=5800.0, fuel_kg_s=0.006))
    assert 0.65 <= res.eta_mech.value <= 0.92
    assert res.eta_mech.quality == res.brake_power_kw.quality


def _nominal_eta_mech(rpm: float) -> float:
    gt = ForwardPhysicsModel(seed=7).compute_ground_truth(time_s=1.0, rpm=rpm, map_pa=110000.0)
    raw = SensorForwardModel(seed=7).convert_to_raw_record(gt, enable_noise=False)
    return evaluate_digital_twin(convert_raw_to_engineering_state(raw)).eta_mechanical.value


def test_eta_mech_plausible_across_rpm_on_nominal_simulator() -> None:
    """Was a strict xfail (Prompt 5: 0.94-0.99). After rpm-dependent simulator
    fuel and Barnes-Moss FMEP: 0.757 (5800 rpm) to 0.888 (2000 rpm) at 110 kPa."""
    values = {rpm: _nominal_eta_mech(rpm) for rpm in (2000.0, 3000.0, 4000.0, 5000.0, 5800.0)}
    assert all(0.65 <= v <= 0.92 for v in values.values()), values


# --------------------------------------------------------------------------
# Propagation to DerivedEngineState and the API
# --------------------------------------------------------------------------

def test_derived_engine_state_carries_sfc_and_load() -> None:
    state = evaluate_digital_twin(make_record(rpm_val=5000.0, fuel_kg_s=0.0055))
    assert state.sfc_kg_kwh.valid and state.engine_load.valid
    assert state.sfc_band != "UNKNOWN" and state.load_band != "UNKNOWN"


def test_engine_health_endpoint_reports_sfc_and_load() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/engine/health")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        for key in ("sfc_kg_kwh", "sfc_band", "sfc_degradation_pct", "engine_load", "load_band"):
            assert key in data
        assert data["load_band"] in ("UNKNOWN", "IDLE", "LIGHT", "CRUISE", "HIGH", "ABOVE_RATED")
