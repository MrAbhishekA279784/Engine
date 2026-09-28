"""
Prompt 10: substituted defaults removed; environment parameters.

    - boost = MAP - ambient; pressure altitude from ISA; engine hours accumulated
      from timestamps; propeller speed = rpm / gearbox_ratio (config, VERIFY).
    - Channels with no raw sensor are valid=False, never a plausible constant.
    - Pressure altitude, ISA deviation and density altitude (ISO 2533 ISA).
"""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.core.units import isa_density, isa_density_altitude_m, isa_pressure, isa_pressure_altitude_m
from src.l2_digital_twin.sensor_inverse import SensorInverseModel, convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import ThermodynamicTwin, EngineGeometry, evaluate_digital_twin
from tests.unit.test_vibration_burst import _base_raw

NOT_INSTRUMENTED = (
    "coolant_temp", "fuel_pressure", "voltage", "current",
    "ignition_timing_cyl_1", "ignition_timing_cyl_2", "ignition_timing_cyl_3", "ignition_timing_cyl_4",
    "intake_air_temp", "wastegate_duty", "throttle_position",
)


def _settings(**engine):
    s = get_settings().model_copy(deep=True)
    for k, v in engine.items():
        setattr(s.engine, k, v)
    return s


def _norm(**raw):
    return convert_raw_to_engineering_state(_base_raw(**raw))


# --------------------------------------------------------------------------
# ISA inverses (src/core/units.py)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("h", [-1500.0, 0.0, 500.0, 2000.0, 5000.0, 8000.0, 11000.0])
def test_isa_pressure_altitude_round_trip(h: float) -> None:
    assert isa_pressure_altitude_m(isa_pressure(h)) == pytest.approx(h, abs=1e-6)


@pytest.mark.parametrize("h", [-1500.0, 0.0, 3000.0, 11000.0])
def test_isa_density_altitude_round_trip(h: float) -> None:
    assert isa_density_altitude_m(isa_density(h)) == pytest.approx(h, abs=1e-6)


@pytest.mark.parametrize("p", [0.0, -5.0, float("nan"), 20000.0, 130000.0])
def test_pressure_altitude_outside_troposphere_is_none(p: float) -> None:
    assert isa_pressure_altitude_m(p) is None


# --------------------------------------------------------------------------
# Environment parameters in DerivedEngineState
# --------------------------------------------------------------------------

def test_standard_day() -> None:
    state = evaluate_digital_twin(_norm(ambient_press_pa=101325.0, ambient_temp_c=15.0))
    assert state.pressure_altitude_m.valid and state.pressure_altitude_m.value == pytest.approx(0.0, abs=5.0)
    assert state.isa_deviation_k.value == pytest.approx(0.0, abs=0.1)
    assert state.density_altitude_m.value == pytest.approx(0.0, abs=30.0)
    assert state.pressure_altitude_m.provenance.value == "DERIVED"


def test_hot_day_at_84_kpa() -> None:
    state = evaluate_digital_twin(_norm(ambient_press_pa=84000.0, ambient_temp_c=35.0))
    assert state.isa_deviation_k.value > 0.0
    assert state.density_altitude_m.value > state.pressure_altitude_m.value
    # h = 44330.8 * (1 - (84000/101325)^0.190263) = 1553.6 m; T_ISA = 278.05 K -> +30.1 K
    assert state.pressure_altitude_m.value == pytest.approx(1553.6, abs=1.0)
    assert state.isa_deviation_k.value == pytest.approx(30.1, abs=0.1)


def test_environment_invalid_without_ambient_inputs() -> None:
    norm = _norm()
    norm = norm.model_copy(update={"ambient_temp": norm.ambient_temp.model_copy(update={"valid": False})})
    state = evaluate_digital_twin(norm)
    assert state.pressure_altitude_m.valid  # pressure alone is enough
    assert state.isa_deviation_k.valid is False and state.isa_deviation_k.value is None
    assert state.density_altitude_m.valid is False


def test_altitude_channel_is_pressure_altitude() -> None:
    norm = _norm(ambient_press_pa=84000.0)
    assert norm.altitude.valid and norm.altitude.value == pytest.approx(isa_pressure_altitude_m(84000.0))


# --------------------------------------------------------------------------
# Boost, propeller speed, engine hours
# --------------------------------------------------------------------------

def test_boost_zero_at_ambient_and_positive_under_boost() -> None:
    amb = 101325.0
    counts_at_ambient = round(amb / 200000.0 * 4095)
    near = _norm(map_counts=counts_at_ambient, ambient_press_pa=amb).boost_pressure
    assert near.valid and abs(near.value) < 200000.0 / 4095  # within one ADC count
    boosted = _norm(map_counts=round(135000.0 / 200000.0 * 4095), ambient_press_pa=amb).boost_pressure
    assert boosted.value > 30000.0


def test_propeller_speed_uses_configured_gearbox_ratio() -> None:
    norm = convert_raw_to_engineering_state(_base_raw(), _settings(gearbox_ratio=2.0))  # test-only ratio
    assert norm.propeller_speed.valid
    assert norm.propeller_speed.value == pytest.approx(norm.rpm.value / 2.0)


def test_propeller_speed_invalid_without_gearbox_ratio() -> None:
    """Setup changed (pre-Prompt 12): the default ratio is now 2.54, so the
    unconfigured case sets it to None explicitly. Expectation unchanged."""
    prop = convert_raw_to_engineering_state(_base_raw(), _settings(gearbox_ratio=None)).propeller_speed
    assert prop.valid is False and prop.value is None
    assert "VERIFY" in prop.fault_flag


def test_engine_hours_accumulate_while_running_and_reset() -> None:
    model = SensorInverseModel(_settings(engine_hours_at_install=100.0))
    base = _base_raw()
    t0 = base.timestamp
    for k in range(4):  # 3 intervals of 1 s, engine running
        model.convert_raw_to_engineering(base.model_copy(update={"timestamp": t0 + timedelta(seconds=k)}))
    hours = model.convert_raw_to_engineering(
        base.model_copy(update={"timestamp": t0 + timedelta(seconds=4)})).engine_hours
    assert hours.valid and hours.value == pytest.approx(100.0 + 4.0 / 3600.0)
    # gap longer than engine_hours_max_gap_s is not counted
    after_gap = model.convert_raw_to_engineering(
        base.model_copy(update={"timestamp": t0 + timedelta(seconds=64)})).engine_hours
    assert after_gap.value == pytest.approx(hours.value)
    # engine stopped (crank period invalid): no accumulation
    stopped = base.model_copy(update={"timestamp": t0 + timedelta(seconds=65), "crank_period_us": 1e12})
    model.convert_raw_to_engineering(stopped)
    model.reset_state()
    fresh = model.convert_raw_to_engineering(base.model_copy(update={"timestamp": t0}))
    assert fresh.engine_hours.value == pytest.approx(100.0)


def test_engine_hours_invalid_without_install_hours() -> None:
    hours = _norm().engine_hours
    assert hours.valid is False and hours.value is None
    assert "engine_hours_at_install" in hours.fault_flag


# --------------------------------------------------------------------------
# No substituted defaults
# --------------------------------------------------------------------------

@pytest.mark.parametrize("channel", NOT_INSTRUMENTED)
def test_not_instrumented_channels_are_invalid(channel: str) -> None:
    ch = getattr(_norm(), channel)
    assert ch.valid is False
    assert ch.value is None
    assert ch.quality == 0.0
    assert "not instrumented" in ch.fault_flag
    assert ch.model_dump(mode="json")["value"] is None


def test_thermodynamic_twin_does_not_use_intake_air_temp() -> None:
    """Charge temperature is ambient + intercooler rise (D-04), so an invalid or
    changed IAT channel cannot alter the air path."""
    norm = _norm()
    twin = ThermodynamicTwin(EngineGeometry())
    with_iat = norm.model_copy(update={"intake_air_temp": norm.intake_air_temp.model_copy(
        update={"value": 400.0, "valid": True, "quality": 1.0})})
    a, b = twin.compute(norm), twin.compute(with_iat)
    assert a.air_mass_flow_kg_s.value == b.air_mass_flow_kg_s.value
    assert a.air_density_kg_m3.value == b.air_density_kg_m3.value


def test_residual_operating_point_has_no_substituted_throttle() -> None:
    from src.l2_digital_twin.residual_engine import operating_point_from_record

    assert operating_point_from_record(_norm()).throttle_pct is None


def test_engine_health_endpoint_reports_environment() -> None:
    from src.api.app import create_app

    with TestClient(create_app()) as client:
        r = client.get("/api/v1/engine/health")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        for key in ("pressure_altitude_m", "isa_deviation_k", "density_altitude_m"):
            assert key in data
