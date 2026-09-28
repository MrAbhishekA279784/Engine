"""
Simulator fuel chain (air -> lambda -> fuel -> IMEP), Barnes-Moss FMEP in the
twin, and simulator sensor-fault quality flags.

Values below are reported by the tests' docstrings; thresholds come from the
task specification and are not tuned.
"""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.provenance import FaultClass
from src.core.schemas import SignalQuality
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
)
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from src.l2_digital_twin.thermo_mechanical_twin import (
    FMEP_UNSCALED_QUALITY,
    EngineGeometry,
    MechanicalTwin,
)
from tests.unit.test_module6 import _make_dummy_normalized_record as make_record

DISP_M3 = 1.352e-3
RPM_GRID = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 5800.0)
MAP_GRID = (80000.0, 110000.0, 135000.0, 160000.0)
MAX_MAP_PA = 160000.0
CRUISE_MAP_PA = 100000.0


def _gt(rpm: float, map_pa: float):
    return ForwardPhysicsModel(seed=7).compute_ground_truth(time_s=1.0, rpm=rpm, map_pa=map_pa)


# --------------------------------------------------------------------------
# B. Simulator fuel depends on rpm through air flow
# --------------------------------------------------------------------------

@pytest.mark.parametrize("map_pa", [80000.0, 100000.0, 110000.0, 135000.0])
def test_fuel_ratio_5800_over_2000_at_fixed_map(map_pa: float) -> None:
    """Measured: 2.973 at 80-110 kPa, 3.119 at 135 kPa."""
    ratio = _gt(5800.0, map_pa).fuel_flow_kg_s / _gt(2000.0, map_pa).fuel_flow_kg_s
    assert 2.4 <= ratio <= 3.2


def test_fuel_ratio_at_max_map_includes_full_load_enrichment() -> None:
    """At 160 kPa the fuel ratio is 3.418, outside 2.4-3.2: 5800 rpm reaches
    100 % load and is enriched to lambda 0.87 while 2000 rpm stays at 1.00.
    The rpm (air) part alone, fuel * lambda, is 2.973."""
    hi, lo = _gt(5800.0, MAX_MAP_PA), _gt(2000.0, MAX_MAP_PA)
    assert hi.lambda_cmd == pytest.approx(0.87, abs=1e-6)
    assert lo.lambda_cmd == pytest.approx(1.00, abs=1e-6)
    assert hi.fuel_flow_kg_s / lo.fuel_flow_kg_s == pytest.approx(3.418, abs=0.01)
    air_ratio = (hi.fuel_flow_kg_s * hi.lambda_cmd) / (lo.fuel_flow_kg_s * lo.lambda_cmd)
    assert 2.4 <= air_ratio <= 3.2


def test_rated_point_brake_power() -> None:
    """5800 rpm, maximum MAP (160 kPa): measured 100.2 kW."""
    assert 95.0 <= _gt(5800.0, MAX_MAP_PA).brake_power_kw <= 115.0


def test_low_rpm_cruise_brake_power() -> None:
    """2000 rpm, cruise MAP 100 kPa: measured 22.3 kW."""
    assert _gt(2000.0, CRUISE_MAP_PA).brake_power_kw < 50.0


def test_imep_bounded_on_grid() -> None:
    """Prompt 5 grid (2000-5800 rpm x 80-160 kPa): measured maximum 1.90 MPa."""
    imeps = {(r, m): _gt(r, m).imep_pa for r in RPM_GRID for m in MAP_GRID}
    assert max(imeps.values()) <= 2.5e6, max(imeps.items(), key=lambda kv: kv[1])


def test_power_follows_from_fuel_imep_fmep() -> None:
    gt = _gt(4500.0, 135000.0)
    assert gt.fuel_flow_kg_s == pytest.approx(gt.air_mass_flow_kg_s / (14.7 * gt.lambda_cmd))
    expected_kw = (gt.imep_pa - gt.fmep_pa) * DISP_M3 * 4500.0 / 120.0 / 1000.0
    assert gt.brake_power_kw == pytest.approx(expected_kw, rel=1e-9)


def test_lambda_schedule() -> None:
    model = ForwardPhysicsModel(seed=7)
    assert model.lambda_command(0.5) == pytest.approx(1.00)
    assert model.lambda_command(0.75) == pytest.approx(1.00)
    assert model.lambda_command(0.875) == pytest.approx(0.935)
    assert model.lambda_command(1.0) == pytest.approx(0.87)


def test_simulator_does_not_import_l2() -> None:
    tree = ast.parse(Path("src/l1_data/simulator/forward_simulator.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("src.l2"), node.module
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("src.l2") for a in node.names)


# --------------------------------------------------------------------------
# C. Barnes-Moss FMEP (twin and simulator)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("rpm", "fmep_bar"), [(5800.0, 3.522), (2000.0, 1.470)])
def test_simulator_barnes_moss(rpm: float, fmep_bar: float) -> None:
    assert ForwardPhysicsModel(seed=7).fmep_pa(rpm) / 1e5 == pytest.approx(fmep_bar, abs=1e-3)


@pytest.mark.parametrize(("rpm", "fmep_bar"), [(5800.0, 3.522), (2000.0, 1.470)])
def test_twin_barnes_moss_at_nominal_oil_temperature(rpm: float, fmep_bar: float) -> None:
    """Oil at the 90 degC reference -> viscosity ratio 1 -> unscaled value."""
    rec = make_record(rpm_val=rpm)
    rec = rec.model_copy(update={"oil_temp": rec.oil_temp.model_copy(update={"value": 363.15})})
    res = MechanicalTwin(EngineGeometry()).compute(rec)
    assert res.fmep_pa.value / 1e5 == pytest.approx(fmep_bar, abs=1e-3)
    assert res.fmep_pa.quality == 1.0


def test_cold_oil_raises_fmep() -> None:
    base = make_record(rpm_val=4000.0)
    cold = base.model_copy(update={"oil_temp": base.oil_temp.model_copy(update={"value": 323.15})})
    twin = MechanicalTwin(EngineGeometry())
    assert twin.compute(cold).fmep_pa.value > 1.1 * twin.compute(base).fmep_pa.value


def test_invalid_oil_temperature_gives_unscaled_fmep_quality_half() -> None:
    base = make_record(rpm_val=5800.0)
    rec = base.model_copy(update={"oil_temp": base.oil_temp.model_copy(update={"valid": False, "quality": 0.0})})
    res = MechanicalTwin(EngineGeometry()).compute(rec)
    assert res.fmep_pa.valid
    assert res.fmep_pa.value / 1e5 == pytest.approx(3.522, abs=1e-3)
    assert res.fmep_pa.quality == FMEP_UNSCALED_QUALITY


# --------------------------------------------------------------------------
# A2. Simulator sensor faults mark their raw channel; L2 emits it invalid
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("raw_channel", "l2_channel", "reason"), [
    ("egt_cyl1_hot_uv", "egt_cyl_1", "SENSOR_DROPOUT"),
    ("map_counts", "map_pressure", "SENSOR_DROPOUT"),
    ("oil_p_counts", "oil_pressure", "SENSOR_SATURATION"),
])
def test_sensor_fault_marks_channel_and_l2_invalidates_it(raw_channel, l2_channel, reason) -> None:
    fault = FaultScenarioConfig(fault_class=FaultClass.SENSOR_FAULT, onset_time_s=0.0,
                                duration_s=10.0, affected_channel=raw_channel)
    raw = SensorForwardModel(seed=0).convert_to_raw_record(_gt(4000.0, 110000.0), fault, enable_noise=False)
    sq = raw.signal_quality
    assert sq.valid is False and sq.score == 0.0
    assert sq.invalid_channels == [raw_channel]
    assert sq.invalid_reasons == {raw_channel: reason}
    eng = convert_raw_to_engineering_state(raw)
    ch = getattr(eng, l2_channel)
    assert ch.valid is False
    assert ch.fault_flag == reason
    assert eng.rpm.valid  # failure isolated to the affected channel


def test_signal_quality_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SignalQuality(valid=False, quality=0.0)
    with pytest.raises(ValidationError):
        SignalQuality(fault_flag="SENSOR_DROPOUT")
