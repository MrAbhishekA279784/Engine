"""
Invalid-flag propagation L1 -> L2 (OI-12), null serialisation (OI-15),
CSI window fill (OI-13) and simulator consistency (OI-14).

The propagation test is parametrised over the raw data fields of
RawSignalRecord itself, so a raw field added later is covered automatically:
it fails until the field is mapped in sensor_inverse.RAW_FIELD_DEPENDENTS.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.core.provenance import FaultClass
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue
from src.l1_data.simulator.forward_simulator import (
    FaultScenarioConfig,
    ForwardPhysicsModel,
    SensorForwardModel,
)
from src.l1_data.telemetry_security import PacketSigner
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l2_digital_twin.misfire_classifier import MisfireDetector
from src.l2_digital_twin.sensor_inverse import (
    MEASURED_CHANNELS,
    RAW_FIELD_DEPENDENTS,
    convert_raw_to_engineering_state,
)
from src.l2_digital_twin.thermo_mechanical_twin import evaluate_digital_twin
from src.l2_digital_twin.vibration_processor import evaluate_vibration_processor

METADATA_FIELDS = {"timestamp", "sequence_number", "source_type", "integrity_hash", "signal_quality"}
RAW_DATA_FIELDS = sorted(f for f in RawSignalRecord.model_fields if f not in METADATA_FIELDS)
T0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


def _sim_raw(time_s: float = 1.0, fault=None, severity: float = 0.0, map_pa: float = 110000.0,
             sensor_model: SensorForwardModel | None = None, model: ForwardPhysicsModel | None = None):
    scenario = None
    if fault is not None:
        scenario = FaultScenarioConfig(fault_class=fault, severity=severity, onset_time_s=0.0, duration_s=1e4)
    gt = (model or ForwardPhysicsModel(seed=7)).compute_ground_truth(
        time_s=time_s, sequence_number=int(time_s) + 1, rpm=4000.0, map_pa=map_pa,
        fault_scenario=scenario, timestamp=T0 + timedelta(seconds=time_s))
    return (sensor_model or SensorForwardModel(seed=7)).convert_to_raw_record(gt, scenario), gt


def _validate(rec: RawSignalRecord) -> RawSignalRecord:
    res = TelemetryValidator().validate_packet(rec, signature=PacketSigner().sign_record(rec))
    assert res.accepted
    return res.record


# --------------------------------------------------------------------------
# OI-12: source-marked invalid flags survive L1 validation into L2
# --------------------------------------------------------------------------

def test_every_raw_data_field_is_mapped_to_l2_channels() -> None:
    assert set(RAW_DATA_FIELDS) == set(RAW_FIELD_DEPENDENTS), (
        "map every RawSignalRecord raw field in sensor_inverse.RAW_FIELD_DEPENDENTS")


@pytest.mark.parametrize("raw_field", RAW_DATA_FIELDS)
def test_source_invalid_flag_survives_to_l2(raw_field: str) -> None:
    rec, _ = _sim_raw()
    flagged = rec.model_copy(update={"signal_quality": SignalQuality(
        score=0.5, valid=False, invalid_channels=[raw_field], invalid_reasons={raw_field: "SOURCE_TEST"})})
    validated = _validate(flagged)
    assert raw_field in validated.signal_quality.invalid_channels
    assert validated.signal_quality.invalid_reasons[raw_field] == "SOURCE_TEST"

    norm = convert_raw_to_engineering_state(validated)
    for channel in RAW_FIELD_DEPENDENTS[raw_field]:
        ch = getattr(norm, channel)
        assert ch.valid is False, channel
        assert ch.fault_flag == "SOURCE_TEST", (channel, ch.fault_flag)
        assert channel in norm.invalid_channels
    assert norm.channel_coverage < 1.0


def test_validator_keeps_its_own_findings_alongside_source_flags() -> None:
    rec, _ = _sim_raw()
    rec = rec.model_copy(update={"egt_cyl2_hot_uv": -999.0})  # out of raw range
    rec = rec.model_copy(update={"integrity_hash": rec.compute_integrity_hash(),
                                 "signal_quality": SignalQuality(valid=False, invalid_channels=["map_counts"],
                                                                 invalid_reasons={"map_counts": "SENSOR_DROPOUT"})})
    sq = _validate(rec).signal_quality
    assert set(sq.invalid_channels) == {"map_counts", "egt_cyl2_hot_uv"}
    assert sq.invalid_reasons["map_counts"] == "SENSOR_DROPOUT"
    assert "Range violation" in sq.invalid_reasons["egt_cyl2_hot_uv"]


def test_clean_record_full_coverage() -> None:
    """Setup changed in Prompt 10: propeller_speed needs engine.gearbox_ratio,
    which is unset by default (VERIFY), so a TEST-ONLY ratio is configured here."""
    from src.core.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.engine.gearbox_ratio = 2.0  # test-only value, not a Rotax figure
    rec, _ = _sim_raw()
    norm = convert_raw_to_engineering_state(_validate(rec), settings)
    assert norm.invalid_channels == [] and norm.channel_coverage == 1.0


def test_unconfigured_gearbox_ratio_reduces_coverage() -> None:
    """Setup changed (pre-Prompt 12): default ratio is now 2.54; set None here."""
    from src.core.config import get_settings

    settings = get_settings().model_copy(deep=True)
    settings.engine.gearbox_ratio = None
    rec, _ = _sim_raw()
    norm = convert_raw_to_engineering_state(_validate(rec), settings)
    assert norm.invalid_channels == ["propeller_speed"]
    assert "gearbox" in norm.propeller_speed.fault_flag.lower()


# --------------------------------------------------------------------------
# OI-15: invalid channels serialise as null, never NaN
# --------------------------------------------------------------------------

def test_record_with_every_channel_invalid_is_strict_json() -> None:
    rec, _ = _sim_raw()
    all_invalid = rec.model_copy(update={"signal_quality": SignalQuality(
        valid=False, invalid_channels=RAW_DATA_FIELDS, invalid_reasons={f: "ALL_INVALID" for f in RAW_DATA_FIELDS})})
    norm = convert_raw_to_engineering_state(_validate(all_invalid))
    assert set(norm.invalid_channels) == set(MEASURED_CHANNELS) and norm.channel_coverage == 0.0

    data = norm.model_dump(mode="json")
    text = json.dumps(data, allow_nan=False)            # raises on NaN/Infinity
    assert "NaN" not in text and "Infinity" not in text
    for name, value in data.items():
        if isinstance(value, dict) and value.get("valid") is False and "value" in value:
            assert value["value"] is None, name
    assert json.loads(norm.model_dump_json())["lambda_sensor"]["value"] is None


def test_channel_value_nan_and_invalid_serialise_null() -> None:
    nan_ch = ChannelValue(value=float("nan"), provenance="DERIVED", valid=False)
    placeholder = ChannelValue(value=0.0, provenance="DERIVED", valid=False)
    ok = ChannelValue(value=3.5, provenance="DERIVED", valid=True)
    assert nan_ch.model_dump(mode="json")["value"] is None
    assert placeholder.model_dump(mode="json")["value"] is None
    assert ok.model_dump(mode="json")["value"] == 3.5
    assert nan_ch.model_dump()["value"] != nan_ch.model_dump()["value"]  # in memory still NaN


# --------------------------------------------------------------------------
# OI-13: CSI thermal term excluded until the 30 s window is full
# --------------------------------------------------------------------------

def test_csi_thermal_term_excluded_until_window_full() -> None:
    det, sim, sfm = MisfireDetector(), ForwardPhysicsModel(seed=7), SensorForwardModel(seed=7)
    coverage = {}
    for k in range(35):
        raw, _ = _sim_raw(float(k), sensor_model=sfm, model=sim)
        norm = convert_raw_to_engineering_state(raw)
        _, res = det.evaluate(norm, evaluate_digital_twin(norm), None, evaluate_vibration_processor(norm)[1])
        coverage[k] = res.csi_terms.get("coverage")
        if k < 30 and res.csi_value is not None:
            assert "thermal_slope" not in res.csi_terms
            assert "window filling" in res.csi_reason
            assert res.csi_band == "NORMAL"
    assert coverage[29] == pytest.approx(2.0 / 3.0)
    assert coverage[30] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# OI-14: simulator consistency
# --------------------------------------------------------------------------

def test_simulator_oil_pressure_falls_with_oil_temperature() -> None:
    model = ForwardPhysicsModel(seed=7)
    assert model.oil_viscosity_ratio(model._phys.oil_pressure_ref_temp_k) == pytest.approx(1.0)
    assert model.oil_viscosity_ratio(393.15) < 1.0 < model.oil_viscosity_ratio(333.15)
    _, nominal = _sim_raw()
    _, cooling = _sim_raw(fault=FaultClass.COOLING_FAULT, severity=1.0)
    assert cooling.oil_temp_k > nominal.oil_temp_k
    assert cooling.oil_pressure_pa < nominal.oil_pressure_pa


def test_boost_leak_fuel_follows_post_leak_air() -> None:
    _, nominal = _sim_raw()
    _, leak = _sim_raw(fault=FaultClass.INTAKE_BOOST_LEAK, severity=1.0)
    assert leak.map_pressure_pa == pytest.approx(nominal.map_pressure_pa - 35000.0)
    assert leak.air_mass_flow_kg_s < nominal.air_mass_flow_kg_s
    assert leak.fuel_flow_kg_s == pytest.approx(leak.air_mass_flow_kg_s / (14.7 * leak.lambda_cmd))
    assert leak.fuel_flow_kg_s < nominal.fuel_flow_kg_s
    # power follows from the lower IMEP; ground-truth SFC rises (friction share)
    sfc = lambda gt: gt.fuel_flow_kg_s * 3600.0 / gt.brake_power_kw
    assert sfc(leak) > sfc(nominal)
