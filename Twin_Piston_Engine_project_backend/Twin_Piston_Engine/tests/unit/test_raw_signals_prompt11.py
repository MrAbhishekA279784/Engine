"""
Prompt 11: optional raw electrical, injection, ignition, fuel-pressure and
coolant signals — interface and transport only (no physics).
"""

import csv
import hashlib
import json

import pytest

from src.l1_data.adapters.can_transport import (
    MockCANTransport,
    encode_can_frame_0x103,
    encode_can_frame_0x104,
    encode_can_frame_0x105,
    encode_can_frame_0x106,
    encode_can_frame_0x107,
)
from src.l1_data.adapters.csv_replay_adapter import CSVReplayAdapter
from src.l1_data.adapters.live_telemetry_adapter import LiveTelemetryAdapter
from src.l1_data.raw_repository import SQLiteRawTelemetryRepository
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_security import PacketSigner, PacketVerifier
from src.l1_data.telemetry_validator import TelemetryValidator
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state
from tests.unit.test_vibration_burst import _base_raw

NEW_FIELDS = RawSignalRecord.OPTIONAL_RAW_FIELDS

FULL = dict(
    # batt_i_counts added in Prompt 12 (new optional raw field)
    bus_v_counts=2800, alt_i_counts=1500, batt_i_counts=2150, fuel_press_counts=2100, coolant_ntc_ohms=1250.5,
    bus_v_burst_counts=tuple(2800 + (i % 7) for i in range(256)), bus_v_burst_fs_hz=2048.0,
    inj_pw_us=(3100.0, 3120.0, 3090.0, 3110.0), inj_soi_delay_us=(8000.0, 8010.0, 7990.0, 8005.0),
    ign_delay_us=(1040.0, 1045.0, 1038.0, 1042.0),
)


def _legacy_hash(rec: RawSignalRecord) -> str:
    payload = {
        "seq": rec.sequence_number, "egt1": rec.egt_cyl1_hot_uv, "egt2": rec.egt_cyl2_hot_uv,
        "egt3": rec.egt_cyl3_hot_uv, "egt4": rec.egt_cyl4_hot_uv, "egt_cold": rec.egt_cold_c,
        "cht": rec.cht_hot_uv, "cht_cold": rec.cht_cold_c, "oil_rtd": rec.oil_rtd_ohms,
        "oil_p": rec.oil_p_counts, "map": rec.map_counts, "vref": rec.adc_vref_counts,
        "crank_us": rec.crank_period_us, "fuel_hz": rec.fuel_pulse_hz,
        "accel": list(rec.accel_counts_xyz), "amb_t": rec.ambient_temp_c, "amb_p": rec.ambient_press_pa,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Record, hash, signature, back-compatibility
# --------------------------------------------------------------------------

def test_new_fields_default_to_none() -> None:
    rec = _base_raw()
    assert all(getattr(rec, f) is None for f in NEW_FIELDS)


def test_all_none_record_serialises_hashes_and_deserialises() -> None:
    rec = _base_raw()
    assert rec.compute_integrity_hash() == _legacy_hash(rec)  # None fields are not in the hash
    back = RawSignalRecord.model_validate_json(rec.model_dump_json())
    assert back == rec and back.compute_integrity_hash() == rec.integrity_hash


def test_old_record_without_new_keys_still_loads() -> None:
    data = json.loads(_base_raw().model_dump_json())
    for f in NEW_FIELDS:
        data.pop(f, None)
    rec = RawSignalRecord.model_validate(data)
    assert all(getattr(rec, f) is None for f in NEW_FIELDS)


def test_new_fields_enter_hash_and_round_trip() -> None:
    rec = _base_raw(**FULL)
    assert rec.compute_integrity_hash() != _legacy_hash(rec)
    back = RawSignalRecord.model_validate_json(rec.model_dump_json())
    assert back == rec and back.compute_integrity_hash() == rec.integrity_hash


@pytest.mark.parametrize(("field", "tampered"), [
    ("bus_v_counts", 2801), ("alt_i_counts", 1), ("fuel_press_counts", 2101), ("coolant_ntc_ohms", 1250.6),
    ("inj_pw_us", (3100.0, 3120.0, 3090.0, 3111.0)), ("inj_soi_delay_us", (8000.0, 8010.0, 7990.0, 8006.0)),
    ("ign_delay_us", (1041.0, 1045.0, 1038.0, 1042.0)), ("bus_v_burst_fs_hz", 1024.0),
])
def test_tampered_new_field_fails_signature(field: str, tampered) -> None:
    rec = _base_raw(**FULL)
    sig = PacketSigner().sign_record(rec)
    assert PacketVerifier().verify_signature(rec, sig)
    assert not PacketVerifier().verify_signature(rec.model_copy(update={field: tampered}), sig)
    assert not TelemetryValidator().validate_packet(rec.model_copy(update={field: tampered}), signature=sig).accepted


def test_bus_v_burst_requires_sample_rate() -> None:
    with pytest.raises(ValueError):
        _base_raw(bus_v_burst_counts=(1, 2, 3))


def test_validator_range_checks_new_adc_counts() -> None:
    rec = _base_raw(bus_v_counts=5000)
    res = TelemetryValidator().validate_packet(rec, signature=PacketSigner().sign_record(rec))
    assert res.accepted and "bus_v_counts" in res.invalid_channels


# --------------------------------------------------------------------------
# CAN 0x103-0x106
# --------------------------------------------------------------------------

# Frame layouts revised before Prompt 12 (OI-18 decisions 1d/1e): the coolant
# NTC moved from 0x103 to 0x107 (u32 ohm x 10); 0x103 is now fuel_press,
# bus_v, alt_i + spare; 0x106 is signed int16. Cases updated accordingly.
CAN_CASES = {
    "min": dict(v=0, i=0, fp=0, ntc=100.0, pw=(0.0,) * 4, soi=(0.0,) * 4, ign=(-32767.0,) * 4),
    "mid": dict(v=32767, i=16000, fp=2048, ntc=10000.0, pw=(3100.0, 3120.0, 3090.0, 3110.0),
                soi=(32767.0, 1.0, 2.0, 3.0), ign=(1040.0, -1045.0, 1038.0, 1042.0)),
    "max": dict(v=65535, i=65535, fp=65535, ntc=429496729.5, pw=(65535.0,) * 4, soi=(65535.0,) * 4,
                ign=(32767.0,) * 4),
}


async def _decode(case: dict) -> RawSignalRecord:
    transport = MockCANTransport()
    adapter = LiveTelemetryAdapter(transport=transport)
    await adapter.connect()
    for frame in (encode_can_frame_0x103(case["fp"], case["v"], case["i"]), encode_can_frame_0x107(case["ntc"]),
                  encode_can_frame_0x104(case["pw"]), encode_can_frame_0x105(case["soi"]),
                  encode_can_frame_0x106(case["ign"])):
        await transport.send_frame(frame)
    rec = None
    for _ in range(5):
        rec = await adapter.read_next()
    await adapter.disconnect()
    return rec


@pytest.mark.parametrize("level", ["min", "mid", "max"])
async def test_can_round_trip_0x103_to_0x107(level: str) -> None:
    case = CAN_CASES[level]
    rec = await _decode(case)
    assert (rec.bus_v_counts, rec.alt_i_counts, rec.fuel_press_counts) == (case["v"], case["i"], case["fp"])
    assert rec.coolant_ntc_ohms == pytest.approx(case["ntc"], abs=0.05)
    assert rec.inj_pw_us == case["pw"] and rec.inj_soi_delay_us == case["soi"] and rec.ign_delay_us == case["ign"]
    limit_flags = {f for f in ("coolant_ntc_ohms", "inj_pw_us", "inj_soi_delay_us", "ign_delay_us")
                   if f in rec.signal_quality.invalid_channels}
    if level == "max":  # values at the encoding limits: flagged invalid, not passed on as readings
        assert limit_flags == {"coolant_ntc_ohms", "inj_pw_us", "inj_soi_delay_us", "ign_delay_us"}
    else:
        assert limit_flags == set()


def test_can_frames_are_8_bytes_big_endian() -> None:
    f = encode_can_frame_0x103(0x0102, 0x0304, 0x0506)
    assert f.can_id == 0x103 and f.data == bytes([1, 2, 3, 4, 5, 6, 0, 0])
    assert encode_can_frame_0x106((-5.0, 0.0, 0.0, 0.0)).data[:2] == (-5).to_bytes(2, "big", signed=True)
    assert encode_can_frame_0x107(1234.5).data == (12345).to_bytes(4, "big") + bytes(4)


# --------------------------------------------------------------------------
# CSV, repository, API, transport
# --------------------------------------------------------------------------

def test_csv_replay_reads_new_columns_and_missing_as_none(tmp_path) -> None:
    path = tmp_path / "p11.csv"
    cols = ["sequence_number", "bus_v_counts", "alt_i_counts", "fuel_press_counts", "coolant_ntc_ohms",
            *[f"inj_pw_us_{i}" for i in range(1, 5)], *[f"ign_delay_us_{i}" for i in range(1, 5)]]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([1, 2800, 1500, 2100, 1250.5, 3100, 3120, 3090, 3110, 1040, 1045, 1038, 1042])
    row = next(csv.DictReader(path.open(encoding="utf-8")))
    rec = CSVReplayAdapter(path)._row_to_raw_record(row, 1)
    assert (rec.bus_v_counts, rec.alt_i_counts, rec.fuel_press_counts) == (2800, 1500, 2100)
    assert rec.coolant_ntc_ohms == 1250.5
    assert rec.inj_pw_us == (3100.0, 3120.0, 3090.0, 3110.0)
    assert rec.ign_delay_us == (1040.0, 1045.0, 1038.0, 1042.0)
    assert rec.inj_soi_delay_us is None  # columns absent -> not instrumented


def test_sqlite_repository_round_trips_new_fields() -> None:
    rec = _base_raw(**FULL)
    repo = SQLiteRawTelemetryRepository(":memory:")
    assert repo.save(rec)
    assert repo.get_by_sequence(rec.sequence_number) == rec


def test_api_ingest_and_transport_payload_carry_new_fields() -> None:
    from fastapi.testclient import TestClient

    from src.api.app import create_app
    from src.api.dependencies import get_telemetry_repository
    from src.edge_ground.envelope import EdgeTelemetryEnvelope
    from src.l1_data.raw_repository import InMemoryRawTelemetryRepository

    rec = _base_raw(**FULL).model_copy(update={"sequence_number": 424242})
    payload = json.loads(rec.model_dump_json(exclude={"timestamp", "integrity_hash", "signal_quality"}))
    repo = InMemoryRawTelemetryRepository()
    app = create_app()
    app.dependency_overrides[get_telemetry_repository] = lambda: repo
    with TestClient(app) as client:
        assert client.post("/api/v1/telemetry", json=payload).status_code == 201
    stored = repo.get_by_sequence(424242)
    for f in NEW_FIELDS:
        assert getattr(stored, f) == getattr(rec, f), f

    env = EdgeTelemetryEnvelope.from_record(rec)
    body = {name: (list(v) if isinstance(v, tuple) else v)
            for name in env.record.OPTIONAL_RAW_FIELDS if (v := getattr(env.record, name)) is not None}
    assert set(body) == set(NEW_FIELDS)


# --------------------------------------------------------------------------
# L2 view: no derivation yet; None = not instrumented
# --------------------------------------------------------------------------

def test_l2_channels_for_new_signals_are_invalid_with_reason() -> None:
    absent = convert_raw_to_engineering_state(_base_raw())
    present = convert_raw_to_engineering_state(_base_raw(**FULL))
    # Prompt 12 derives voltage and current: removed from the "not implemented yet"
    # list; they are now valid when their raw input is present.
    # Prompt 13 derives fuel pressure, pulse width, SOI and ignition timing:
    # moved from "not implemented yet" to "valid when present".
    # Prompt 14 derives coolant_temp (NTC): every Prompt 11 channel is now derived.
    for name in ("voltage", "current", "battery_current", "fuel_pressure", "ignition_timing_cyl_1",
                 "injector_pulse_width_cyl_1", "injection_timing_cyl_4", "injection_soi_deg_cyl_2",
                 "coolant_temp"):
        assert getattr(absent, name).valid is False and "not instrumented" in getattr(absent, name).fault_flag
        assert getattr(present, name).valid is True
    # None fields are "not instrumented" and do not reduce coverage
    assert absent.channel_coverage == convert_raw_to_engineering_state(_base_raw()).channel_coverage
