"""
OI-18 decisions (before Prompt 12): live CAN adapter emits None / invalid
until a frame group arrives; thermocouple uV offset encoding (0x101/0x102);
signed ignition delay (0x106); coolant NTC on 0x107 (u32 ohm x 10).
"""

import json

import pytest

from src.l1_data.adapters.can_transport import (
    MockCANTransport,
    encode_can_frame_0x100,
    encode_can_frame_0x101,
    encode_can_frame_0x102,
    encode_can_frame_0x103,
    encode_can_frame_0x106,
    encode_can_frame_0x107,
)
from src.l1_data.adapters.live_telemetry_adapter import FRAME_FIELDS, NOT_ON_CAN, LiveTelemetryAdapter
from src.l2_digital_twin.sensor_inverse import convert_raw_to_engineering_state


async def _records(*frames):
    transport = MockCANTransport()
    adapter = LiveTelemetryAdapter(transport=transport)
    await adapter.connect()
    for f in frames:
        await transport.send_frame(f)
    out = []
    for _ in frames:
        out.append(await adapter.read_next())
    await adapter.disconnect()
    return out


# --------------------------------------------------------------------------
# 1b. No placeholders before the first frame of each group
# --------------------------------------------------------------------------

async def test_fields_none_and_flagged_until_their_frame_arrives() -> None:
    (rec,) = await _records(encode_can_frame_0x101(30000, 30100, 29900, 30050))
    sq = rec.signal_quality
    assert rec.egt_cyl1_hot_uv == 30000.0 and "egt_cyl1_hot_uv" not in sq.invalid_channels
    for can_id, fields in FRAME_FIELDS.items():
        if can_id == 0x101:
            continue
        for f in fields:
            assert getattr(rec, f) is None, f
            assert sq.invalid_reasons[f] == f"No CAN frame 0x{can_id:03X} received yet"
    for f in NOT_ON_CAN:
        assert getattr(rec, f) is None
        assert sq.invalid_reasons[f] == "Not carried on the CAN bus (no data)"
    assert sq.valid is False


async def test_group_becomes_valid_once_received() -> None:
    first, second = await _records(encode_can_frame_0x101(30000, 30000, 30000, 30000),
                                   encode_can_frame_0x100(15000.0, 120.0))
    assert first.crank_period_us is None
    assert second.crank_period_us == 15000.0 and second.fuel_pulse_hz == 120.0
    assert "crank_period_us" not in second.signal_quality.invalid_channels
    assert second.egt_cyl1_hot_uv == 30000.0  # earlier group retained


async def test_l2_handles_partial_live_record() -> None:
    (rec,) = await _records(encode_can_frame_0x100(15000.0, 120.0))
    norm = convert_raw_to_engineering_state(rec)
    assert norm.rpm.valid and norm.rpm.value == pytest.approx(4000.0)
    for ch in ("egt_cyl_1", "cht_cyl_1", "oil_temp", "map_pressure", "vibration_rms", "ambient_temp", "altitude"):
        assert getattr(norm, ch).valid is False, ch
    # EGT: its hot-junction frame has not arrived AND the cold junction is not on CAN
    assert "0x101" in norm.egt_cyl_1.fault_flag
    text = json.dumps(norm.model_dump(mode="json"), allow_nan=False)
    assert "NaN" not in text


# --------------------------------------------------------------------------
# 1c. Thermocouple offset encoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("uv", [-500.0, 0.0, 39000.0, -9999.0, 55534.0])
async def test_thermocouple_offset_round_trip(uv: float) -> None:
    egt, cht = await _records(encode_can_frame_0x101(uv, uv, uv, uv), encode_can_frame_0x102(uv, 135.0, 2048, 2048))
    assert egt.egt_cyl1_hot_uv == uv and egt.egt_cyl4_hot_uv == uv
    assert cht.cht_hot_uv == uv
    assert "egt_cyl1_hot_uv" not in cht.signal_quality.invalid_channels
    assert "cht_hot_uv" not in cht.signal_quality.invalid_channels


@pytest.mark.parametrize(("uv", "decoded"), [(-10000.0, -10000.0), (-15000.0, -10000.0),
                                             (55535.0, 55535.0), (60000.0, 55535.0)])
async def test_thermocouple_saturation_limits_flagged(uv: float, decoded: float) -> None:
    (rec,) = await _records(encode_can_frame_0x101(uv, 30000, 30000, 30000))
    assert rec.egt_cyl1_hot_uv == decoded
    assert "encoding limit" in rec.signal_quality.invalid_reasons["egt_cyl1_hot_uv"]
    assert "egt_cyl2_hot_uv" not in rec.signal_quality.invalid_channels


# --------------------------------------------------------------------------
# 1d. Signed ignition delay
# --------------------------------------------------------------------------

async def test_ignition_delay_signed_round_trip() -> None:
    values = (-1500.0, 1040.0, 0.0, -1.0)
    (rec,) = await _records(encode_can_frame_0x106(values))
    assert rec.ign_delay_us == values
    assert "ign_delay_us" not in [c for c, r in rec.signal_quality.invalid_reasons.items() if "limit" in r]


@pytest.mark.parametrize("edge", [-32768.0, 32767.0, -40000.0, 40000.0])
async def test_ignition_delay_limits_flagged(edge: float) -> None:
    (rec,) = await _records(encode_can_frame_0x106((edge, 0.0, 0.0, 0.0)))
    assert rec.ign_delay_us[0] == max(-32768.0, min(32767.0, edge))
    assert "int16 limit" in rec.signal_quality.invalid_reasons["ign_delay_us"]


# --------------------------------------------------------------------------
# 1e. Coolant NTC on 0x107; 0x103 layout
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ohms", [100.0, 1000.0, 10000.0, 97000.0, 400000.0])
async def test_coolant_ntc_round_trip_100_ohm_to_400_kohm(ohms: float) -> None:
    (rec,) = await _records(encode_can_frame_0x107(ohms))
    assert rec.coolant_ntc_ohms == pytest.approx(ohms, abs=0.05)
    assert "coolant_ntc_ohms" not in [c for c, r in rec.signal_quality.invalid_reasons.items() if "0x107" in r]


async def test_coolant_ntc_short_circuit_flagged() -> None:
    (rec,) = await _records(encode_can_frame_0x107(0.0))
    assert "short circuit" in rec.signal_quality.invalid_reasons["coolant_ntc_ohms"]


async def test_0x103_new_layout() -> None:
    (rec,) = await _records(encode_can_frame_0x103(2100, 2800, 1500))
    assert (rec.fuel_press_counts, rec.bus_v_counts, rec.alt_i_counts) == (2100, 2800, 1500)
    assert rec.coolant_ntc_ohms is None  # no longer on 0x103
