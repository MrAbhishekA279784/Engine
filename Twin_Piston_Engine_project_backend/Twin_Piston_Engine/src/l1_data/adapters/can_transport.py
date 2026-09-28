"""
CAN 2.0 Transport Abstraction & Mock Transport.

Isolates CAN bus protocol parsing from physical hardware interfaces.
Provides an in-memory MockCANTransport for unit testing and offline replay
without requiring socketcan or CAN hardware drivers.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CANFrame:
    """Standard CAN 2.0 Data Frame (8 bytes payload max)."""

    can_id: int                          # 11-bit standard or 29-bit extended arbitration ID
    data: bytes                          # Up to 8 bytes payload
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if len(self.data) > 8:
            raise ValueError("CAN 2.0 payload exceeds maximum length of 8 bytes")


@runtime_checkable
class CANTransportProtocol(Protocol):
    """Protocol for CAN bus network interface abstraction."""

    async def connect(self) -> None:
        """Connect to CAN interface."""
        ...

    async def read_frame(self) -> CANFrame | None:
        """Read next CAN frame from bus."""
        ...

    async def send_frame(self, frame: CANFrame) -> None:
        """Send a CAN frame over the bus."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from CAN bus."""
        ...


class MockCANTransport:
    """In-memory Mock CAN Transport for testing without CAN hardware."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[CANFrame] = asyncio.Queue()
        self._connected: bool = False

    async def connect(self) -> None:
        self._connected = True

    async def read_frame(self) -> CANFrame | None:
        if not self._connected or self._queue.empty():
            return None
        return await self._queue.get()

    async def send_frame(self, frame: CANFrame) -> None:
        if not self._connected:
            raise RuntimeError("MockCANTransport is not connected")
        await self._queue.put(frame)

    async def disconnect(self) -> None:
        self._connected = False
        while not self._queue.empty():
            self._queue.get_nowait()

    @property
    def is_connected(self) -> bool:
        return self._connected


# Standard CAN 2.0 Signal Frame Encoders / Decoders for Rotax 915 iS ECU
# Big-endian, 8 data bytes. The live adapter decodes every frame; a value at an
# encoding limit (saturation) is flagged invalid, never passed on as a reading.
#
# CAN ID 0x100: crank_period_us (u32, us) + fuel_pulse_hz (u16, Hz x 10) + 2 bytes reserved
# CAN ID 0x101: egt_cyl1..4_hot_uv, u16 each, 1 uV per count with a +10 000 uV offset:
#               wire = uV + 10000 -> range -10 000 .. 55 535 uV (negative uV is physical:
#               hot junction colder than the cold junction). 0 and 65535 = saturated.
# CAN ID 0x102: cht_hot_uv (u16, same offset encoding as 0x101), oil_rtd_ohms (u16, ohm x 10),
#               map_counts (u16), oil_p_counts (u16)
# CAN ID 0x103: fuel_press_counts (u16), bus_v_counts (u16), alt_i_counts (u16),
#               batt_i_counts (u16; was the spare word until Prompt 12)
# CAN ID 0x104: inj_pw_us cyl 1-4 (u16 x4, us; 0xFFFF = saturated)
# CAN ID 0x105: inj_soi_delay_us cyl 1-4 (u16 x4, us; 0xFFFF = saturated)
# CAN ID 0x106: ign_delay_us cyl 1-4 (int16 x4, us, SIGNED: negative = coil fire after the
#               TDC reference; -32768 and 32767 = saturated)
# CAN ID 0x107: coolant_ntc_ohms (u32, ohm x 10 -> 0 .. 429 496 729.4 ohm) + 4 bytes reserved;
#               0 = short circuit, 0xFFFFFFFF = saturated
# Bursts (accelerometer, bus voltage) are too large for classic CAN: bulk path only.

TC_UV_OFFSET = 10000
U16_MAX = 65535
I16_MIN, I16_MAX = -32768, 32767
U32_MAX = 0xFFFFFFFF


def _clamp_int(value: float, lo: int, hi: int) -> int:
    return int(min(max(lo, round(value)), hi))


def _u16(value: float) -> int:
    return _clamp_int(value, 0, U16_MAX)


def tc_uv_to_wire(uv: float) -> int:
    """Thermocouple uV -> u16 with the +10 000 uV offset (clamped to the u16 range)."""
    return _u16(uv + TC_UV_OFFSET)


def tc_wire_to_uv(wire: int) -> float:
    return float(wire - TC_UV_OFFSET)


def encode_can_frame_0x100(crank_period_us: float, fuel_pulse_hz: float) -> CANFrame:
    """Encode speed & fuel parameters into CAN ID 0x100."""
    period_int = _clamp_int(crank_period_us, 0, U32_MAX)
    hz_int = _u16(fuel_pulse_hz * 10.0)
    data = struct.pack(">IH", period_int, hz_int) + b"\x00\x00"
    return CANFrame(can_id=0x100, data=data)


def encode_can_frame_0x101(egt1_uv: float, egt2_uv: float, egt3_uv: float, egt4_uv: float) -> CANFrame:
    """Encode EGT thermocouple microvolts (offset encoding) into CAN ID 0x101."""
    data = struct.pack(">HHHH", *(tc_uv_to_wire(v) for v in (egt1_uv, egt2_uv, egt3_uv, egt4_uv)))
    return CANFrame(can_id=0x101, data=data)


def encode_can_frame_0x102(cht_uv: float, oil_rtd_ohms: float, map_counts: int, oil_p_counts: int) -> CANFrame:
    """Encode CHT (offset encoding), oil RTD, MAP and oil pressure counts into CAN ID 0x102."""
    data = struct.pack(">HHHH", tc_uv_to_wire(cht_uv), _u16(oil_rtd_ohms * 10.0), _u16(map_counts), _u16(oil_p_counts))
    return CANFrame(can_id=0x102, data=data)


def encode_can_frame_0x103(fuel_press_counts: int, bus_v_counts: int, alt_i_counts: int,
                           batt_i_counts: int = 0) -> CANFrame:
    """Encode fuel-pressure, bus-voltage, alternator- and battery-current ADC counts into CAN ID 0x103."""
    data = struct.pack(">HHHH", _u16(fuel_press_counts), _u16(bus_v_counts), _u16(alt_i_counts), _u16(batt_i_counts))
    return CANFrame(can_id=0x103, data=data)


def _encode_cyl4(can_id: int, values: tuple[float, float, float, float], signed: bool) -> CANFrame:
    if len(values) != 4:
        raise ValueError("expected 4 per-cylinder values")
    if signed:
        return CANFrame(can_id=can_id, data=struct.pack(">hhhh", *(_clamp_int(v, I16_MIN, I16_MAX) for v in values)))
    return CANFrame(can_id=can_id, data=struct.pack(">HHHH", *(_u16(v) for v in values)))


def encode_can_frame_0x104(inj_pw_us: tuple[float, float, float, float]) -> CANFrame:
    """Encode injector open time per cylinder [us] into CAN ID 0x104."""
    return _encode_cyl4(0x104, inj_pw_us, signed=False)


def encode_can_frame_0x105(inj_soi_delay_us: tuple[float, float, float, float]) -> CANFrame:
    """Encode TDC-reference -> injector-open delay per cylinder [us] into CAN ID 0x105."""
    return _encode_cyl4(0x105, inj_soi_delay_us, signed=False)


def encode_can_frame_0x106(ign_delay_us: tuple[float, float, float, float]) -> CANFrame:
    """Encode coil-fire -> TDC-reference delay per cylinder [us, signed] into CAN ID 0x106."""
    return _encode_cyl4(0x106, ign_delay_us, signed=True)


def encode_can_frame_0x107(coolant_ntc_ohms: float) -> CANFrame:
    """Encode coolant NTC resistance (ohm x 10, u32) into CAN ID 0x107."""
    return CANFrame(can_id=0x107, data=struct.pack(">I", _clamp_int(coolant_ntc_ohms * 10.0, 0, U32_MAX)) + bytes(4))
