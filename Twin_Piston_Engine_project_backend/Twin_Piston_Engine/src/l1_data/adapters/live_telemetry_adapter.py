"""
Live Telemetry Adapter — Interface & CAN 2.0 parser for live engine hardware.

Parses incoming CAN 2.0 bus frames (or mock transport) into canonical
RawSignalRecord instances tagged with Provenance.REAL.

Zero hardware dependency in downstream layers — works with real hardware interfaces
or isolated MockCANTransport.
"""

from __future__ import annotations

import struct
from typing import Any
from datetime import datetime, timezone

from src.core.exceptions import AdapterConnectionError
from src.core.provenance import Provenance
from src.core.schemas import SignalQuality
from src.l1_data.adapters.base import SourceAdapter
from src.l1_data.adapters.can_transport import (
    TC_UV_OFFSET,
    U16_MAX,
    U32_MAX,
    CANFrame,
    CANTransportProtocol,
    MockCANTransport,
    tc_wire_to_uv,
)
from src.l1_data.raw_signal_record import RawSignalRecord


# Raw fields carried by each CAN frame (layouts in can_transport).
FRAME_FIELDS: dict[int, tuple[str, ...]] = {
    0x100: ("crank_period_us", "fuel_pulse_hz"),
    0x101: ("egt_cyl1_hot_uv", "egt_cyl2_hot_uv", "egt_cyl3_hot_uv", "egt_cyl4_hot_uv"),
    0x102: ("cht_hot_uv", "oil_rtd_ohms", "map_counts", "oil_p_counts"),
    0x103: ("fuel_press_counts", "bus_v_counts", "alt_i_counts", "batt_i_counts"),
    0x104: ("inj_pw_us",),
    0x105: ("inj_soi_delay_us",),
    0x106: ("ign_delay_us",),
    0x107: ("coolant_ntc_ohms",),
}
# Raw fields no CAN frame carries: always None from this adapter.
NOT_ON_CAN: tuple[str, ...] = (
    "egt_cold_c", "cht_cold_c", "adc_vref_counts", "accel_counts_xyz", "ambient_temp_c", "ambient_press_pa",
)


class LiveTelemetryAdapter(SourceAdapter):
    """Adapter for live engine telemetry via CAN 2.0 bus or mock transport.

    All output is emitted as canonical RawSignalRecords tagged with Provenance.REAL.

    Usage:
        transport = MockCANTransport()
        adapter = LiveTelemetryAdapter(transport=transport)
        async with adapter:
            raw_record = await adapter.read_next()
    """

    def __init__(
        self,
        transport: CANTransportProtocol | None = None,
        protocol: str = "can2.0",
        interface: str = "can0",
        baud_rate: int = 500000,
    ) -> None:
        self._transport = transport or MockCANTransport()
        self._protocol = protocol
        self._interface = interface
        self._baud_rate = baud_rate
        self._connected = False
        self._sequence = 0

        # Accelerometer bursts: the CAN stream parsed below (IDs 0x100-0x102)
        # carries NO accelerometer samples, so records from this adapter have
        # no accel_burst_* data and L2 reports every spectral vibration
        # feature as valid=False. If a CAN frame delivering accelerometer
        # samples AT THE VIBRATION RATE (e.g. 2048 Hz) is added, assemble
        # accel_burst_counts_* from it with the ring buffer in
        # physics/B01_vibration_ring_buffer.py (move it to src/core first: L1
        # must not import L2). NEVER feed that buffer at the 1 Hz record rate:
        # 2048 record-rate samples span ~34 minutes, not a 1 s window.

        # Latest value per raw field, None until the frame carrying it arrives.
        # No placeholder values: a record assembled before a frame group has
        # been received carries None for that group, flagged invalid.
        self._latest: dict[str, Any] = {f: None for fields in FRAME_FIELDS.values() for f in fields}
        self._received: set[int] = set()
        # raw field -> reason, for values at an encoding limit (saturated etc.)
        self._saturated: dict[str, str] = {}

    async def connect(self) -> None:
        """Establish connection to CAN bus interface."""
        try:
            await self._transport.connect()
            self._connected = True
        except Exception as e:
            raise AdapterConnectionError(f"Failed to connect live telemetry transport: {e}") from e

    async def read_next(self) -> RawSignalRecord | None:
        """Read and parse incoming CAN frame into canonical RawSignalRecord.

        Returns:
            RawSignalRecord with Provenance.REAL, or None if no frame is available.
        """
        if not self._connected:
            return None

        frame = await self._transport.read_frame()
        if frame is None:
            return None

        self._parse_can_frame(frame)
        self._sequence += 1

        flags: dict[str, str] = {}
        for can_id, fields in FRAME_FIELDS.items():
            if can_id not in self._received:
                for f in fields:
                    flags[f] = f"No CAN frame 0x{can_id:03X} received yet"
        for f in NOT_ON_CAN:
            flags[f] = "Not carried on the CAN bus (no data)"
        flags.update(self._saturated)

        raw_rec = RawSignalRecord(
            timestamp=frame.timestamp,
            sequence_number=self._sequence,
            source_type=Provenance.REAL,
            **self._latest,
            **{f: None for f in NOT_ON_CAN},
            signal_quality=SignalQuality(
                score=max(0.0, 1.0 - len(flags) / (len(self._latest) + len(NOT_ON_CAN))),
                valid=not flags,
                invalid_channels=sorted(flags),
                invalid_reasons=flags,
            ),
        )

        return raw_rec.model_copy(update={"integrity_hash": raw_rec.compute_integrity_hash()})

    async def disconnect(self) -> None:
        """Disconnect from CAN bus transport."""
        await self._transport.disconnect()
        self._connected = False

    def _parse_can_frame(self, frame: CANFrame) -> None:
        """Unpack raw binary payload based on CAN ID (layouts in can_transport)."""
        cid, d = frame.can_id, frame.data
        if cid not in FRAME_FIELDS or len(d) < 8 and cid != 0x100 or len(d) < 6:
            return
        L = self._latest
        if cid == 0x100:
            period_int, hz_int = struct.unpack(">IH", d[:6])
            L["crank_period_us"] = float(period_int) if period_int > 0 else None
            L["fuel_pulse_hz"] = float(hz_int) / 10.0
            self._limit("crank_period_us", period_int in (0, U32_MAX), "CAN 0x100 crank period 0 or saturated")
        elif cid == 0x101:
            wires = struct.unpack(">HHHH", d[:8])
            for i, w in enumerate(wires, start=1):
                L[f"egt_cyl{i}_hot_uv"] = tc_wire_to_uv(w)
                self._limit(f"egt_cyl{i}_hot_uv", w in (0, U16_MAX),
                            f"CAN 0x101 EGT{i} at encoding limit ({-TC_UV_OFFSET} / {U16_MAX - TC_UV_OFFSET} uV)")
        elif cid == 0x102:
            c, rtd, m_cnt, o_cnt = struct.unpack(">HHHH", d[:8])
            L["cht_hot_uv"] = tc_wire_to_uv(c)
            L["oil_rtd_ohms"] = rtd / 10.0
            L["map_counts"], L["oil_p_counts"] = m_cnt, o_cnt
            self._limit("cht_hot_uv", c in (0, U16_MAX), "CAN 0x102 CHT at encoding limit")
        elif cid == 0x103:
            fp, v, i, bi = struct.unpack(">HHHH", d[:8])
            L["fuel_press_counts"], L["bus_v_counts"], L["alt_i_counts"], L["batt_i_counts"] = fp, v, i, bi
        elif cid in (0x104, 0x105):
            vals = struct.unpack(">HHHH", d[:8])
            field = FRAME_FIELDS[cid][0]
            L[field] = tuple(float(x) for x in vals)
            self._limit(field, max(vals) >= U16_MAX, f"CAN 0x{cid:03X} {field} saturated (0xFFFF)")
        elif cid == 0x106:
            vals = struct.unpack(">hhhh", d[:8])
            L["ign_delay_us"] = tuple(float(x) for x in vals)
            self._limit("ign_delay_us", any(x in (-32768, 32767) for x in vals),
                        "CAN 0x106 ign_delay_us at int16 limit")
        elif cid == 0x107:
            (ntc,) = struct.unpack(">I", d[:4])
            L["coolant_ntc_ohms"] = ntc / 10.0
            if ntc == 0:
                self._saturated["coolant_ntc_ohms"] = "CAN 0x107 coolant NTC reads 0 ohm (short circuit)"
            else:
                self._limit("coolant_ntc_ohms", ntc >= U32_MAX, "CAN 0x107 coolant NTC saturated")
        self._received.add(cid)

    def _limit(self, field: str, at_limit: bool, reason: str) -> None:
        """A value at an encoding limit is flagged invalid instead of being
        passed on as a reading."""
        if at_limit:
            self._saturated[field] = reason
        else:
            self._saturated.pop(field, None)

    @property
    def source_type(self) -> Provenance:
        return Provenance.REAL

    @property
    def source_id(self) -> str:
        return f"live_telemetry:{self._protocol}:{self._interface}"

    @property
    def is_connected(self) -> bool:
        return self._connected
