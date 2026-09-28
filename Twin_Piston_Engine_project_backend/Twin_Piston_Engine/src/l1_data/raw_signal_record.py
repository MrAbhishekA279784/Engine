"""
Canonical RawSignalRecord — Acquisition-level telemetry schema.

Contains ONLY raw sensor/acquisition quantities (microvolts, ohms, ADC counts,
microseconds, pulse frequencies), NEVER derived engineering quantities (RPM, Pa, K).

This record preserves physical raw signal fidelity and carries complete
provenance and signal quality metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.provenance import Provenance
from src.core.schemas import SignalQuality, TimestampedRecord


class RawSignalRecord(TimestampedRecord):
    """Canonical raw signal record for hardware acquisition layer.

    All fields are raw acquisition units:
        - Thermocouples: microvolts [uV] and cold-junction temp [°C]
        - RTD: resistance [ohms]
        - Pressure / ADC: raw ADC counts
        - Engine speed: crank period [microseconds]
        - Fuel flow: pulse frequency [Hz]
        - Accelerometer: 3-axis ADC counts, plus an optional per-record burst
          of N samples per axis at the vibration sample rate
        - Crank: optional burst of the last K per-revolution periods [us]
        - Ambient: ambient temp [°C] and pressure [Pa]
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of hardware sampling",
    )
    sequence_number: int = Field(ge=0, description="Monotonically increasing packet counter")
    source_type: Provenance = Field(
        default=Provenance.REAL,
        description="Origin of raw data (REAL / SIMULATED / CSV_REPLAY)",
    )

    # Core raw signals are REQUIRED but NULLABLE: every record must state each
    # one, and None means "no data" (e.g. no CAN frame received yet, or not
    # carried by this source). L2 marks every channel derived from a None
    # field valid=False; nothing is substituted.

    # Thermocouple Exhaust Gas Temperature raw signals (microvolts)
    egt_cyl1_hot_uv: float | None = Field(description="Cylinder 1 EGT thermocouple hot junction [uV]")
    egt_cyl2_hot_uv: float | None = Field(description="Cylinder 2 EGT thermocouple hot junction [uV]")
    egt_cyl3_hot_uv: float | None = Field(description="Cylinder 3 EGT thermocouple hot junction [uV]")
    egt_cyl4_hot_uv: float | None = Field(description="Cylinder 4 EGT thermocouple hot junction [uV]")
    egt_cold_c: float | None = Field(description="EGT cold junction reference temperature [°C]")

    # Thermocouple Cylinder Head Temperature raw signals (microvolts)
    cht_hot_uv: float | None = Field(description="CHT thermocouple hot junction [uV]")
    cht_cold_c: float | None = Field(description="CHT cold junction reference temperature [°C]")

    # RTD Resistance (Oil temperature)
    oil_rtd_ohms: float | None = Field(description="Oil temperature RTD resistance [ohms]")

    # ADC counts for pressure and reference
    oil_p_counts: int | None = Field(description="Oil pressure sensor ADC counts")
    map_counts: int | None = Field(description="Manifold absolute pressure sensor ADC counts")
    adc_vref_counts: int | None = Field(description="ADC reference voltage channel counts")

    # Crankshaft timing and fuel pulse
    crank_period_us: float | None = Field(gt=0.0, description="Crankshaft revolution period [us]")
    fuel_pulse_hz: float | None = Field(ge=0.0, description="Fuel flow turbine pulse frequency [Hz]")

    # Accelerometer raw counts (X, Y, Z)
    accel_counts_xyz: tuple[int, int, int] | None = Field(description="3-axis accelerometer ADC counts (X, Y, Z)")

    # High-rate bursts (optional; empty for sources that cannot supply them).
    # Spectral vibration features need a full window at the vibration sample
    # rate; they cannot be assembled from one sample per 1 Hz record.
    accel_burst_counts_x: tuple[int, ...] = Field(
        default=(), description="X-axis accelerometer burst, N ADC counts at accel_burst_fs_hz"
    )
    accel_burst_counts_y: tuple[int, ...] = Field(
        default=(), description="Y-axis accelerometer burst, N ADC counts at accel_burst_fs_hz"
    )
    accel_burst_counts_z: tuple[int, ...] = Field(
        default=(), description="Z-axis accelerometer burst, N ADC counts at accel_burst_fs_hz"
    )
    accel_burst_fs_hz: float = Field(
        default=0.0, ge=0.0, description="Accelerometer burst sample rate [Hz]; 0 when no burst"
    )
    crank_period_burst_us: tuple[float, ...] = Field(
        default=(), description="Last K per-revolution crank periods [us], oldest first"
    )

    # --- Prompt 11 raw signals. Optional: None means "not instrumented". ---
    # Electrical (10 Hz; burst at 2048 Hz)
    bus_v_counts: int | None = Field(default=None, ge=0, description="Bus voltage via divider, ADC counts")
    alt_i_counts: int | None = Field(default=None, ge=0, description="Alternator current (Hall/shunt amp), ADC counts")
    # Battery-lead current (bidirectional Hall/shunt, zero at mid-scale). Needed
    # to observe battery internal resistance: -dV/dI_batt over load steps.
    batt_i_counts: int | None = Field(default=None, ge=0, description="Battery current (bidirectional), ADC counts")
    bus_v_burst_counts: tuple[int, ...] | None = Field(
        default=None, description="Bus voltage burst for ripple analysis, ADC counts at bus_v_burst_fs_hz"
    )
    bus_v_burst_fs_hz: float | None = Field(default=None, gt=0.0, description="Bus voltage burst sample rate [Hz]")
    # Fuel and injection: time intervals against the crank TDC reference (per
    # cycle, reported at 10 Hz), cylinders 1..4
    inj_pw_us: tuple[float, float, float, float] | None = Field(
        default=None, description="Injector open time per cylinder [us]"
    )
    inj_soi_delay_us: tuple[float, float, float, float] | None = Field(
        default=None, description="TDC-reference edge -> injector open, per cylinder [us]"
    )
    ign_delay_us: tuple[float, float, float, float] | None = Field(
        default=None, description="Coil fire edge -> TDC-reference edge, per cylinder [us]"
    )
    fuel_press_counts: int | None = Field(default=None, ge=0, description="Fuel rail pressure transducer, ADC counts")
    # Cooling (1 Hz)
    # ge=0: a shorted sensor (0 ohm) is a real raw reading; it is flagged by
    # the acquisition/validation layer, not rejected by the schema.
    coolant_ntc_ohms: float | None = Field(default=None, ge=0.0, description="Coolant NTC thermistor resistance [ohms]")

    # Ambient conditions
    ambient_temp_c: float | None = Field(description="Ambient air temperature [°C]")
    ambient_press_pa: float | None = Field(description="Ambient atmospheric pressure [Pa]")

    # Integrity & Quality metadata
    integrity_hash: str = Field(default="", description="SHA-256 payload integrity signature")
    signal_quality: SignalQuality = Field(default_factory=SignalQuality, description="Signal quality metadata")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _check_bursts(self) -> RawSignalRecord:
        n = len(self.accel_burst_counts_x)
        if not (n == len(self.accel_burst_counts_y) == len(self.accel_burst_counts_z)):
            raise ValueError("accel burst axes must have equal length")
        if n > 0 and self.accel_burst_fs_hz <= 0.0:
            raise ValueError("accel_burst_fs_hz must be > 0 when an accel burst is present")
        if any(not (p > 0.0) for p in self.crank_period_burst_us):
            raise ValueError("crank_period_burst_us values must be > 0")
        if self.bus_v_burst_counts is not None:
            if len(self.bus_v_burst_counts) > 0 and self.bus_v_burst_fs_hz is None:
                raise ValueError("bus_v_burst_fs_hz is required when bus_v_burst_counts is present")
        return self

    OPTIONAL_RAW_FIELDS: ClassVar[tuple[str, ...]] = (
        "bus_v_counts", "alt_i_counts", "batt_i_counts", "bus_v_burst_counts", "bus_v_burst_fs_hz",
        "inj_pw_us", "inj_soi_delay_us", "ign_delay_us", "fuel_press_counts", "coolant_ntc_ohms",
    )

    def burst_payload(self) -> dict[str, object]:
        """Burst and optional (Prompt 11) fields for hashing/signing (SRD-SEC-001).

        Absent bursts and None optional fields are omitted, which keeps hashes
        and signatures of records without them identical to those produced
        before the fields existed.
        """
        payload: dict[str, object] = {}
        if self.accel_burst_counts_x:
            payload["accel_burst_x"] = list(self.accel_burst_counts_x)
            payload["accel_burst_y"] = list(self.accel_burst_counts_y)
            payload["accel_burst_z"] = list(self.accel_burst_counts_z)
            payload["accel_burst_fs_hz"] = self.accel_burst_fs_hz
        if self.crank_period_burst_us:
            payload["crank_period_burst_us"] = list(self.crank_period_burst_us)
        for name in self.OPTIONAL_RAW_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = list(value) if isinstance(value, tuple) else value
        return payload

    def compute_integrity_hash(self) -> str:
        """Compute SHA-256 integrity hash of raw payload fields."""
        payload = {
            "seq": self.sequence_number,
            "egt1": self.egt_cyl1_hot_uv,
            "egt2": self.egt_cyl2_hot_uv,
            "egt3": self.egt_cyl3_hot_uv,
            "egt4": self.egt_cyl4_hot_uv,
            "egt_cold": self.egt_cold_c,
            "cht": self.cht_hot_uv,
            "cht_cold": self.cht_cold_c,
            "oil_rtd": self.oil_rtd_ohms,
            "oil_p": self.oil_p_counts,
            "map": self.map_counts,
            "vref": self.adc_vref_counts,
            "crank_us": self.crank_period_us,
            "fuel_hz": self.fuel_pulse_hz,
            "accel": list(self.accel_counts_xyz) if self.accel_counts_xyz is not None else None,
            "amb_t": self.ambient_temp_c,
            "amb_p": self.ambient_press_pa,
        }
        payload.update(self.burst_payload())
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()
