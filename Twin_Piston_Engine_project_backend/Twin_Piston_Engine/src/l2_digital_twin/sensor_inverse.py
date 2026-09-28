"""
Sensor Inverse Modelling — Original Module 5.

First processing stage of L2 Digital Twin.
Converts acquisition-level RawSignalRecord (uV, ohms, ADC counts, us, Hz) into
canonical physical engineering quantities (K, Pa, rev/min, kg/s, m/s²) wrapped in
ChannelValue with Provenance.DERIVED.

STRICT BOUNDARY CONSTRAINTS:
    - Input: RawSignalRecord (immutable)
    - Output: NormalizedSignalRecord (engineering units, Provenance.DERIVED)
    - Zero simulator internals or ground truth dependencies
    - Zero thermodynamic/mechanical digital twin derivations (Module 6)
    - Zero diagnostic, ML, advisory, or API logic
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.units import isa_pressure_altitude_m
from src.core.provenance import ChannelValidity, FlightPhase, Provenance
from src.core.schemas import SignalQuality
from src.core.sensor_physics import (
    TYPE_K_TEMP_MAX_C,
    TYPE_K_TEMP_MIN_C,
    pt100_temp_kelvin,
    thermocouple_k_to_kelvin,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord, SampleBurst
from src.l2_digital_twin.injection_model import interval_us_to_deg
from src.core.sensor_physics import ntc_temp_k

logger = get_logger(__name__)


class _NoneAsNaN:
    """Read-only view of a RawSignalRecord for the conversions: a None raw
    number reads as NaN (and a None accelerometer triple as three NaNs), so
    the existing finite/positive guards produce invalid channels instead of
    crashing. Validity itself comes from the dependency map, not the NaN."""

    def __init__(self, record: RawSignalRecord) -> None:
        self._record = record

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._record, name)
        if value is None:
            return (math.nan, math.nan, math.nan) if name == "accel_counts_xyz" else math.nan
        return value


# Raw acquisition field -> normalised channels derived from it. A channel is
# invalid if ANY raw field it depends on is flagged invalid (by the source or
# by L1 validation). Every raw data field of RawSignalRecord must appear here;
# tests/unit/test_invalid_flag_propagation.py enforces that for new fields.
RAW_FIELD_DEPENDENTS: dict[str, tuple[str, ...]] = {
    "egt_cyl1_hot_uv": ("egt_cyl_1",),
    "egt_cyl2_hot_uv": ("egt_cyl_2",),
    "egt_cyl3_hot_uv": ("egt_cyl_3",),
    "egt_cyl4_hot_uv": ("egt_cyl_4",),
    "egt_cold_c": ("egt_cyl_1", "egt_cyl_2", "egt_cyl_3", "egt_cyl_4"),
    "cht_hot_uv": ("cht_cyl_1", "cht_cyl_2", "cht_cyl_3", "cht_cyl_4"),
    "cht_cold_c": ("cht_cyl_1", "cht_cyl_2", "cht_cyl_3", "cht_cyl_4"),
    "oil_rtd_ohms": ("oil_temp",),
    "oil_p_counts": ("oil_pressure",),
    "map_counts": ("map_pressure", "boost_pressure"),
    "adc_vref_counts": ("map_pressure", "boost_pressure", "oil_pressure", "voltage", "current",
                        "battery_current", "bus_voltage_burst", "fuel_pressure"),
    # Timing intervals become angles with the instantaneous rpm (Prompt 13)
    "crank_period_us": ("rpm", "propeller_speed",
                        *(f"ignition_timing_cyl_{i}" for i in range(1, 5)),
                        *(f"injection_soi_deg_cyl_{i}" for i in range(1, 5))),
    "fuel_pulse_hz": ("fuel_flow",),
    "accel_counts_xyz": ("vibration_x", "vibration_y", "vibration_z", "vibration_rms"),
    "accel_burst_counts_x": ("vibration_burst_x",),
    "accel_burst_counts_y": ("vibration_burst_y",),
    "accel_burst_counts_z": ("vibration_burst_z",),
    "accel_burst_fs_hz": ("vibration_burst_x", "vibration_burst_y", "vibration_burst_z"),
    "crank_period_burst_us": ("crank_period_burst",),
    "ambient_temp_c": ("ambient_temp",),
    "ambient_press_pa": ("ambient_pressure", "boost_pressure", "altitude"),
    # Prompt 11 raw signals (optional; derivations added in Prompts 12-14)
    "bus_v_counts": ("voltage",),
    "bus_v_burst_counts": ("bus_voltage_burst",),
    "bus_v_burst_fs_hz": ("bus_voltage_burst",),
    "alt_i_counts": ("current",),
    "batt_i_counts": ("battery_current",),
    "fuel_press_counts": ("fuel_pressure",),
    "coolant_ntc_ohms": ("coolant_temp",),
    "inj_pw_us": tuple(f"injector_pulse_width_cyl_{i}" for i in range(1, 5)),
    "inj_soi_delay_us": (*(f"injection_timing_cyl_{i}" for i in range(1, 5)),
                         *(f"injection_soi_deg_cyl_{i}" for i in range(1, 5))),
    "ign_delay_us": tuple(f"ignition_timing_cyl_{i}" for i in range(1, 5)),
}

# Channels whose raw inputs exist but whose derivation is not implemented.
# Empty since Prompt 14 (coolant_temp).
PENDING_DERIVATION_CHANNELS: tuple[str, ...] = ()

CHANNEL_RAW_DEPENDENCIES: dict[str, tuple[str, ...]] = {}
for _raw, _chans in RAW_FIELD_DEPENDENTS.items():
    for _ch in _chans:
        CHANNEL_RAW_DEPENDENCIES[_ch] = CHANNEL_RAW_DEPENDENCIES.get(_ch, ()) + (_raw,)
MEASURED_CHANNELS: tuple[str, ...] = tuple(sorted(CHANNEL_RAW_DEPENDENCIES))


class SensorInverseModel:
    """Sensor Inverse Model for Rotax 915 iS acquisition telemetry."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.reset_state()

    def reset_state(self) -> None:
        """Clear accumulated engine running time (and the last record time)."""
        self._running_hours: float = 0.0
        self._last_timestamp: datetime | None = None

    def _accumulate_engine_hours(self, timestamp: datetime | None, running: bool) -> None:
        """Add the time since the previous record while the engine is running.
        Gaps longer than engine_hours_max_gap_s are not counted."""
        if timestamp is None:
            return
        if self._last_timestamp is not None and running:
            dt = (timestamp - self._last_timestamp).total_seconds()
            if 0.0 < dt <= self._settings.engine.engine_hours_max_gap_s:
                self._running_hours += dt / 3600.0
        self._last_timestamp = timestamp

    def convert_raw_to_engineering(
        self, raw_record: RawSignalRecord
    ) -> NormalizedSignalRecord:
        """Convert a RawSignalRecord into a NormalizedSignalRecord.

        Emits physical engineering values with Provenance.DERIVED while preserving
        signal quality and failure isolation metadata.
        """
        cal = self._settings.sensor_calibration
        r = _NoneAsNaN(raw_record)  # numbers for the conversions; None -> NaN
        raw_sq = raw_record.signal_quality
        invalid_raw_channels = set(raw_sq.invalid_channels)

        def _flagged_dependency(channel_name: str) -> str | None:
            for raw_key in CHANNEL_RAW_DEPENDENCIES.get(channel_name, ()):
                if raw_key in invalid_raw_channels:
                    return raw_key
            return None

        def _absent_dependency(channel_name: str) -> str | None:
            for raw_key in CHANNEL_RAW_DEPENDENCIES.get(channel_name, ()):
                if getattr(raw_record, raw_key) is None:
                    return raw_key
            return None

        def _make_channel(
            channel_name: str,
            eng_value: float | None,
            default_valid: bool = True,
            error_reason: str | None = None,
        ) -> ChannelValue:
            """ChannelValue whose validity follows every raw field it depends on."""
            flagged = _flagged_dependency(channel_name)
            absent = _absent_dependency(channel_name)
            is_valid = default_valid and flagged is None and absent is None
            validity_reason = ChannelValidity.VALID if is_valid else (
                ChannelValidity.MISSING if (flagged is None and absent is not None) else ChannelValidity.INVALID_RANGE)

            fault_flag = None
            if not is_valid:
                if flagged is not None:
                    fault_flag = raw_sq.invalid_reasons.get(
                        flagged, f"Raw channel {flagged} invalid or out of bounds"
                    )
                elif absent is not None and error_reason is None:
                    kind = "not instrumented" if absent in RawSignalRecord.OPTIONAL_RAW_FIELDS else "no data"
                    fault_flag = f"Raw {absent} is None ({kind})"
                else:
                    fault_flag = error_reason

            quality = raw_sq.score if is_valid else 0.0

            return ChannelValue(
                value=eng_value,
                provenance=Provenance.DERIVED,
                valid=is_valid,
                validity_reason=validity_reason,
                quality=quality,
                fault_flag=fault_flag,
            )

        # ---------------------------------------------------------------------
        # 1. Crank Period -> Engine Speed (RPM)
        # ---------------------------------------------------------------------
        rpm_valid = r.crank_period_us > 0
        rpm_val = (60.0 * 1e6) / r.crank_period_us if rpm_valid else 0.0
        rpm_channel = _make_channel(
            "rpm",
            rpm_val,
            default_valid=rpm_valid,
            error_reason="Non-positive crank period us <= 0",
        )

        # ---------------------------------------------------------------------
        # 2. Thermocouple EGT (Microvolts -> Kelvin with Cold Junction)
        # ---------------------------------------------------------------------
        # NIST ITS-90 Type K. The measured EMF is referenced to the cold
        # junction, so the cold junction's own EMF is added before inversion:
        # T_hot = E^-1(E_measured + E(T_cold)). Negative microvolts are
        # physical (hot junction colder than the cold junction). Undefined
        # results are emitted as NaN with valid=False, never a plausible value.
        def _convert_tc_kelvin(hot_uv: float, cold_c: float) -> tuple[float, bool, str | None]:
            if not (math.isfinite(hot_uv) and math.isfinite(cold_c)):
                return math.nan, False, "Non-finite thermocouple input"
            try:
                temp_k = thermocouple_k_to_kelvin(hot_uv, cold_c)
            except ValueError as exc:
                return math.nan, False, f"Type K inversion failed: {exc}"
            temp_c = temp_k - 273.15
            if not TYPE_K_TEMP_MIN_C <= temp_c <= TYPE_K_TEMP_MAX_C:
                return math.nan, False, (
                    f"Type K result {temp_c:.2f} degC outside "
                    f"[{TYPE_K_TEMP_MIN_C}, {TYPE_K_TEMP_MAX_C}] degC"
                )
            return temp_k, True, None

        def _convert_tc_egt(hot_uv: float, cold_c: float, channel: str) -> ChannelValue:
            egt_k, valid, reason = _convert_tc_kelvin(hot_uv, cold_c)
            return _make_channel(
                channel,
                egt_k,
                default_valid=valid,
                error_reason=reason,
            )

        egt1 = _convert_tc_egt(r.egt_cyl1_hot_uv, r.egt_cold_c, "egt_cyl_1")
        egt2 = _convert_tc_egt(r.egt_cyl2_hot_uv, r.egt_cold_c, "egt_cyl_2")
        egt3 = _convert_tc_egt(r.egt_cyl3_hot_uv, r.egt_cold_c, "egt_cyl_3")
        egt4 = _convert_tc_egt(r.egt_cyl4_hot_uv, r.egt_cold_c, "egt_cyl_4")

        # ---------------------------------------------------------------------
        # 3. Thermocouple CHT (Microvolts -> Kelvin with Cold Junction)
        # ---------------------------------------------------------------------
        cht_k, cht_valid, cht_reason = _convert_tc_kelvin(r.cht_hot_uv, r.cht_cold_c)
        cht_ch = _make_channel(
            "cht_cyl_1",
            cht_k,
            default_valid=cht_valid,
            error_reason=cht_reason,
        )

        # ---------------------------------------------------------------------
        # 4. Oil RTD (Resistance -> Oil Temperature Kelvin)
        # ---------------------------------------------------------------------
        # IEC 60751 Callendar-Van Dusen Pt100.
        rtd_ohms = r.oil_rtd_ohms
        rtd_reason: str | None = None
        if not math.isfinite(rtd_ohms) or rtd_ohms <= 0.0:
            oil_temp_k, rtd_valid = math.nan, False
            rtd_reason = "Non-positive RTD resistance"
        else:
            try:
                oil_temp_k, rtd_valid = pt100_temp_kelvin(rtd_ohms, cal.pt100_r0_ohms), True
            except ValueError as exc:
                oil_temp_k, rtd_valid = math.nan, False
                rtd_reason = f"Pt100 inversion failed: {exc}"
        oil_temp_ch = _make_channel(
            "oil_temp",
            oil_temp_k,
            default_valid=rtd_valid,
            error_reason=rtd_reason,
        )

        # ---------------------------------------------------------------------
        # 5. ADC Pressures (MAP & Oil Pressure)
        # ---------------------------------------------------------------------
        vref_scale = (
            r.adc_vref_counts
            if r.adc_vref_counts > 0
            else 4095
        )

        map_pa = (r.map_counts / vref_scale) * cal.map_full_scale_pa
        map_ch = _make_channel("map_pressure", map_pa)

        oil_p_pa = (r.oil_p_counts / vref_scale) * cal.oil_p_full_scale_pa
        oil_p_ch = _make_channel("oil_pressure", oil_p_pa)

        # ---------------------------------------------------------------------
        # 6. Fuel Flow (Pulse Hz -> kg/s)
        # ---------------------------------------------------------------------
        fuel_valid = r.fuel_pulse_hz >= 0.0
        fuel_kg_s = r.fuel_pulse_hz * cal.fuel_flow_kg_s_per_hz if fuel_valid else 0.0
        fuel_ch = _make_channel(
            "fuel_flow",
            fuel_kg_s,
            default_valid=fuel_valid,
            error_reason="Negative fuel pulse frequency",
        )

        # ---------------------------------------------------------------------
        # 7. Accelerometer (Counts -> m/s²)
        # ---------------------------------------------------------------------
        accel_scale = cal.accel_counts_per_m_s2 if cal.accel_counts_per_m_s2 > 0 else 10.0
        vx = r.accel_counts_xyz[0] / accel_scale
        vy = r.accel_counts_xyz[1] / accel_scale
        vz = r.accel_counts_xyz[2] / accel_scale
        vrms = math.sqrt(vx**2 + vy**2 + vz**2)

        vib_x_ch = _make_channel("vibration_x", vx)
        vib_y_ch = _make_channel("vibration_y", vy)
        vib_z_ch = _make_channel("vibration_z", vz)
        vib_rms_ch = _make_channel("vibration_rms", vrms)

        # 7b. Accelerometer and crank bursts (counts -> m/s², us -> us).
        # Absent bursts stay valid=False with no samples; nothing is padded.
        def _burst(samples: tuple[float, ...], fs_hz: float, unit: str, channel: str,
                   ok: bool, reason: str) -> SampleBurst:
            flagged = _flagged_dependency(channel)
            valid = ok and flagged is None
            return SampleBurst(
                samples=samples if valid else (),
                sample_rate_hz=fs_hz if valid else 0.0,
                unit=unit,
                provenance=Provenance.DERIVED,
                valid=valid,
                quality=raw_sq.score if valid else 0.0,
                fault_flag=None if valid else (
                    (raw_sq.invalid_reasons.get(flagged) if flagged else None) or reason
                ),
            )

        n_burst = len(raw_record.accel_burst_counts_x)
        accel_ok = n_burst > 0 and raw_record.accel_burst_fs_hz > 0.0
        accel_reason = "No accelerometer burst in RawSignalRecord"
        vib_burst_x = _burst(tuple(c / accel_scale for c in raw_record.accel_burst_counts_x),
                             raw_record.accel_burst_fs_hz, "m/s^2", "vibration_burst_x",
                             accel_ok, accel_reason)
        vib_burst_y = _burst(tuple(c / accel_scale for c in raw_record.accel_burst_counts_y),
                             raw_record.accel_burst_fs_hz, "m/s^2", "vibration_burst_y",
                             accel_ok, accel_reason)
        vib_burst_z = _burst(tuple(c / accel_scale for c in raw_record.accel_burst_counts_z),
                             raw_record.accel_burst_fs_hz, "m/s^2", "vibration_burst_z",
                             accel_ok, accel_reason)
        crank_burst = _burst(tuple(float(p) for p in raw_record.crank_period_burst_us), 0.0, "us",
                             "crank_period_burst", len(raw_record.crank_period_burst_us) > 0,
                             "No crank period burst in RawSignalRecord")

        # ---------------------------------------------------------------------
        # 8. Ambient Conditions
        # ---------------------------------------------------------------------
        amb_t_k = r.ambient_temp_c + 273.15
        amb_temp_ch = _make_channel("ambient_temp", amb_t_k)

        amb_press_pa = r.ambient_press_pa
        amb_press_ch = _make_channel("ambient_pressure", amb_press_pa)

        # ---------------------------------------------------------------------
        # 9. Derived context channels
        # ---------------------------------------------------------------------
        amb_p_ok = math.isfinite(amb_press_pa) and amb_press_pa > 0.0
        # Boost = MAP - ambient (gauge pressure over ambient), not MAP itself.
        boost_press_ch = _make_channel("boost_pressure", map_pa - amb_press_pa, default_valid=amb_p_ok,
                                       error_reason="Ambient pressure not positive")
        # Pressure altitude by inverting ISA (troposphere only).
        p_alt = isa_pressure_altitude_m(amb_press_pa) if amb_p_ok else None
        alt_ch = _make_channel("altitude", p_alt, default_valid=p_alt is not None,
                               error_reason="Ambient pressure outside ISA troposphere (-2 000..11 000 m)")
        # Propeller speed through the gearbox (ratio from engine config).
        ratio = self._settings.engine.gearbox_ratio
        prop_rpm_ch = _make_channel(
            "propeller_speed",
            rpm_val / ratio if (ratio and ratio > 0.0) else None,
            default_valid=rpm_valid and ratio is not None and ratio > 0.0,
            error_reason=("Gearbox ratio not configured (engine.gearbox_ratio, VERIFY against "
                          "Rotax 915 iS Operators Manual)") if not ratio else "Non-positive crank period us <= 0",
        )
        # Engine hours: running time accumulated from record timestamps while rpm > 0.
        self._accumulate_engine_hours(raw_record.timestamp, rpm_channel.valid and rpm_val > 0.0)
        hours0 = self._settings.engine.engine_hours_at_install
        hours_ch = ChannelValue(
            value=(hours0 + self._running_hours) if hours0 is not None else None,
            provenance=Provenance.DERIVED,
            valid=hours0 is not None,
            validity_reason=ChannelValidity.VALID if hours0 is not None else ChannelValidity.MISSING,
            quality=1.0 if hours0 is not None else 0.0,
            fault_flag=None if hours0 is not None else (
                f"engine_hours_at_install not configured; {self._running_hours:.4f} h accumulated since "
                "monitoring started"),
        )

        # ---------------------------------------------------------------------
        # 10. Channels with no raw sensor in RawSignalRecord: invalid, never
        #     substituted (SRD-INT-004). Prompts 12-14 add the real signals.
        # ---------------------------------------------------------------------
        def _not_instrumented(what: str) -> ChannelValue:
            return ChannelValue(value=None, provenance=Provenance.DERIVED, valid=False,
                                validity_reason=ChannelValidity.MISSING, quality=0.0,
                                fault_flag=f"No raw {what} channel in RawSignalRecord (not instrumented)")

        iat_ch = _not_instrumented("intake air temperature")
        wastegate_ch = _not_instrumented("wastegate position")
        throttle_ch = _not_instrumented("throttle position")

        # Channels fed by the optional Prompt 11 raw signals. No derivation yet:
        # valid=False, with the reason saying whether the raw input is absent
        # (not instrumented) or present (derivation pending); a source flag on
        # the raw input takes precedence (via _make_channel).
        def _pending(channel: str) -> ChannelValue:
            deps = CHANNEL_RAW_DEPENDENCIES[channel]
            absent = [d for d in deps if getattr(raw_record, d) is None]
            reason = (f"Raw {', '.join(absent)} is None (not instrumented)" if absent
                      else f"Raw {', '.join(deps)} present; derivation not implemented yet (Prompts 12-14)")
            return _make_channel(channel, None, default_valid=False, error_reason=reason)

        pending = {name: _pending(name) for name in PENDING_DERIVATION_CHANNELS}

        # Coolant (Prompt 14): NTC ohms -> K by Steinhart-Hart. Without a liquid
        # coolant circuit the channel is unavailable, never inferred (SRD-FUN-045).
        ntc = raw_record.coolant_ntc_ohms
        if not self._settings.engine.has_coolant_circuit:
            coolant_ch = ChannelValue(
                value=None, provenance=Provenance.DERIVED, valid=False, validity_reason=ChannelValidity.MISSING,
                quality=0.0, fault_flag="No liquid coolant circuit (engine.has_coolant_circuit = False): "
                                        "coolant parameters unavailable")
        else:
            ntc_ok = ntc is not None and math.isfinite(ntc) and ntc > 0.0
            coolant_ch = _make_channel(
                "coolant_temp",
                ntc_temp_k(ntc, cal.ntc_sh_a, cal.ntc_sh_b, cal.ntc_sh_c) if ntc_ok else None,
                default_valid=ntc_ok,
                error_reason=None if (ntc is None or ntc_ok) else "Non-positive or non-finite NTC resistance")

        # ---------------------------------------------------------------------
        # 11. Electrical (Prompt 12): ratiometric like the pressure sensors
        #     (SRD-FUN-004). value = counts / vref * full scale.
        # ---------------------------------------------------------------------
        vref_ok = math.isfinite(r.adc_vref_counts) and r.adc_vref_counts > 0
        vref = r.adc_vref_counts if vref_ok else math.nan

        def _ratiometric(counts: int | None, full_scale: float, zero_fraction: float = 0.0) -> float | None:
            if counts is None or not vref_ok:
                return None  # not NaN: None keeps records comparable and serialises as null
            return (counts / vref - zero_fraction) * full_scale

        vref_reason = None if vref_ok else "ADC reference invalid"
        voltage_ch = _make_channel("voltage", _ratiometric(raw_record.bus_v_counts, cal.bus_v_full_scale_v),
                                   default_valid=vref_ok, error_reason=vref_reason)
        current_ch = _make_channel("current", _ratiometric(raw_record.alt_i_counts, cal.alt_i_full_scale_a),
                                   default_valid=vref_ok, error_reason=vref_reason)
        batt_i_ch = _make_channel(
            "battery_current",
            _ratiometric(raw_record.batt_i_counts, cal.batt_i_full_scale_a, cal.batt_i_zero_fraction),
            default_valid=vref_ok, error_reason=vref_reason)
        bus_burst_counts = raw_record.bus_v_burst_counts or ()
        bus_v_burst = _burst(
            tuple(c / vref * cal.bus_v_full_scale_v for c in bus_burst_counts) if vref_ok else (),
            raw_record.bus_v_burst_fs_hz or 0.0, "V", "bus_voltage_burst",
            vref_ok and len(bus_burst_counts) > 0 and (raw_record.bus_v_burst_fs_hz or 0.0) > 0.0,
            "No bus-voltage burst in RawSignalRecord" if vref_ok else "ADC reference invalid")
        # ---------------------------------------------------------------------
        # 12. Injection, ignition and fuel rail (Prompt 13). Timer intervals are
        #     kept as intervals [s] and converted to crank angle with this
        #     record's rpm: deg = delay_us x rpm x 6e-6. Fuel pressure is a
        #     ratiometric gauge sensor [Pa].
        # ---------------------------------------------------------------------
        fuel_press_ch = _make_channel(
            "fuel_pressure", _ratiometric(raw_record.fuel_press_counts, cal.fuel_press_full_scale_pa),
            default_valid=vref_ok, error_reason=vref_reason)
        rpm_reason = None if rpm_valid else "rpm invalid; interval cannot be converted to crank angle"
        timing: dict[str, ChannelValue] = {}
        for i in range(4):
            pw = raw_record.inj_pw_us[i] if raw_record.inj_pw_us is not None else None
            soi = raw_record.inj_soi_delay_us[i] if raw_record.inj_soi_delay_us is not None else None
            ign = raw_record.ign_delay_us[i] if raw_record.ign_delay_us is not None else None
            timing[f"injector_pulse_width_cyl_{i + 1}"] = _make_channel(
                f"injector_pulse_width_cyl_{i + 1}", pw * 1e-6 if pw is not None else None,
                default_valid=pw is not None and pw >= 0.0,
                error_reason=None if (pw is None or pw >= 0.0) else "negative pulse width")
            timing[f"injection_timing_cyl_{i + 1}"] = _make_channel(
                f"injection_timing_cyl_{i + 1}", soi * 1e-6 if soi is not None else None)
            timing[f"injection_soi_deg_cyl_{i + 1}"] = _make_channel(
                f"injection_soi_deg_cyl_{i + 1}",
                interval_us_to_deg(soi, rpm_val) if (soi is not None and rpm_valid) else None,
                default_valid=rpm_valid, error_reason=rpm_reason if soi is not None else None)
            timing[f"ignition_timing_cyl_{i + 1}"] = _make_channel(
                f"ignition_timing_cyl_{i + 1}",
                interval_us_to_deg(ign, rpm_val) if (ign is not None and rpm_valid) else None,
                default_valid=rpm_valid, error_reason=rpm_reason if ign is not None else None)

        # RawSignalRecord has no lambda channel, so lambda cannot come from L1.
        # It is derived from air and fuel mass flows in ThermodynamicTwin.
        lambda_ch = ChannelValue(
            value=math.nan,
            provenance=Provenance.DERIVED,
            valid=False,
            validity_reason=ChannelValidity.MISSING,
            quality=0.0,
            fault_flag="No lambda channel in RawSignalRecord; lambda is derived in ThermodynamicTwin",
        )

        measured = {
            "rpm": rpm_channel, "egt_cyl_1": egt1, "egt_cyl_2": egt2, "egt_cyl_3": egt3, "egt_cyl_4": egt4,
            "cht_cyl_1": cht_ch, "cht_cyl_2": cht_ch, "cht_cyl_3": cht_ch, "cht_cyl_4": cht_ch,
            "oil_temp": oil_temp_ch, "oil_pressure": oil_p_ch, "map_pressure": map_ch, "boost_pressure": boost_press_ch,
            "fuel_flow": fuel_ch, "propeller_speed": prop_rpm_ch, "altitude": alt_ch,
            "vibration_x": vib_x_ch, "vibration_y": vib_y_ch, "vibration_z": vib_z_ch, "vibration_rms": vib_rms_ch,
            "vibration_burst_x": vib_burst_x, "vibration_burst_y": vib_burst_y, "vibration_burst_z": vib_burst_z,
            "crank_period_burst": crank_burst,
            "ambient_temp": amb_temp_ch, "ambient_pressure": amb_press_ch,
            "voltage": voltage_ch, "current": current_ch, "battery_current": batt_i_ch,
            "bus_voltage_burst": bus_v_burst,
            "fuel_pressure": fuel_press_ch,
            **timing,
            "coolant_temp": coolant_ch,
            **pending,
        }
        assert set(measured) == set(MEASURED_CHANNELS), "measured channels out of sync with RAW_FIELD_DEPENDENTS"

        # Coverage counts a channel only if it is instrumented in this record
        # (every raw input present, i.e. not None) or the source flagged one of
        # its inputs. A None raw field means "not instrumented", not "failed".
        def _counted(channel: str) -> bool:
            deps = CHANNEL_RAW_DEPENDENCIES[channel]
            return (all(getattr(raw_record, d) is not None for d in deps)
                    or _flagged_dependency(channel) is not None)

        counted = [name for name in measured if _counted(name)]
        invalid_measured = sorted(name for name in counted if not measured[name].valid)
        coverage = 1.0 - len(invalid_measured) / len(counted) if counted else 0.0

        return NormalizedSignalRecord(
            timestamp=raw_record.timestamp,
            sequence_number=raw_record.sequence_number,
            source_id=f"sensor_inverse:{raw_record.source_type.value}",
            provenance=Provenance.DERIVED,
            integrity_hash=raw_record.integrity_hash,
            flight_phase=FlightPhase.GROUND,
            rpm=rpm_channel,
            map_pressure=map_ch,
            throttle_position=throttle_ch,
            egt_cyl_1=egt1,
            egt_cyl_2=egt2,
            egt_cyl_3=egt3,
            egt_cyl_4=egt4,
            cht_cyl_1=cht_ch,
            cht_cyl_2=cht_ch,
            cht_cyl_3=cht_ch,
            cht_cyl_4=cht_ch,
            oil_temp=oil_temp_ch,
            oil_pressure=oil_p_ch,
            coolant_temp=coolant_ch,
            fuel_flow=fuel_ch,
            fuel_pressure=fuel_press_ch,
            intake_air_temp=iat_ch,
            ambient_pressure=amb_press_ch,
            ambient_temp=amb_temp_ch,
            voltage=voltage_ch,
            current=current_ch,
            vibration_x=vib_x_ch,
            vibration_y=vib_y_ch,
            vibration_z=vib_z_ch,
            vibration_rms=vib_rms_ch,
            vibration_burst_x=vib_burst_x,
            vibration_burst_y=vib_burst_y,
            vibration_burst_z=vib_burst_z,
            crank_period_burst=crank_burst,
            bus_voltage_burst=bus_v_burst,
            battery_current=batt_i_ch,
            propeller_speed=prop_rpm_ch,
            boost_pressure=boost_press_ch,
            wastegate_duty=wastegate_ch,
            lambda_sensor=lambda_ch,
            **timing,
            altitude=alt_ch,
            engine_hours=hours_ch,
            invalid_channels=invalid_measured,
            channel_coverage=coverage,
        )


def convert_raw_to_engineering_state(
    raw_record: RawSignalRecord,
    settings: AppSettings | None = None,
) -> NormalizedSignalRecord:
    """Convenience function for Module 5 sensor inverse modelling."""
    model = SensorInverseModel(settings)
    return model.convert_raw_to_engineering(raw_record)
