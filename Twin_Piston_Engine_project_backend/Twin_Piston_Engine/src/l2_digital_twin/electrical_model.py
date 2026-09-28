"""
Electrical Model — charging system and battery health (Prompt 12).

L2 stage. Consumes the NormalizedSignalRecord channels voltage (bus),
current (alternator), battery_current (positive = discharge),
bus_voltage_burst and rpm. Stateful (load-step history, residual median);
call evaluate() once per record in time order.

Derivations
    expected bus voltage   battery OCV below cut-in (battery carries the bus),
                           regulator setpoint above cut-in (engine rpm x
                           engine.generator_drive_ratio vs electrical.cut_in_rpm)
    charging_residual_v    measured - expected
    battery_resistance     R_int = -dV / dI_batt across a load step. Valid only
                           when |dI| >= r_est_min_di_a, rpm steady, samples
                           consecutive, and the battery supplies the step (below
                           cut-in, or alternator current ~ 0). With the regulator
                           in control the step is absorbed by the alternator:
                           "regulator holding bus; not observable". Robust
                           median over the last N accepted steps.
    voltage_ripple_pct     RMS of the in-band AC part of the voltage burst /
                           burst mean x 100. The analysis band is checked
                           against the burst Nyquist limit (M-05) first.
    ripple order           dominant ripple frequency / generator rev/s. A
                           shaft-locked rectifier ripple has an integer order
                           at every rpm (rpm-scaling check, needs no pole count).
    dominant frequency     valid only when engine.generator_poles is known, the
                           expected ripple (2 x phases x f_e) is below Nyquist,
                           and the peak matches it or the open-diode f_e line.
    EHI                    component factors f = 0.5 ** d (d = 1 at the alarm
                           threshold), weighted log-space product re-normalised
                           over valid components; coverage = valid weight share.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from src.core.config import AppSettings, get_settings
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import ElectricalState, ProvenanceTaggedValue, make_tagged
from src.core.sensor_physics.M05_nyquist_guard import band_is_representable
from src.l1_data.signal_record import NormalizedSignalRecord

NOT_OBSERVABLE_REGULATED = "regulator holding bus; not observable"


def _invalid(reason: str) -> ProvenanceTaggedValue[float | None]:
    return ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False,
                                 validity_reason=ChannelValidity.MISSING, quality=0.0, fault_flag=reason)


def _band(dev: float, warning: float, alarm: float) -> DiagnosticStatus:
    if dev >= alarm:
        return DiagnosticStatus.CRITICAL
    if dev >= warning:
        return DiagnosticStatus.WARNING
    return DiagnosticStatus.NORMAL


_RANK = {DiagnosticStatus.INVALID: 0, DiagnosticStatus.NORMAL: 1,
         DiagnosticStatus.WARNING: 2, DiagnosticStatus.CRITICAL: 3}


class ElectricalModel:
    """Charging-system and battery health from bus voltage and currents."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg = self._settings.electrical
        self._eng = self._settings.engine
        self.reset_state()

    def reset_state(self) -> None:
        self._prev: dict[str, Any] | None = None
        self._r_steps: deque[float] = deque(maxlen=self._cfg.r_est_median_n)
        self._residuals: deque[float] = deque(maxlen=self._cfg.residual_median_n)

    # ------------------------------------------------------------------ helpers
    def _generator_rev_s(self, rpm: float) -> float:
        return rpm * self._eng.generator_drive_ratio / 60.0

    def _regime(self, record: NormalizedSignalRecord) -> str:
        if not record.rpm.valid or record.rpm.value is None:
            return "UNKNOWN"
        gen_rpm = record.rpm.value * self._eng.generator_drive_ratio
        return "REGULATED" if gen_rpm >= self._cfg.cut_in_rpm else "BATTERY"

    # ------------------------------------------------------------------ R_int
    def _r_int(self, record: NormalizedSignalRecord, regime: str, ts: datetime | None
               ) -> tuple[ProvenanceTaggedValue[float | None], list[str]]:
        c = self._cfg
        v, ib, ia, rpm = record.voltage, record.battery_current, record.current, record.rpm
        evidence: list[str] = []
        if not (v.valid and ib.valid and rpm.valid):
            missing = [n for n, ch in (("voltage", v), ("battery_current", ib), ("rpm", rpm)) if not ch.valid]
            self._prev = None
            return _invalid(f"R_int not derivable: {', '.join(missing)} invalid"), evidence

        alt_quiet = ia.valid and ia.value is not None and abs(ia.value) < c.r_est_max_alt_i_a
        observable = regime == "BATTERY" or alt_quiet
        cur = {"v": v.value, "ib": ib.value, "rpm": rpm.value, "ts": ts, "observable": observable,
               "q": min(v.quality, ib.quality)}
        prev, self._prev = self._prev, cur

        if prev is not None and prev["observable"] and observable:
            d_i = cur["ib"] - prev["ib"]
            dt = (cur["ts"] - prev["ts"]).total_seconds() if cur["ts"] and prev["ts"] else None
            if (abs(d_i) >= c.r_est_min_di_a and abs(cur["rpm"] - prev["rpm"]) <= c.r_est_max_rpm_step
                    and dt is not None and 0.0 < dt <= c.r_est_max_dt_s):
                r = -(cur["v"] - prev["v"]) / d_i
                if r > 0.0:
                    self._r_steps.append(r)
                    evidence.append(f"load step dI_batt {d_i:+.2f} A -> {r * 1000.0:.1f} mOhm")

        if not observable:
            return _invalid(NOT_OBSERVABLE_REGULATED), evidence
        n = len(self._r_steps)
        if n < c.r_est_min_steps:
            return _invalid(f"insufficient load steps ({n} < {c.r_est_min_steps}) with the battery "
                            f"supplying the load"), evidence
        r_med = float(np.median(self._r_steps))
        return make_tagged(r_med * 1000.0, Provenance.DERIVED, valid=True, quality=cur["q"]), evidence

    # ------------------------------------------------------------------ ripple
    def _ripple(self, record: NormalizedSignalRecord
                ) -> tuple[ProvenanceTaggedValue, ProvenanceTaggedValue, ProvenanceTaggedValue, list[str]]:
        c = self._cfg
        burst = record.bus_voltage_burst
        evidence: list[str] = []
        if not burst.valid or len(burst) < 16:
            reason = f"voltage burst unavailable ({burst.fault_flag or 'not instrumented'})"
            return _invalid(reason), _invalid(reason), _invalid(reason), evidence
        fs = burst.sample_rate_hz
        if not band_is_representable(c.ripple_band_high_hz, fs):  # M-05
            reason = (f"ripple band {c.ripple_band_high_hz:.0f} Hz exceeds burst Nyquist {fs / 2.0:.0f} Hz "
                      f"(fs {fs:.0f} Hz); aliased, not reported")
            return _invalid(reason), _invalid(reason), _invalid(reason), evidence

        x = np.asarray(burst.samples, dtype=float)
        mean = float(x.mean())
        if mean <= 0.0:
            reason = "voltage burst mean not positive"
            return _invalid(reason), _invalid(reason), _invalid(reason), evidence
        spec = np.fft.rfft(x - mean)
        freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
        in_band = (freqs >= c.ripple_band_low_hz) & (freqs <= c.ripple_band_high_hz)
        # Parseval: RMS of the band-limited AC component
        power = np.abs(spec) ** 2
        power[1:-1 if len(x) % 2 == 0 else None] *= 2.0
        ac_rms = math.sqrt(float(power[in_band].sum()) / len(x) ** 2)
        ripple_pct = 100.0 * ac_rms / mean
        ripple_tv = make_tagged(ripple_pct, Provenance.DERIVED, valid=True, quality=burst.quality)
        evidence.append(f"ripple {ripple_pct:.3f} % RMS in {c.ripple_band_low_hz:.0f}-{c.ripple_band_high_hz:.0f} Hz")

        k = int(np.argmax(np.where(in_band, power, 0.0)))
        f_peak = float(freqs[k])
        floor = float(np.median(power[in_band]))
        if floor > 0.0 and power[k] < c.ripple_peak_snr * floor:
            reason = (f"no ripple line above the noise floor (peak/median power {power[k] / floor:.1f} "
                      f"< {c.ripple_peak_snr:.0f}); alternator not producing ripple")
            return ripple_tv, _invalid(reason), _invalid(reason), evidence
        rpm = record.rpm
        if not (rpm.valid and rpm.value and rpm.value > 0.0):
            reason = "rpm invalid; ripple order not derivable"
            return ripple_tv, _invalid(reason), _invalid(reason), evidence
        rev_s = self._generator_rev_s(rpm.value)
        order = f_peak / rev_s
        order_ok = abs(order - round(order)) <= c.ripple_order_tolerance
        if order_ok:
            order_tv = make_tagged(order, Provenance.DERIVED, valid=True, quality=min(burst.quality, rpm.quality))
            evidence.append(f"ripple peak {f_peak:.0f} Hz = order {order:.2f} of generator speed (shaft-locked)")
        else:
            order_tv = _invalid(f"ripple peak {f_peak:.0f} Hz is not an integer order of generator speed "
                                f"(order {order:.2f}); not shaft-locked")

        poles = self._eng.generator_poles
        if poles is None:
            freq_tv = _invalid("generator pole count unknown (engine.generator_poles = None, VERIFY against "
                               "Rotax 915 iS Operators Manual); frequency not attributable")
        else:
            f_e = poles / 2.0 * rev_s
            f_expected = 2.0 * self._eng.generator_phases * f_e
            if not band_is_representable(f_expected, fs):
                freq_tv = _invalid(f"expected ripple {f_expected:.0f} Hz exceeds burst Nyquist {fs / 2.0:.0f} Hz; "
                                   f"peak would be aliased")
            elif min(abs(f_peak - f_expected), abs(f_peak - f_e)) <= max(2.0 * fs / len(x), 0.02 * f_expected):
                freq_tv = make_tagged(f_peak, Provenance.DERIVED, valid=True, quality=order_tv.quality)
            else:
                freq_tv = _invalid(f"ripple peak {f_peak:.0f} Hz matches neither {f_expected:.0f} Hz "
                                   f"(rectifier) nor {f_e:.0f} Hz (electrical)")
        return ripple_tv, freq_tv, order_tv, evidence

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, record: NormalizedSignalRecord) -> ElectricalState:
        c = self._cfg
        ts = record.timestamp
        evidence: list[str] = []

        def tag(ch) -> ProvenanceTaggedValue[float | None]:
            if ch.valid and ch.value is not None:
                return make_tagged(ch.value, Provenance.DERIVED, valid=True, quality=ch.quality)
            return _invalid(ch.fault_flag or "channel invalid")

        v_tv, ia_tv, ib_tv = tag(record.voltage), tag(record.current), tag(record.battery_current)
        regime = self._regime(record)

        # --- charging residual
        if regime == "UNKNOWN":
            exp_tv = _invalid("rpm invalid; charging regime unknown")
        else:
            exp_v = c.regulator_setpoint_v if regime == "REGULATED" else c.battery_ocv_nominal_v
            exp_tv = make_tagged(exp_v, Provenance.DERIVED, valid=True, quality=record.rpm.quality)
        if v_tv.valid and exp_tv.valid:
            res = v_tv.value - exp_tv.value
            res_tv = make_tagged(res, Provenance.DERIVED, valid=True, quality=min(v_tv.quality, exp_tv.quality))
            self._residuals.append(res)
            res_med = float(np.median(self._residuals))
            res_med_tv = make_tagged(res_med, Provenance.DERIVED, valid=True, quality=res_tv.quality)
            charging_status = _band(abs(res_med), c.charging_warning_v, c.charging_alarm_v)
        else:
            reason = v_tv.fault_flag if not v_tv.valid else exp_tv.fault_flag
            res_tv = res_med_tv = _invalid(f"charging residual not derivable: {reason}")
            self._residuals.clear()
            charging_status = DiagnosticStatus.INVALID

        # --- battery internal resistance
        r_tv, r_ev = self._r_int(record, regime, ts)
        evidence += r_ev
        if r_tv.valid:
            r_ratio = r_tv.value / c.battery_r_int_nominal_mohm
            battery_status = _band(r_ratio, c.r_int_warning_ratio, c.r_int_alarm_ratio)
        else:
            battery_status = DiagnosticStatus.INVALID

        # --- ripple
        rip_tv, freq_tv, order_tv, rip_ev = self._ripple(record)
        evidence += rip_ev
        if rip_tv.valid:
            rip_ratio = rip_tv.value / c.ripple_healthy_pct
            ripple_status = _band(rip_ratio, c.ripple_warning_ratio, c.ripple_alarm_ratio)
        else:
            ripple_status = DiagnosticStatus.INVALID

        # --- Electrical Health Index (LHI pattern)
        comps: dict[str, tuple[float, float | None]] = {
            "charging": (c.ehi_weight_charging,
                         abs(res_med_tv.value) / c.charging_alarm_v if res_med_tv.valid else None),
            "ripple": (c.ehi_weight_ripple,
                       max(0.0, rip_tv.value / c.ripple_healthy_pct - 1.0) / (c.ripple_alarm_ratio - 1.0)
                       if rip_tv.valid else None),
            "r_int": (c.ehi_weight_r_int,
                      max(0.0, r_tv.value / c.battery_r_int_nominal_mohm - 1.0) / (c.r_int_alarm_ratio - 1.0)
                      if r_tv.valid else None),
        }
        w_all = sum(w for w, _ in comps.values())
        w_valid = sum(w for w, d in comps.values() if d is not None)
        coverage = w_valid / w_all if w_all > 0.0 else 0.0
        if w_valid > 0.0:
            log_sum = sum(w * d * math.log(0.5) for w, d in comps.values() if d is not None)
            ehi = math.exp(log_sum / w_valid)
            qualities = [tv.quality for tv, (_, d) in zip((res_med_tv, rip_tv, r_tv), comps.values()) if d is not None]
            ehi_tv = make_tagged(ehi, Provenance.DERIVED, valid=True, quality=min(qualities))
            ehi_band = "NORMAL" if ehi >= c.ehi_warning else "WARNING" if ehi >= c.ehi_alarm else "ALARM"
            for name, (w, d) in comps.items():
                if d is None:
                    evidence.append(f"{name} excluded (invalid), weight 0")
                else:
                    evidence.append(f"{name} factor {0.5 ** d:.3f} (weight {w:.2f})")
            evidence.append(f"coverage {coverage:.2f} (weights re-normalised over valid components)")
        else:
            ehi_tv = _invalid("no electrical component derivable (electrical signals not instrumented or invalid)")
            ehi_band = "UNKNOWN"

        statuses = [charging_status, ripple_status, battery_status]
        status = max(statuses, key=lambda s: _RANK[s])

        return ElectricalState(
            timestamp=ts,
            bus_voltage_v=v_tv,
            alternator_current_a=ia_tv,
            battery_current_a=ib_tv,
            expected_bus_voltage_v=exp_tv,
            charging_regime=regime,
            charging_residual_v=res_tv,
            charging_residual_median_v=res_med_tv,
            battery_resistance_mohm=r_tv,
            r_int_step_count=len(self._r_steps),
            voltage_ripple_pct=rip_tv,
            ripple_dominant_frequency_hz=freq_tv,
            ripple_order=order_tv,
            charging_status=charging_status,
            ripple_status=ripple_status,
            battery_status=battery_status,
            status=status,
            ehi=ehi_tv,
            ehi_band=ehi_band,
            ehi_coverage=coverage,
            ehi_evidence=evidence,
        )
