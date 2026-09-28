"""
Injection, ignition and fuel-system diagnosis (L2; Prompt 13).

Inputs: NormalizedSignalRecord (injector pulse widths, SOI and ignition angles,
fuel rail gauge pressure, EGTs, CHT, MAP, ambient, measured fuel flow), the
DerivedEngineState (operating λ) and the CombustionStabilityState (misfire
gate, M-06).

    rail_dp            = p_fuel,gauge + p_ambient - MAP   (manifold-referenced regulator; VERIFY)
    rail residual      = rail_dp - expected rail_dp(current injector demand)
    fuel_cmd_i         = K_inj sqrt(rail_dp / dP_ref) (PW_i - t_dead)          [kg per cycle]
    injector duty_i    = PW_i / (120 / rpm)
    fuel_delivery_ratio= measured fuel flow / Σ fuel_cmd_i · rpm/120          (SRD-FUN-084)
    injector flow ratio: each cylinder's EGT residual (M-11 per-cylinder
        expectation) relative to the median residual of the OTHER cylinders,
        read through an exhaust energy balance x(λ) η_c(λ) f/(1+f) to the λ
        that cylinder actually burns; flow ratio = λ_operating / λ_cylinder.
        Hot = lean (less fuel than commanded); cold = rich, unless the misfire
        gate CONFIRMS a misfire on that cylinder (non-firing, SRD-FUN-044).
        The common-mode part of the EGT residual (all cylinders; OI-5 bias,
        engine-wide mixture) is deliberately not used: rail pressure and fuel
        delivery cover engine-wide fuel problems.
    ignition residual  = measured advance - ECU scheduled advance; a consistent
        retard on a cylinder is ECU knock control, i.e. evidence for
        DETONATION_KNOCK, supported by that cylinder running hot and CHT rising.

Knock energy (several kHz) is above the 1024 Hz Nyquist limit of the 2048 Hz
accelerometer burst, so no knock evidence is taken from the vibration spectrum.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median

from src.core.config import AppSettings, get_settings
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import (
    CombustionStabilityState,
    DerivedEngineState,
    InjectionState,
    ProvenanceTaggedValue,
    make_tagged,
)
from src.core.units import interp2d_clamped
from src.l1_data.signal_record import NormalizedSignalRecord
from src.l2_digital_twin.residual_engine import HealthyExpectationModel, operating_point_from_record

N_CYL = 4


def interval_us_to_deg(delay_us: float, rpm: float) -> float:
    """Timer interval to crank angle at the instantaneous speed:
    deg = us x (rpm / 60 rev/s) x 360 deg/rev x 1e-6 s/us = us x rpm x 6e-6."""
    return delay_us * rpm * 6e-6


def _invalid(reason: str) -> ProvenanceTaggedValue[float | None]:
    return ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False,
                                 validity_reason=ChannelValidity.MISSING, quality=0.0, fault_flag=reason)


def _valid(value: float, quality: float = 1.0) -> ProvenanceTaggedValue[float | None]:
    return make_tagged(value, Provenance.DERIVED, valid=True, quality=quality)


def _band(deviation: float | None, warning: float, alarm: float) -> DiagnosticStatus:
    if deviation is None:
        return DiagnosticStatus.INVALID
    if abs(deviation) >= alarm:
        return DiagnosticStatus.CRITICAL
    if abs(deviation) >= warning:
        return DiagnosticStatus.WARNING
    return DiagnosticStatus.NORMAL


_RANK = {DiagnosticStatus.INVALID: 0, DiagnosticStatus.NORMAL: 1, DiagnosticStatus.WARNING: 2,
         DiagnosticStatus.CRITICAL: 3}


def _worst(*statuses: DiagnosticStatus) -> DiagnosticStatus:
    valid = [s for s in statuses if s != DiagnosticStatus.INVALID]
    return max(valid, key=_RANK.get) if valid else DiagnosticStatus.INVALID


class InjectionModel:
    """Stateful (running medians); call evaluate() once per record in time order."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.cfg = self._settings.injection
        self.expectations = HealthyExpectationModel(self._settings)
        self._lambda_peak = self._find_lambda_peak()
        self.reset_state()

    def reset_state(self) -> None:
        n = self.cfg.median_n
        self._last_bound: float | None = None
        self._rail_hist: deque[float] = deque(maxlen=n)
        self._delivery_hist: deque[float] = deque(maxlen=n)
        self._flow_hist = [deque(maxlen=n) for _ in range(N_CYL)]
        # upper bound on the flow ratio when a cylinder is hotter than the
        # model's peak-EGT rise (lean of peak): None when not bounded
        self._bound_hist: list[deque[float | None]] = [deque(maxlen=n) for _ in range(N_CYL)]
        self._ign_hist = [deque(maxlen=n) for _ in range(N_CYL)]
        self._cht_res_hist: deque[float] = deque(maxlen=n)

    # ------------------------------------------------------------------ energy balance (L2's own)
    def exhaust_energy(self, lam: float) -> float:
        c = self.cfg
        if lam <= 1.0:
            x = c.egt_x_stoich - c.egt_x_rich_slope * (1.0 - lam)
        else:
            x = c.egt_x_stoich + c.egt_x_lean_gain * (1.0 - math.exp(-(lam - 1.0) / c.egt_x_lean_scale))
        f = 1.0 / (c.stoichiometric_afr * lam)
        return x * c.combustion_efficiency_max * min(1.0, lam) * f / (1.0 + f)

    def _find_lambda_peak(self) -> float:
        grid = [0.9 + 0.001 * k for k in range(601)]
        return max(grid, key=self.exhaust_energy)

    @property
    def lambda_peak(self) -> float:
        return self._lambda_peak

    def _solve_lambda(self, target: float, lo: float, hi: float) -> float:
        """λ in [lo, hi] with exhaust_energy(λ) = target; E is monotonic there."""
        increasing = self.exhaust_energy(hi) >= self.exhaust_energy(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (self.exhaust_energy(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def scheduled_advance_deg(self, rpm: float, map_pa: float) -> float:
        c = self.cfg
        return interp2d_clamped(c.ign_rpm_axis, c.ign_map_axis_pa, c.ign_table_deg, rpm, map_pa)

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, record: NormalizedSignalRecord, derived_state: DerivedEngineState | None = None,
                 comb_state: CombustionStabilityState | None = None) -> InjectionState:
        c = self.cfg
        evidence: list[str] = []
        rpm_ok = record.rpm.valid and record.rpm.value is not None and record.rpm.value > 0.0
        rpm = record.rpm.value if rpm_ok else None
        op = operating_point_from_record(record)
        op_ok = all(ch.valid for ch in (record.rpm, record.map_pressure, record.ambient_pressure, record.ambient_temp))

        # ---------------------------------------------------------- fuel rail
        fp, amb, mp = record.fuel_pressure, record.ambient_pressure, record.map_pressure
        if fp.valid and amb.valid and mp.valid:
            rail_dp = fp.value + amb.value - mp.value
            rail_gauge_tv = _valid(fp.value / 1000.0, fp.quality)
            rail_dp_tv = _valid(rail_dp / 1000.0, min(fp.quality, mp.quality))
            mix = math.sqrt(max(rail_dp, 0.0) / c.rail_dp_ref_pa)
            mix_tv = _valid(mix)
        else:
            rail_dp = None
            bad = [n for n, ch in (("fuel_pressure", fp), ("ambient_pressure", amb), ("map_pressure", mp))
                   if not ch.valid]
            reason = f"rail pressure not derivable: {', '.join(bad)} invalid"
            rail_gauge_tv = _valid(fp.value / 1000.0, fp.quality) if fp.valid else _invalid(
                fp.fault_flag or "fuel_pressure invalid")
            rail_dp_tv = mix_tv = _invalid(reason)
            mix = None

        # ---------------------------------------------------------- commanded fuel, duty
        pws = [record.injector_pulse_width_cyl_1, record.injector_pulse_width_cyl_2,
               record.injector_pulse_width_cyl_3, record.injector_pulse_width_cyl_4]
        dead_s = c.injector_dead_time_us * 1e-6
        fuel_cmd: list[ProvenanceTaggedValue] = []
        duty: list[ProvenanceTaggedValue] = []
        saturated = [False] * N_CYL
        fuel_cmd_kg: list[float | None] = []
        for i, pw in enumerate(pws):
            if pw.valid and rpm is not None:
                d = pw.value / (120.0 / rpm) * 100.0
                duty.append(_valid(d, pw.quality))
                saturated[i] = d > c.duty_saturation_pct
            else:
                duty.append(_invalid(pw.fault_flag or "pulse width invalid") if not pw.valid
                            else _invalid("rpm invalid; cycle time unknown"))
            if pw.valid and mix is not None:
                m = c.injector_static_flow_kg_s * mix * max(pw.value - dead_s, 0.0)
                fuel_cmd_kg.append(m)
                fuel_cmd.append(_valid(m * 1e6, pw.quality))
            else:
                fuel_cmd_kg.append(None)
                fuel_cmd.append(_invalid(pw.fault_flag if not pw.valid else "rail pressure invalid"))
        if any(saturated):
            evidence.append(f"injector duty above {c.duty_saturation_pct:.0f} % on cylinder(s) "
                            f"{[i + 1 for i, s_ in enumerate(saturated) if s_]}")

        if rpm is not None and all(m is not None for m in fuel_cmd_kg):
            total = sum(fuel_cmd_kg) * rpm / 120.0  # type: ignore[arg-type]
            total_tv = _valid(total)
        else:
            total = None
            total_tv = _invalid("commanded fuel not derivable for every cylinder")

        # rail expectation at the current demand, residual
        if rail_dp is not None and op_ok:
            exp_dp = self.expectations.expected_fuel_rail_dp_pa(op, total or 0.0)
            res = rail_dp - exp_dp
            self._rail_hist.append(res)
            exp_tv, res_tv = _valid(exp_dp / 1000.0), _valid(res / 1000.0)
            res_med = median(self._rail_hist)
            res_med_tv = _valid(res_med / 1000.0)
        else:
            exp_tv = res_tv = res_med_tv = _invalid("rail pressure or operating point invalid")
            res_med = None

        # fuel delivery consistency (SRD-FUN-084)
        ff = record.fuel_flow
        if total is not None and total > 0.0 and ff.valid:
            ratio = ff.value / total
            self._delivery_hist.append(ratio)
            del_tv, del_med = _valid(ratio, ff.quality), median(self._delivery_hist)
            del_med_tv = _valid(del_med)
        else:
            del_tv = del_med_tv = _invalid("measured fuel flow or commanded fuel invalid")
            del_med = None

        rail_status = _band(res_med, c.rail_residual_warning_pa, c.rail_residual_alarm_pa)
        delivery_status = _band(None if del_med is None else del_med - 1.0, c.fuel_delivery_warning,
                                c.fuel_delivery_alarm)
        fuel_system_status = _worst(rail_status, delivery_status)
        if rail_status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
            evidence.append(f"rail pressure residual {res_med / 1000.0:+.1f} kPa (median of {len(self._rail_hist)})")
            if mix is not None and mix < 1.0:
                evidence.append(f"all cylinders lean together: fuel per injection {100.0 * (mix - 1.0):+.1f} % "
                                f"vs the regulated rail (rail mixture factor {mix:.3f})")
        if delivery_status in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
            evidence.append(f"fuel delivery ratio {del_med:.3f}: measured flow differs from the injector command")

        # ---------------------------------------------------------- EGT differential residuals
        egts = [record.egt_cyl_1, record.egt_cyl_2, record.egt_cyl_3, record.egt_cyl_4]
        exp_cyl = self.expectations.expected_egt_per_cylinder_k(op) if op_ok else None
        resid = [(e.value - x) if (e.valid and exp_cyl is not None) else None
                 for e, x in zip(egts, exp_cyl or (None,) * N_CYL)]
        diffs: list[float | None] = []
        diff_reason: list[str] = []
        for i in range(N_CYL):
            others = [r for j, r in enumerate(resid) if j != i and r is not None]
            if resid[i] is None or len(others) < 2:
                diffs.append(None)
                diff_reason.append("EGT or its expectation invalid" if resid[i] is None
                                   else "fewer than 2 other valid cylinders")
            else:
                diffs.append(resid[i] - median(others))  # reference excludes this cylinder
                diff_reason.append("")

        # ---------------------------------------------------------- ignition timing, knock
        igns = [record.ignition_timing_cyl_1, record.ignition_timing_cyl_2,
                record.ignition_timing_cyl_3, record.ignition_timing_cyl_4]
        if rpm is not None and record.map_pressure.valid:
            sched = self.scheduled_advance_deg(rpm, record.map_pressure.value)
            sched_tv = _valid(sched)
        else:
            sched = None
            sched_tv = _invalid("rpm or MAP invalid; schedule lookup not possible")
        cht_exp = self.expectations.expected_cht_k(op) if op_ok else None
        cht_rise = None
        if cht_exp is not None and record.cht_cyl_1.valid:
            self._cht_res_hist.append(record.cht_cyl_1.value - cht_exp)
            cht_rise = self._cht_res_hist[-1] - min(self._cht_res_hist)
        ign_res_tv: list[ProvenanceTaggedValue] = []
        ign_med_tv: list[ProvenanceTaggedValue] = []
        knock = [False] * N_CYL
        knock_support = [False] * N_CYL
        for i, ch in enumerate(igns):
            if ch.valid and sched is not None:
                r = ch.value - sched
                self._ign_hist[i].append(r)
                ign_res_tv.append(_valid(r, ch.quality))
                med = median(self._ign_hist[i])
                ign_med_tv.append(_valid(med))
                if med <= -c.knock_retard_threshold_deg:
                    knock[i] = True
                    egt_hot = diffs[i] is not None and diffs[i] >= c.knock_egt_support_k
                    cht_up = cht_rise is not None and cht_rise >= c.knock_cht_support_k
                    knock_support[i] = egt_hot or cht_up
                    evidence.append(
                        f"cylinder {i + 1} ignition retarded {med:+.1f} deg vs schedule (ECU knock control)"
                        + (f"; EGT {diffs[i]:+.0f} K vs other cylinders" if egt_hot else "")
                        + (f"; CHT residual up {cht_rise:.0f} K" if cht_up else ""))
            else:
                reason = ch.fault_flag if not ch.valid else "schedule lookup not possible"
                ign_res_tv.append(_invalid(reason or "ignition timing invalid"))
                ign_med_tv.append(_invalid(reason or "ignition timing invalid"))
        if not any(tv.valid for tv in ign_res_tv):
            knock_status = DiagnosticStatus.INVALID
        elif any(k and s_ for k, s_ in zip(knock, knock_support)):
            knock_status = DiagnosticStatus.CRITICAL
        elif any(knock):
            knock_status = DiagnosticStatus.WARNING
        else:
            knock_status = DiagnosticStatus.NORMAL

        # ---------------------------------------------------------- injector flow from EGT
        lam_tv = derived_state.lambda_derived if derived_state is not None else None
        lam_op = lam_tv.value if (lam_tv is not None and lam_tv.valid and lam_tv.value is not None) else None
        lambda_operating_tv = _valid(lam_op) if lam_op is not None else _invalid("operating λ not derivable")
        t_charge = record.ambient_temp.value + c.charge_temp_rise_k if record.ambient_temp.valid else None
        misfire = comb_state.misfire_detected if comb_state is not None else [False] * N_CYL
        diff_tv: list[ProvenanceTaggedValue] = []
        flow_tv: list[ProvenanceTaggedValue] = []
        flow_med_tv: list[ProvenanceTaggedValue] = []
        injector_status: list[DiagnosticStatus] = []
        # commanded share of each cylinder relative to the others (equal PW -> 1)
        pw_eff = [max(pw.value - dead_s, 0.0) if pw.valid else None for pw in pws]
        for i in range(N_CYL):
            others_pw = [p_ for j, p_ in enumerate(pw_eff) if j != i and p_ is not None]
            cmd_share = (pw_eff[i] / median(others_pw)
                         if pw_eff[i] is not None and others_pw and median(others_pw) > 0.0 else None)
            if diffs[i] is None:
                diff_tv.append(_invalid(diff_reason[i]))
                flow_tv.append(_invalid(diff_reason[i]))
            elif cmd_share is None:
                diff_tv.append(_valid(diffs[i]))
                flow_tv.append(_invalid("commanded fuel unknown: injector pulse width invalid or not "
                                        "instrumented" if not pws[i].valid else
                                        "commanded fuel unknown: other cylinders' pulse widths invalid"))
            elif knock[i]:
                diff_tv.append(_valid(diffs[i]))
                flow_tv.append(_invalid("ignition retarded on this cylinder: its EGT reflects timing, "
                                        "not fuel"))
            else:
                diff_tv.append(_valid(diffs[i]))
                flow_tv.append(self._flow_ratio(i, egts[i].value, diffs[i], lam_op, t_charge, misfire[i],
                                                cmd_share))
            bound = self._last_bound if (diffs[i] is not None and not flow_tv[-1].valid) else None
            self._bound_hist[i].append(bound)
            self._last_bound = None
            if flow_tv[-1].valid:
                self._flow_hist[i].append(flow_tv[-1].value)
            # the median only speaks for the current record if this record is valid
            if flow_tv[-1].valid and self._flow_hist[i]:
                med = median(self._flow_hist[i])
                flow_med_tv.append(_valid(med))
                injector_status.append(_band(med - 1.0, c.injector_flow_warning, c.injector_flow_alarm))
            else:
                flow_med_tv.append(_invalid(flow_tv[-1].fault_flag or "flow ratio invalid"))
                injector_status.append(self._bound_status(i))
        identified = [i + 1 for i, s_ in enumerate(injector_status)
                      if s_ in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)]
        for i in identified:
            if flow_med_tv[i - 1].valid:
                med = flow_med_tv[i - 1].value
                evidence.append(f"cylinder {i} injector flow ratio {med:.3f} "
                                f"({'less' if med < 1.0 else 'more'} fuel than commanded; "
                                f"{'clogged' if med < 1.0 else 'leaking/dripping'} injector suspected)")
            else:
                b = median([x for x in self._bound_hist[i - 1] if x is not None])
                evidence.append(f"cylinder {i} injector flow ratio at most {b:.3f} (bound: the cylinder runs "
                                "hotter than the model's peak-EGT rise, lean of peak; clogged injector suspected)")

        duty_status = DiagnosticStatus.WARNING if any(saturated) else (
            DiagnosticStatus.NORMAL if any(tv.valid for tv in duty) else DiagnosticStatus.INVALID)
        status = _worst(fuel_system_status, knock_status, duty_status, *injector_status)

        return InjectionState(
            timestamp=record.timestamp,
            fuel_rail_pressure_kpa=rail_gauge_tv,
            rail_dp_kpa=rail_dp_tv,
            expected_rail_dp_kpa=exp_tv,
            rail_pressure_residual_kpa=res_tv,
            rail_pressure_residual_median_kpa=res_med_tv,
            rail_mixture_factor=mix_tv,
            fuel_cmd_per_cycle_mg=fuel_cmd,
            injector_duty_pct=duty,
            injector_duty_saturated=saturated,
            fuel_cmd_total_kg_s=total_tv,
            fuel_delivery_ratio=del_tv,
            fuel_delivery_ratio_median=del_med_tv,
            lambda_operating=lambda_operating_tv,
            egt_differential_k=diff_tv,
            injector_flow_ratio=flow_tv,
            injector_flow_ratio_median=flow_med_tv,
            injector_status=injector_status,
            identified_injectors=identified,
            scheduled_advance_deg=sched_tv,
            ign_timing_residual_deg=ign_res_tv,
            ign_timing_residual_median_deg=ign_med_tv,
            knock_suspected=knock,
            fuel_system_status=fuel_system_status,
            knock_status=knock_status,
            status=status,
            evidence=evidence,
        )

    def _bound_status(self, i: int) -> DiagnosticStatus:
        """Status from the flow-ratio upper bound when most of the window is
        lean of peak; INVALID otherwise. The bound never becomes a value."""
        hist = self._bound_hist[i]
        bounded = [b for b in hist if b is not None]
        if len(hist) < min(3, hist.maxlen or 3) or len(bounded) * 2 <= len(hist):
            return DiagnosticStatus.INVALID
        b = median(bounded)
        if b > 1.0 - self.cfg.injector_flow_warning:
            return DiagnosticStatus.INVALID  # bound too weak to say anything
        return _band(b - 1.0, self.cfg.injector_flow_warning, self.cfg.injector_flow_alarm)

    def _flow_ratio(self, i: int, egt_k: float, diff_k: float, lam_op: float | None,
                    t_charge: float | None, misfire_confirmed: bool,
                    cmd_share: float = 1.0) -> ProvenanceTaggedValue:
        """Fuel actually burnt / fuel commanded for cylinder i, from its EGT.
        cmd_share = this cylinder's commanded fuel / the other cylinders' (the
        EGT reference is the other cylinders, which burn their own command)."""
        c = self.cfg
        if misfire_confirmed:
            return _invalid("misfire gate CONFIRMED on this cylinder: non-firing, EGT does not reflect "
                            "its fuel (SRD-FUN-044)")
        if lam_op is None or t_charge is None:
            return _invalid("operating λ or charge temperature not derivable")
        if lam_op > self._lambda_peak - c.lambda_peak_margin:
            return _invalid(f"operating λ {lam_op:.3f} is not rich of peak EGT (model peak λ "
                            f"{self._lambda_peak:.3f}, margin {c.lambda_peak_margin}): EGT is not monotonic "
                            "in λ here, so a hot or cold cylinder cannot be read as lean or rich")
        rise = egt_k - t_charge
        rise_ref = rise - diff_k
        if rise <= 0.0 or rise_ref <= 0.0:
            return _invalid("EGT at or below the charge temperature")
        target = self.exhaust_energy(lam_op) * rise / rise_ref
        if diff_k >= 0.0:  # hot: lean branch up to the peak
            if target > self.exhaust_energy(self._lambda_peak):
                bound = lam_op / self._lambda_peak / cmd_share
                self._last_bound = bound
                return _invalid(f"EGT beyond the model's peak-EGT rise: flow ratio below {bound:.3f} "
                                "(bound only; lean of peak EGT)")
            lam_i = self._solve_lambda(target, lam_op, self._lambda_peak)
        else:  # cold: rich branch (firing, per the misfire gate)
            if target < self.exhaust_energy(c.lambda_search_min):
                return _invalid(f"EGT below the rich limit of the model (λ {c.lambda_search_min}); "
                                "non-firing suspected although the misfire gate did not confirm")
            lam_i = self._solve_lambda(target, c.lambda_search_min, lam_op)
        return _valid(lam_op / lam_i / cmd_share)
