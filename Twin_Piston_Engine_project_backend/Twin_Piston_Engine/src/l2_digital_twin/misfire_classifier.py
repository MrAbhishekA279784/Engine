"""
Misfire and Combustion Stability Diagnostics — Original Module 10.

Sixth processing stage of L2 Digital Twin.
Performs deterministic multi-signal evidence fusion (EGT drop/deviation, crank speed stability,
and vibration spectral features) to characterize combustion stability and detect possible misfire
per cylinder.

STRICT BOUNDARY CONSTRAINTS:
    - Input: NormalizedSignalRecord (Module 5), DerivedEngineState (Module 6),
             DiagnosticState/EGTDiagnosticResult (Module 7), VibrationState/Result (Module 9)
    - Output: CombustionStabilityState & MisfireDiagnosticResult
    - Zero ML fault classifiers, anomaly detectors, RUL, or advisory recommendation logic
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, MisfireConfig, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity, DiagnosticStatus, Provenance
from src.core.schemas import (
    CombustionStabilityState,
    DerivedEngineState,
    DiagnosticState,
    ProvenanceTaggedValue,
    VibrationState,
    make_tagged,
)
from src.l1_data.signal_record import ChannelValue, NormalizedSignalRecord
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticResult
from src.l2_digital_twin.physics.M09_combustion_stability_index import csi_band as m09_csi_band
from src.l2_digital_twin.physics.M06_dual_channel_misfire_gate import (
    CrankSpeedVariability,
    DualChannelMisfireGate,
    MisfireAssessment,
    MisfireVerdict,
)
from src.l2_digital_twin.vibration_processor import VibrationSignalProcessingResult

logger = get_logger(__name__)


def _severity_rank(status: DiagnosticStatus) -> int:
    """Helper for severity comparison."""
    ranks = {
        DiagnosticStatus.INVALID: 0,
        DiagnosticStatus.NORMAL: 1,
        DiagnosticStatus.WARNING: 2,
        DiagnosticStatus.CRITICAL: 3,
    }
    return ranks.get(status, 0)


def _max_status(s1: DiagnosticStatus, s2: DiagnosticStatus) -> DiagnosticStatus:
    """Return the status with higher severity."""
    if s1 == DiagnosticStatus.INVALID and s2 != DiagnosticStatus.INVALID:
        return s2
    if s2 == DiagnosticStatus.INVALID and s1 != DiagnosticStatus.INVALID:
        return s1
    return s1 if _severity_rank(s1) >= _severity_rank(s2) else s2


def _finite_or_none(v: float) -> float | None:
    return v if (v is not None and math.isfinite(v)) else None


def _tagged(v: float) -> ProvenanceTaggedValue[float | None]:
    fv = _finite_or_none(v)
    return make_tagged(fv, Provenance.DERIVED, valid=fv is not None, quality=1.0 if fv is not None else 0.0)


class CylinderCombustionEvidence(BaseModel):
    """Multi-signal combustion evidence breakdown for a single cylinder (1..4)."""

    cylinder_id: int = Field(ge=1, le=4)
    status: DiagnosticStatus
    possible_misfire: bool
    evidence_score: float = Field(ge=0.0, le=1.0)
    egt_evidence: float = Field(ge=0.0, le=1.0)
    crank_evidence: float = Field(ge=0.0, le=1.0)
    vibration_evidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class MisfireDiagnosticResult(BaseModel):
    """Complete engine combustion stability & misfire diagnostic result container."""

    timestamp: datetime
    provenance: Provenance = Field(default=Provenance.DERIVED)
    cylinder_evidence: list[CylinderCombustionEvidence]
    misfire_detected: list[bool]
    overall_status: DiagnosticStatus
    affected_cylinders: list[int]
    # Dual-channel AND gate (M-06)
    gate_verdict: str = "NOT_EVALUATED"
    gate_triggering_channel: str | None = None
    gate_evidence: str = ""
    crank_cov_pct: float | None = None
    half_order_fraction: float | None = None
    crank_cov_limit_pct: float | None = None
    half_order_limit: float | None = None
    # Combustion Stability Index (M-09)
    csi_value: float | None = None
    csi_band: str = "UNKNOWN"
    csi_dominant_term: str = "none"
    csi_terms: dict[str, float] = Field(default_factory=dict)
    csi_reason: str = ""

    model_config = ConfigDict(frozen=True)


class MisfireDetector:
    """Deterministic multi-signal combustion stability and misfire analyzer."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cfg: MisfireConfig = self._settings.misfire
        self._prev_timestamp: datetime | None = None
        self._prev_egts: dict[int, float] = {}
        self._prev_rpm: float | None = None
        # (t_s, cylinder-mean EGT K, CHT K, brake power kW, MAP Pa) over the CSI window
        self._csi_hist: deque[tuple[float, float | None, float | None, float | None, float | None]] = deque()
        # Channel A limit = max(floor, multiplier x healthy CoV) via M-06's
        # baseline branch; the 2.0 % M-06 constant is documentation only.
        self._gate = DualChannelMisfireGate(
            crank_cov_threshold_pct=self._cfg.gate_crank_cov_floor_pct,
            half_order_threshold=self._cfg.gate_half_order_threshold,
            baseline_multiplier=self._cfg.gate_baseline_multiplier,
        )

    def _crank_cov_pct(self, record: NormalizedSignalRecord) -> float | None:
        """Channel A: coefficient of variation of the per-revolution crank
        periods in this record's burst (a dispersion, not a rate of change).
        None if the burst is missing or too short for a stable statistic."""
        burst = record.crank_period_burst
        if not burst.valid:
            return None
        cv = CrankSpeedVariability(window=self._cfg.gate_crank_window_revs)
        for period_us in burst.samples[-self._cfg.gate_crank_window_revs:]:
            cv.push(period_us)
        return cv.coefficient_of_variation_pct() if cv.ready() else None

    @staticmethod
    def _half_order(vib_diag: VibrationSignalProcessingResult | VibrationState | None) -> float | None:
        """Channel B: overall half-order (0.5X) energy fraction."""
        if isinstance(vib_diag, VibrationSignalProcessingResult):
            return vib_diag.half_order_fraction if vib_diag.valid else None
        if isinstance(vib_diag, VibrationState):
            tv = vib_diag.half_order_fraction
            return tv.value if (tv.valid and tv.value is not None) else None
        return None

    def _run_gate(self, record: NormalizedSignalRecord, vib_diag) -> MisfireAssessment:
        cov = self._crank_cov_pct(record)
        half = self._half_order(vib_diag)
        assessment = self._gate.evaluate(
            crank_cov_pct=cov if cov is not None else float("nan"),
            half_order_fraction=half if half is not None else float("nan"),
            baseline_cov_pct=self._cfg.gate_healthy_crank_cov_pct,
            inputs_valid=cov is not None and half is not None,
        )
        if assessment.verdict == MisfireVerdict.INVALID:
            missing = [name for name, v in (("crank-period burst (channel A)", cov),
                                            ("half-order fraction (channel B)", half)) if v is None]
            return MisfireAssessment(
                MisfireVerdict.INVALID, float("nan"), False, float("nan"), False, None,
                f"No misfire verdict: missing {', '.join(missing)}.",
            )
        return assessment

    def reset_state(self) -> None:
        """Reset temporal state tracking (useful for testing)."""
        self._prev_timestamp = None
        self._prev_egts.clear()
        self._prev_rpm = None
        self._csi_hist.clear()

    @staticmethod
    def _slope(ts: list[float], ys: list[float]) -> float | None:
        """Least-squares slope dy/dt; None if the window has no time span."""
        n = len(ts)
        if n < 2:
            return None
        t_mean = sum(ts) / n
        y_mean = sum(ys) / n
        den = sum((t - t_mean) ** 2 for t in ts)
        if den <= 0.0:
            return None
        return sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys)) / den

    def _combustion_stability(
        self,
        record: NormalizedSignalRecord,
        crank_cov_pct: float | None,
        derived_state: DerivedEngineState | None,
    ):
        """M-09 CSI = EGT std/mean + crank CoV/100 + |CHT slope| / ref (thermal term gated).

        The healthy CHT expectation is not calibrated (OI-5), so it does not
        drive CSI. Instead the thermal term is computed only in steady
        operation: when brake power and MAP rates over the window are below
        configured limits. During a transient it is marked invalid and
        excluded (coverage reduced), not set to zero. The CHT slope is a
        least-squares fit over csi_window_s (30 s).
        """
        cfg = self._cfg
        t_s = record.timestamp.timestamp() if record.timestamp is not None else None
        egts = [ch.value for ch in (record.egt_cyl_1, record.egt_cyl_2, record.egt_cyl_3, record.egt_cyl_4)
                if ch.valid and ch.value > 0.0]
        egt_mean = sum(egts) / len(egts) if egts else None
        cht = record.cht_cyl_1.value if record.cht_cyl_1.valid else None
        power = (derived_state.brake_power_kw.value
                 if derived_state is not None and derived_state.brake_power_kw.valid else None)
        map_pa = record.map_pressure.value if record.map_pressure.valid else None

        if t_s is not None:
            if self._csi_hist and t_s <= self._csi_hist[-1][0]:
                self._csi_hist.clear()  # time went backwards: new sequence
            self._csi_hist.append((t_s, egt_mean, cht, power, map_pa))
            while self._csi_hist and t_s - self._csi_hist[0][0] > cfg.csi_window_s:
                self._csi_hist.popleft()

        hist = list(self._csi_hist)
        egt_win = [(h[0], h[1]) for h in hist if h[1] is not None]
        if crank_cov_pct is None or not math.isfinite(crank_cov_pct):
            return None, "UNKNOWN", "none", {}, "crank_cov unavailable", 0.0
        if len(egt_win) < cfg.csi_min_samples:
            reason = f"CSI window has {len(egt_win)} EGT samples, needs {cfg.csi_min_samples}"
            return None, "UNKNOWN", "none", {}, reason, 0.0

        egt_vals = [e for _, e in egt_win]
        n = len(egt_vals)
        mu = sum(egt_vals) / n
        egt_std = math.sqrt(sum((e - mu) ** 2 for e in egt_vals) / (n - 1))
        terms: dict[str, float] = {
            "egt_dispersion": egt_std / mu if mu > 0.0 else float("nan"),
            "speed_variability": crank_cov_pct / 100.0,
        }

        # Thermal term: steady-state only.
        thermal_reason = ""
        cht_win = [(h[0], h[2]) for h in hist if h[2] is not None]
        pwr_win = [(h[0], h[3]) for h in hist if h[3] is not None]
        map_win = [(h[0], h[4]) for h in hist if h[4] is not None]
        cht_slope = self._slope([t for t, _ in cht_win], [v for _, v in cht_win]) if len(cht_win) >= cfg.csi_min_samples else None
        p_rate = self._slope([t for t, _ in pwr_win], [v for _, v in pwr_win]) if len(pwr_win) >= cfg.csi_min_samples else None
        m_rate = self._slope([t for t, _ in map_win], [v for _, v in map_win]) if len(map_win) >= cfg.csi_min_samples else None
        cht_span = (cht_win[-1][0] - cht_win[0][0]) if len(cht_win) >= 2 else 0.0
        if cht_slope is None:
            thermal_reason = "CHT slope unavailable"
        elif cht_span < cfg.csi_window_s - 1e-6:
            thermal_reason = f"CHT window filling ({cht_span:.0f} of {cfg.csi_window_s:.0f} s)"
        elif p_rate is None or m_rate is None:
            thermal_reason = "power or MAP rate unavailable; cannot confirm steady state"
        elif abs(p_rate) > cfg.csi_transient_max_power_rate_kw_s or abs(m_rate) > cfg.csi_transient_max_map_rate_pa_s:
            thermal_reason = (f"transient: dP/dt {p_rate:+.2f} kW/s, dMAP/dt {m_rate:+.0f} Pa/s "
                              f"(limits {cfg.csi_transient_max_power_rate_kw_s} kW/s, "
                              f"{cfg.csi_transient_max_map_rate_pa_s:.0f} Pa/s)")
        else:
            terms["thermal_slope"] = abs(cht_slope) / cfg.csi_reference_cht_slope_k_s

        valid_terms = {k: v for k, v in terms.items() if math.isfinite(v)}
        value = sum(valid_terms.values())
        coverage = len(valid_terms) / 3.0
        dominant = max(valid_terms, key=valid_terms.get)
        info = dict(valid_terms)
        info["coverage"] = coverage
        if cht_slope is not None:
            info["cht_slope_k_s"] = cht_slope
        if p_rate is not None:
            info["power_rate_kw_s"] = p_rate
        if m_rate is not None:
            info["map_rate_pa_s"] = m_rate
        reason = f"thermal term excluded: {thermal_reason}" if thermal_reason else ""
        return value, m09_csi_band(value), dominant, info, reason, coverage

    def evaluate(
        self,
        record: NormalizedSignalRecord,
        derived_state: DerivedEngineState | None = None,
        egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
        vib_diag: VibrationSignalProcessingResult | VibrationState | None = None,
    ) -> tuple[CombustionStabilityState, MisfireDiagnosticResult]:
        """Evaluate multi-signal combustion stability and misfire indicators."""
        raw_egts: list[tuple[int, ChannelValue]] = [
            (1, record.egt_cyl_1),
            (2, record.egt_cyl_2),
            (3, record.egt_cyl_3),
            (4, record.egt_cyl_4),
        ]

        dt_s: float | None = None
        if self._prev_timestamp is not None and record.timestamp is not None:
            dt_s = (record.timestamp - self._prev_timestamp).total_seconds()

        # Step 1: Crank Speed Irregularity Evidence
        rpm_val = record.rpm.value if (record.rpm.valid and record.rpm.value > 0) else None
        crank_evidence_val = 0.0
        crank_reasons: list[str] = []

        if rpm_val is not None and self._prev_rpm is not None and dt_s is not None and 0.0 < dt_s <= 10.0:
            delta_rpm = abs(rpm_val - self._prev_rpm)
            rate_rpm_s = delta_rpm / dt_s

            if rate_rpm_s >= self._cfg.rpm_std_threshold:
                crank_evidence_val = min(1.0, rate_rpm_s / (self._cfg.rpm_std_threshold * 2.0))
                crank_reasons.append(f"Crank speed fluctuation ({rate_rpm_s:.1f} RPM/s) detected")
        elif rpm_val is None:
            crank_reasons.append("RPM telemetry missing/invalid")

        # Step 2: Vibration Evidence
        vibration_evidence_val = 0.0
        vibration_reasons: list[str] = []

        overall_rms = 0.0
        if isinstance(vib_diag, VibrationSignalProcessingResult) and vib_diag.valid:
            overall_rms = vib_diag.overall_rms_m_s2
        elif isinstance(vib_diag, VibrationState) and vib_diag.overall_rms_m_s2.valid:
            overall_rms = vib_diag.overall_rms_m_s2.value
        else:
            vib_rms_ch = record.vibration_rms
            if vib_rms_ch.valid:
                overall_rms = vib_rms_ch.value

        if overall_rms >= self._cfg.vibration_rms_threshold_m_s2:
            vibration_evidence_val = min(1.0, overall_rms / (self._cfg.vibration_rms_threshold_m_s2 * 1.5))
            vibration_reasons.append(f"Elevated vibration RMS ({overall_rms:.1f} m/s²) detected")

        # Step 2b: Dual-channel AND gate (M-06). Decides WHETHER a misfire is
        # confirmed; the per-cylinder evidence below decides WHICH cylinder.
        gate = self._run_gate(record, vib_diag)
        confirmed = gate.verdict == MisfireVerdict.CONFIRMED
        if confirmed:
            alarm = (
                gate.crank_cov_pct >= self._cfg.gate_alarm_margin * gate.crank_limit_pct
                and gate.half_order_fraction >= self._cfg.gate_alarm_margin * gate.half_order_limit
            )
            confirmed_status = DiagnosticStatus.CRITICAL if alarm else DiagnosticStatus.WARNING
        gate_reason = f"Misfire gate {gate.verdict.value}: {gate.evidence}"

        # Step 2c: Combustion Stability Index (M-09). Uses channel A's crank CoV
        # directly: it must not depend on the vibration channel being present.
        csi_val, csi_band, csi_dom, csi_terms, csi_reason, csi_cov = self._combustion_stability(
            record, self._crank_cov_pct(record), derived_state
        )

        # Step 3: Extract EGT mean / deviations
        valid_egt_vals = [ch.value for _, ch in raw_egts if ch.valid and ch.value > 0.0]
        mean_egt = (sum(valid_egt_vals) / len(valid_egt_vals)) if valid_egt_vals else None

        # Step 4: Per-Cylinder Evidence Fusion
        cyl_evidences: list[CylinderCombustionEvidence] = []
        misfire_flags: list[bool] = [False, False, False, False]
        status_list: list[DiagnosticStatus] = [DiagnosticStatus.NORMAL] * 4
        score_list: list[float] = [0.0] * 4
        affected_cyls: list[int] = []

        overall_status = DiagnosticStatus.NORMAL if valid_egt_vals else DiagnosticStatus.INVALID
        new_prev_egts: dict[int, float] = {}

        for cyl_id, ch in raw_egts:
            idx = cyl_id - 1
            if not ch.valid or ch.value <= 0.0:
                cyl_evidences.append(
                    CylinderCombustionEvidence(
                        cylinder_id=cyl_id,
                        status=DiagnosticStatus.INVALID,
                        possible_misfire=False,
                        evidence_score=0.0,
                        egt_evidence=0.0,
                        crank_evidence=0.0,
                        vibration_evidence=0.0,
                        reasons=[ch.fault_flag or f"Cylinder {cyl_id} EGT channel invalid"],
                    )
                )
                status_list[idx] = DiagnosticStatus.INVALID
                continue

            egt_val = ch.value
            new_prev_egts[cyl_id] = egt_val

            egt_evidence_val = 0.0
            reasons: list[str] = []

            # 1. EGT drop / deviation indicators
            if mean_egt is not None:
                dev = egt_val - mean_egt
                if dev <= -self._cfg.egt_dev_threshold_k:
                    egt_evidence_val = min(1.0, abs(dev) / (self._cfg.egt_dev_threshold_k * 1.5))
                    reasons.append(f"Cylinder {cyl_id} EGT deviation ({dev:+.1f} K) below mean")

            # 2. Sudden temporal EGT drop
            if dt_s is not None and 0.0 < dt_s <= 10.0 and cyl_id in self._prev_egts:
                prev_val = self._prev_egts[cyl_id]
                egt_drop_rate = (prev_val - egt_val) / dt_s
                if egt_drop_rate >= self._cfg.egt_rate_drop_threshold_k_s:
                    egt_ev_drop = min(1.0, egt_drop_rate / (self._cfg.egt_rate_drop_threshold_k_s * 1.5))
                    egt_evidence_val = max(egt_evidence_val, egt_ev_drop)
                    reasons.append(f"Rapid EGT drop rate (-{egt_drop_rate:.1f} K/s)")

            # Multi-Signal Evidence Fusion
            total_score = (
                (self._cfg.egt_evidence_weight * egt_evidence_val)
                + (self._cfg.crank_evidence_weight * crank_evidence_val)
                + (self._cfg.vibration_evidence_weight * vibration_evidence_val)
            )
            total_score = float(min(1.0, max(0.0, total_score)))
            score_list[idx] = total_score

            status = DiagnosticStatus.NORMAL
            possible_misfire = False

            # Existing evidence logic: localises a candidate cylinder.
            if total_score >= self._cfg.misfire_confidence_threshold or (egt_evidence_val >= 0.8 and crank_evidence_val >= 0.5):
                possible_misfire = True
                reasons.append(f"High multi-signal evidence ({total_score:.2f}) indicates possible misfire")
            if possible_misfire or total_score >= self._cfg.unstable_confidence_threshold or egt_evidence_val >= 0.5:
                status = DiagnosticStatus.WARNING
                affected_cyls.append(cyl_id)
                if not possible_misfire:
                    reasons.append(f"Moderate evidence ({total_score:.2f}) indicates combustion instability")

            # Gate: only a CONFIRMED verdict makes a candidate a detected
            # misfire; otherwise candidates stay at WARNING at most.
            if status != DiagnosticStatus.NORMAL:
                if confirmed:
                    status = confirmed_status
                    misfire_flags[idx] = True
                    reasons.append(gate_reason)
                elif possible_misfire:
                    reasons.append(f"Not confirmed. {gate_reason}")

            status_list[idx] = status
            overall_status = _max_status(overall_status, status)

            all_reasons = reasons + crank_reasons + vibration_reasons
            cyl_evidences.append(
                CylinderCombustionEvidence(
                    cylinder_id=cyl_id,
                    status=status,
                    possible_misfire=possible_misfire,
                    evidence_score=total_score,
                    egt_evidence=egt_evidence_val,
                    crank_evidence=crank_evidence_val,
                    vibration_evidence=vibration_evidence_val,
                    reasons=all_reasons if all_reasons else ["Normal combustion"],
                )
            )

        if confirmed:
            overall_status = _max_status(overall_status, confirmed_status)
        elif gate.verdict == MisfireVerdict.UNCONFIRMED:
            overall_status = _max_status(overall_status, DiagnosticStatus.WARNING)
        elif gate.verdict == MisfireVerdict.INVALID and overall_status == DiagnosticStatus.NORMAL:
            overall_status = DiagnosticStatus.INVALID  # cannot claim NORMAL without both channels

        if record.timestamp is not None:
            self._prev_timestamp = record.timestamp
        self._prev_egts = new_prev_egts
        if rpm_val is not None:
            self._prev_rpm = rpm_val

        misfire_result = MisfireDiagnosticResult(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            cylinder_evidence=cyl_evidences,
            misfire_detected=misfire_flags,
            overall_status=overall_status,
            affected_cylinders=sorted(list(set(affected_cyls))),
            gate_verdict=gate.verdict.value,
            gate_triggering_channel=gate.triggering_channel,
            gate_evidence=gate.evidence,
            crank_cov_pct=_finite_or_none(gate.crank_cov_pct),
            half_order_fraction=_finite_or_none(gate.half_order_fraction),
            crank_cov_limit_pct=gate.crank_limit_pct or None,
            half_order_limit=gate.half_order_limit or None,
            csi_value=csi_val,
            csi_band=csi_band,
            csi_dominant_term=csi_dom,
            csi_terms=csi_terms,
            csi_reason=csi_reason,
        )

        stability_state = CombustionStabilityState(
            timestamp=record.timestamp,
            provenance=Provenance.DERIVED,
            misfire_detected=misfire_flags,
            misfire_status=status_list,
            evidence_score=score_list,
            overall_combustion_status=overall_status,
            misfire_verdict=gate.verdict.value,
            misfire_gate_channel=gate.triggering_channel,
            misfire_gate_evidence=gate.evidence,
            crank_cov_pct=_tagged(gate.crank_cov_pct),
            half_order_fraction=_tagged(gate.half_order_fraction),
            csi_value=_tagged(csi_val if csi_val is not None else float("nan")),
            csi_band=csi_band,
            csi_dominant_term=csi_dom,
            csi_terms=csi_terms,
        )

        return stability_state, misfire_result


def evaluate_misfire_detector(
    record: NormalizedSignalRecord,
    derived_state: DerivedEngineState | None = None,
    egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
    vib_diag: VibrationSignalProcessingResult | VibrationState | None = None,
    settings: AppSettings | None = None,
) -> tuple[CombustionStabilityState, MisfireDiagnosticResult]:
    """Convenience function for Module 10 misfire & combustion stability evaluation."""
    detector = MisfireDetector(settings)
    return detector.evaluate(record, derived_state, egt_diag, vib_diag)
