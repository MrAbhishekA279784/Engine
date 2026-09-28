"""
Health Index and Degradation Supervision — Original Module 14.

Third stage of L3 ML & Supervision Layer.
Provides deterministic, explainable aggregate engine Health Index (HI) calculation,
degradation state monitoring, trend supervision, and hysteresis filtering.

STRICT BOUNDARY CONSTRAINTS:
    - Diagnostic/supervisory ONLY. Zero engine/UAV control or actuation commands.
    - ML & physics inputs MUST NOT consume RawSignalRecord directly.
    - Health Index is strictly normalized and bounded in range [0.0, 1.0].
    - Missing evidence is NOT silently treated as 1.0 healthy; quality/validity degrade appropriately.
    - Zero RUL prediction or lifetime extrapolation (Module 15 owns RUL).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    DegradationState,
    DiagnosticStatus,
    FaultClass,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    AnomalyResult,
    CombustionStabilityState,
    DerivedEngineState,
    DiagnosticState,
    ElectricalState,
    FaultClassificationResult,
    HealthState,
    LubricationState,
    ProvenanceTaggedValue,
    ResidualState,
    VibrationState,
    make_tagged,
)
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticResult
from src.l2_digital_twin.overheat_trend import OperatingPointStepDetector, TrendWindow

logger = get_logger(__name__)


_NO_DATA_REASON = {
    "thermal_health": "no valid EGT diagnosis and no valid oil-temperature residual",
    "lubrication_health": "no valid oil pressure and no valid oil-pressure residual",
    "vibration_health": "no valid vibration RMS and no valid vibration residual",
    "combustion_health": "combustion stability state absent or INVALID",
    "performance_health": "no valid brake-power or MAP residual",
    "anomaly_fault_health": "no successful anomaly or fault-classifier result",
}


class HealthSupervisionEngine:
    """Supervisory engine for Health Index calculation and degradation supervision."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._config = self._settings.health

        # Internal state for trend supervision & hysteresis
        self._previous_timestamp: datetime | None = None
        self._previous_hi: float | None = None
        self._previous_degradation_state: DegradationState | None = None
        self._previous_trend: str = "INSUFFICIENT_DATA"
        # Health-index trend: the Prompt 14 windowed Theil-Sen code
        self._trend_window = TrendWindow(self._config.trend_window_s)
        self._trend_steps = OperatingPointStepDetector.from_settings(self._settings)

    def reset_state(self) -> None:
        """Reset historical supervision state for clean replay/testing."""
        self._previous_timestamp = None
        self._previous_hi = None
        self._previous_degradation_state = None
        self._previous_trend = "INSUFFICIENT_DATA"
        self._trend_window.clear()
        self._trend_steps.reset()

    def evaluate_health(
        self,
        residual_state: ResidualState | None = None,
        anomaly_result: AnomalyResult | None = None,
        fault_result: FaultClassificationResult | None = None,
        derived_state: DerivedEngineState | None = None,
        egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
        lub_state: LubricationState | None = None,
        vib_state: VibrationState | None = None,
        comb_state: CombustionStabilityState | None = None,
        timestamp: datetime | None = None,
        elec_state: ElectricalState | None = None,
    ) -> HealthState:
        """Evaluate aggregate engine Health Index and degradation state.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        # Strict boundary check against raw telemetry
        inputs = [residual_state, anomaly_result, fault_result, derived_state, egt_diag, lub_state, vib_state, comb_state,
                  elec_state]
        for arg in inputs:
            if isinstance(arg, RawSignalRecord):
                raise TypeError(
                    "STRICT BOUNDARY VIOLATION: HealthSupervisionEngine MUST NOT consume RawSignalRecord directly."
                )

        ts = (
            timestamp
            or (residual_state.timestamp if residual_state else None)
            or (anomaly_result.timestamp if anomaly_result else None)
            or (fault_result.timestamp if fault_result else None)
            or (derived_state.timestamp if derived_state else None)
            or datetime.now(timezone.utc)
        )

        # Evaluate individual health components
        comp_scores: dict[str, float] = {}
        comp_qualities: dict[str, float] = {}
        comp_weights: dict[str, float] = {}
        evidence: dict[str, Any] = {}

        # 1. Thermal Health
        t_score, t_qual, t_ev = self._eval_thermal_health(residual_state, egt_diag)
        comp_scores["thermal_health"] = t_score
        comp_qualities["thermal_health"] = t_qual
        comp_weights["thermal_health"] = self._config.weight_thermal
        evidence["thermal"] = t_ev

        # 2. Lubrication Health
        l_score, l_qual, l_ev = self._eval_lubrication_health(residual_state, lub_state)
        comp_scores["lubrication_health"] = l_score
        comp_qualities["lubrication_health"] = l_qual
        comp_weights["lubrication_health"] = self._config.weight_lubrication
        evidence["lubrication"] = l_ev

        # 3. Vibration Health
        v_score, v_qual, v_ev = self._eval_vibration_health(residual_state, vib_state)
        comp_scores["vibration_health"] = v_score
        comp_qualities["vibration_health"] = v_qual
        comp_weights["vibration_health"] = self._config.weight_vibration
        evidence["vibration"] = v_ev

        # 4. Combustion Health
        c_score, c_qual, c_ev = self._eval_combustion_health(comb_state)
        comp_scores["combustion_health"] = c_score
        comp_qualities["combustion_health"] = c_qual
        comp_weights["combustion_health"] = self._config.weight_combustion
        evidence["combustion"] = c_ev

        # 5. Performance Health
        p_score, p_qual, p_ev = self._eval_performance_health(residual_state, derived_state)
        comp_scores["performance_health"] = p_score
        comp_qualities["performance_health"] = p_qual
        comp_weights["performance_health"] = self._config.weight_performance
        evidence["performance"] = p_ev

        # 6. Anomaly & Fault Health
        af_score, af_qual, af_ev, contrib_fault = self._eval_anomaly_fault_health(anomaly_result, fault_result)
        comp_scores["anomaly_fault_health"] = af_score
        comp_qualities["anomaly_fault_health"] = af_qual
        comp_weights["anomaly_fault_health"] = self._config.weight_anomaly_fault
        evidence["anomaly_fault"] = af_ev

        # 7. Electrical Health (Prompt 12). When the electrical signals are not
        # instrumented (or nothing is derivable) the component is EXCLUDED: its
        # weight leaves both the HI and the quality denominator, and the
        # reduced coverage is reported in evidence["coverage"].
        e_score, e_qual, e_ev = self._eval_electrical_health(elec_state)
        evidence["electrical"] = e_ev
        excluded: list[str] = []
        if e_qual > 0.0:
            comp_scores["electrical_health"] = e_score
            comp_qualities["electrical_health"] = e_qual
            comp_weights["electrical_health"] = self._config.weight_electrical
        else:
            excluded.append("electrical_health")
        configured_weight = sum(comp_weights.values()) + (
            self._config.weight_electrical if excluded else 0.0)

        # Components without valid data: no score, a reason instead (OI-20)
        # A component has a score only if valid evidence was actually used.
        # Quality keeps its existing convention (e.g. an absent combustion
        # input still counts 0.5 toward evidence validity), but a component
        # without valid data is reported unavailable with its reason and does
        # not enter the Health Index average (no default 1.0 score).
        egt_status = getattr(egt_diag, "overall_status", None)
        comb_status = comb_state.overall_combustion_status if comb_state is not None else None
        valid_data = {
            "thermal_health": (egt_status is not None and egt_status != DiagnosticStatus.INVALID)
                              or "oil_temp_residual_c" in t_ev,
            "lubrication_health": "oil_pressure_bar" in l_ev or "oil_pressure_residual_pa" in l_ev,
            "vibration_health": "vibration_rms" in v_ev or "vibration_residual" in v_ev,
            "combustion_health": comb_status is not None and comb_status != DiagnosticStatus.INVALID,
            "performance_health": "brake_power_residual_kw" in p_ev or "map_residual_pa" in p_ev,
            "anomaly_fault_health": "anomaly_score" in af_ev or "predicted_class" in af_ev,
        }
        unavailable: dict[str, str] = {}
        if excluded:
            unavailable["electrical_health"] = e_ev.get("excluded", "no electrical data")
        for name, reason in _NO_DATA_REASON.items():
            if name in comp_qualities and (comp_qualities[name] <= 0.0 or not valid_data[name]):
                unavailable[name] = reason
        has_data = {k: comp_qualities[k] > 0.0 and k not in unavailable for k in comp_scores}
        data_weight = sum(w for k, w in comp_weights.items() if has_data[k])
        evidence["coverage"] = {
            "weight_included": sum(comp_weights.values()),
            "weight_configured": configured_weight,
            "coverage": sum(comp_weights.values()) / configured_weight if configured_weight > 0.0 else 0.0,
            "excluded": excluded,
            # share of the configured weight backed by valid data
            "data_coverage": data_weight / configured_weight if configured_weight > 0.0 else 0.0,
            "unavailable": dict(unavailable),
        }

        # Calculate weighted average HI & overall quality
        total_weight = sum(comp_weights.values())
        if total_weight <= 0.0:
            total_weight = 1.0

        # Sum scores over components with available evidence (quality > 0.0)
        active_weights = [comp_weights[k] for k in comp_scores if has_data[k]]
        active_scores = [comp_scores[k] * comp_weights[k] for k in comp_scores if has_data[k]]

        if active_weights and sum(active_weights) > 0.0:
            weighted_hi = sum(active_scores) / sum(active_weights)
        else:
            weighted_hi = 1.0

        weighted_quality = sum(comp_qualities[k] * comp_weights[k] for k in comp_qualities) / total_weight

        # Bounded clamping [0.0, 1.0]
        hi_clamped = max(0.0, min(1.0, weighted_hi))

        # Check for insufficient evidence / invalid inputs
        is_valid = weighted_quality >= self._config.min_evidence_quality
        if not is_valid:
            health_status = DiagnosticStatus.INVALID
            hi_clamped = 0.0
            deg_state = DegradationState.CRITICAL
        else:
            if hi_clamped >= self._config.healthy_threshold:
                health_status = DiagnosticStatus.NORMAL
            elif hi_clamped >= self._config.caution_threshold:
                health_status = DiagnosticStatus.WARNING
            else:
                health_status = DiagnosticStatus.CRITICAL

            # Apply degradation state mapping with hysteresis
            deg_state = self._determine_degradation_state(hi_clamped)

        # Trend supervision & rate of change
        trend, rate_of_change, trend_info = self._evaluate_trend(
            ts, hi_clamped, is_valid, residual_state.operating_point if residual_state else None)
        evidence["trend"] = trend_info

        # Update historical state
        if is_valid:
            self._previous_timestamp = ts
            self._previous_hi = hi_clamped
            self._previous_degradation_state = deg_state
            self._previous_trend = trend

        ptv_hi = make_tagged(
            value=hi_clamped,
            provenance=Provenance.DERIVED,
            valid=is_valid,
            quality=weighted_quality,
        )

        return HealthState(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            health_index=ptv_hi,
            degradation_state=deg_state,
            component_health={k: v for k, v in comp_scores.items() if has_data[k]},
            component_unavailable=unavailable,
            health_status=health_status,
            trend=trend,
            health_rate=rate_of_change,
            evidence=evidence,
            contributing_fault=contrib_fault,
            quality=weighted_quality,
            model_version=self._settings.version,
        )

    def _eval_thermal_health(
        self,
        residual_state: ResidualState | None,
        egt_diag: EGTDiagnosticResult | DiagnosticState | None,
    ) -> tuple[float, float, dict[str, Any]]:
        score = 1.0
        quality = 1.0
        evidence: dict[str, Any] = {}

        if egt_diag:
            if hasattr(egt_diag, "overall_status"):
                if egt_diag.overall_status == DiagnosticStatus.WARNING:
                    score -= 0.25
                elif egt_diag.overall_status == DiagnosticStatus.CRITICAL:
                    score -= 0.60
            if isinstance(egt_diag, EGTDiagnosticResult) and egt_diag.spread_egt_k is not None:
                evidence["egt_spread_k"] = egt_diag.spread_egt_k
                if egt_diag.spread_egt_k > 80.0:
                    score -= 0.20

        if residual_state and "oil_temp" in residual_state.residuals:
            ot_res = residual_state.residuals["oil_temp"]
            if ot_res.valid and ot_res.value is not None:
                evidence["oil_temp_residual_c"] = ot_res.value
                if ot_res.value > 10.0:
                    score -= min(0.4, (ot_res.value - 10.0) * 0.02)
            else:
                quality -= 0.3

        if not residual_state and not egt_diag:
            quality = 0.0

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, quality)), evidence

    def _eval_lubrication_health(
        self,
        residual_state: ResidualState | None,
        lub_state: LubricationState | None,
    ) -> tuple[float, float, dict[str, Any]]:
        score = 1.0
        quality = 1.0
        evidence: dict[str, Any] = {}

        if lub_state:
            p_bar = None
            if hasattr(lub_state, "oil_pressure_bar") and getattr(lub_state, "oil_pressure_bar") is not None:
                op_b = getattr(lub_state, "oil_pressure_bar")
                if getattr(op_b, "valid", True) and getattr(op_b, "value", None) is not None:
                    p_bar = getattr(op_b, "value")
            elif hasattr(lub_state, "oil_pressure_pa") and getattr(lub_state, "oil_pressure_pa") is not None:
                op_pa = getattr(lub_state, "oil_pressure_pa")
                if getattr(op_pa, "valid", True) and getattr(op_pa, "value", None) is not None:
                    p_bar = getattr(op_pa, "value") / 100000.0

            if p_bar is not None:
                evidence["oil_pressure_bar"] = p_bar
                if p_bar < 2.0:
                    score -= min(0.8, (2.0 - p_bar) * 0.5)
            else:
                quality -= 0.4

        if residual_state and "oil_pressure" in residual_state.residuals:
            op_res = residual_state.residuals["oil_pressure"]
            if op_res.valid and op_res.value is not None:
                evidence["oil_pressure_residual_pa"] = op_res.value
                if op_res.value < -20000.0:
                    score -= min(0.5, abs(op_res.value + 20000.0) / 100000.0)
            else:
                quality -= 0.3

        if not residual_state and not lub_state:
            quality = 0.0

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, quality)), evidence

    def _eval_vibration_health(
        self,
        residual_state: ResidualState | None,
        vib_state: VibrationState | None,
    ) -> tuple[float, float, dict[str, Any]]:
        score = 1.0
        quality = 1.0
        evidence: dict[str, Any] = {}

        if vib_state and vib_state.overall_rms_m_s2.valid and vib_state.overall_rms_m_s2.value is not None:
            rms = vib_state.overall_rms_m_s2.value
            evidence["vibration_rms"] = rms
            if rms > 10.0:
                score -= min(0.8, (rms - 10.0) * 0.05)
        elif not vib_state:
            quality -= 0.5

        if residual_state and "vibration_rms" in residual_state.residuals:
            vr_res = residual_state.residuals["vibration_rms"]
            if vr_res.valid and vr_res.value is not None:
                evidence["vibration_residual"] = vr_res.value
                if vr_res.value > 2.0:
                    score -= min(0.4, (vr_res.value - 2.0) * 0.1)

        if not residual_state and not vib_state:
            quality = 0.0

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, quality)), evidence

    def _eval_electrical_health(
        self,
        elec_state: ElectricalState | None,
    ) -> tuple[float, float, dict[str, Any]]:
        """Electrical component = L2 Electrical Health Index (EHI); quality =
        EHI quality x EHI coverage. Quality 0 (component excluded) when the
        state is absent or the EHI is not derivable."""
        if elec_state is None:
            return 1.0, 0.0, {"excluded": "no electrical state (electrical signals not instrumented)"}
        ehi = elec_state.ehi
        if not ehi.valid or ehi.value is None:
            return 1.0, 0.0, {"excluded": ehi.fault_flag or "EHI not derivable"}
        evidence: dict[str, Any] = {
            "ehi": ehi.value,
            "ehi_band": elec_state.ehi_band,
            "ehi_coverage": elec_state.ehi_coverage,
            "status": elec_state.status.value,
        }
        for name in ("charging_residual_median_v", "voltage_ripple_pct", "battery_resistance_mohm"):
            tv = getattr(elec_state, name)
            evidence[name] = tv.value if tv.valid else None
        return max(0.0, min(1.0, ehi.value)), max(0.0, min(1.0, ehi.quality * elec_state.ehi_coverage)), evidence

    def _eval_combustion_health(
        self,
        comb_state: CombustionStabilityState | None,
    ) -> tuple[float, float, dict[str, Any]]:
        score = 1.0
        quality = 1.0
        evidence: dict[str, Any] = {}

        if comb_state:
            if any(comb_state.misfire_detected):
                score -= 0.50
                evidence["misfire_detected"] = True
            if comb_state.overall_combustion_status == DiagnosticStatus.WARNING:
                score -= 0.20
            elif comb_state.overall_combustion_status == DiagnosticStatus.CRITICAL:
                score -= 0.50
        else:
            quality = 0.5

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, quality)), evidence

    def _eval_performance_health(
        self,
        residual_state: ResidualState | None,
        derived_state: DerivedEngineState | None,
    ) -> tuple[float, float, dict[str, Any]]:
        score = 1.0
        quality = 1.0
        evidence: dict[str, Any] = {}

        if residual_state:
            if "brake_power_kw" in residual_state.residuals:
                bp_res = residual_state.residuals["brake_power_kw"]
                if bp_res.valid and bp_res.value is not None:
                    evidence["brake_power_residual_kw"] = bp_res.value
                    if bp_res.value < -2.0:
                        score -= min(0.5, abs(bp_res.value + 2.0) * 0.05)
                else:
                    quality -= 0.3

            if "map_pressure" in residual_state.residuals:
                map_res = residual_state.residuals["map_pressure"]
                if map_res.valid and map_res.value is not None:
                    evidence["map_residual_pa"] = map_res.value
                    if map_res.value < -10000.0:
                        score -= min(0.4, abs(map_res.value + 10000.0) / 50000.0)

        if not residual_state and not derived_state:
            quality = 0.0

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, quality)), evidence

    def _eval_anomaly_fault_health(
        self,
        anomaly_result: AnomalyResult | None,
        fault_result: FaultClassificationResult | None,
    ) -> tuple[float, float, dict[str, Any], FaultClass | None]:
        score = 1.0
        quality = 1.0
        evidence: dict[str, Any] = {}
        contrib_fault: FaultClass | None = None

        if anomaly_result:
            if anomaly_result.status == InferenceStatus.MODEL_UNAVAILABLE:
                quality -= 0.2
                evidence["anomaly_status"] = "MODEL_UNAVAILABLE"
            elif anomaly_result.status == InferenceStatus.SUCCESS:
                evidence["anomaly_score"] = anomaly_result.anomaly_score
                if anomaly_result.is_anomaly:
                    score -= min(0.5, anomaly_result.anomaly_score * 0.5)

        if fault_result:
            if fault_result.status == InferenceStatus.MODEL_UNAVAILABLE:
                quality -= 0.2
                evidence["fault_status"] = "MODEL_UNAVAILABLE"
            elif fault_result.status == InferenceStatus.SUCCESS:
                evidence["predicted_class"] = fault_result.class_name
                if fault_result.predicted_class != FaultClass.NOMINAL:
                    contrib_fault = fault_result.predicted_class
                    penalty = 0.4 * (fault_result.confidence if fault_result.confidence is not None else 1.0)
                    score -= penalty

        if not anomaly_result and not fault_result:
            quality = 0.5

        return max(0.0, min(1.0, score)), max(0.0, min(1.0, quality)), evidence, contrib_fault

    def _determine_degradation_state(self, hi: float) -> DegradationState:
        """Map Health Index to DegradationState with hysteresis."""
        # Baseline threshold mapping
        if hi >= self._config.healthy_threshold:
            target_state = DegradationState.HEALTHY
        elif hi >= self._config.watch_threshold:
            target_state = DegradationState.WATCH
        elif hi >= self._config.caution_threshold:
            target_state = DegradationState.CAUTION
        elif hi >= self._config.warning_threshold:
            target_state = DegradationState.WARNING
        else:
            target_state = DegradationState.CRITICAL

        prev = self._previous_degradation_state
        if prev is None or prev == target_state:
            return target_state

        # State order index for hysteresis comparisons
        state_order = {
            DegradationState.HEALTHY: 4,
            DegradationState.WATCH: 3,
            DegradationState.CAUTION: 2,
            DegradationState.WARNING: 1,
            DegradationState.CRITICAL: 0,
        }

        # If transitioning to a BETTER (healthier) state, require hysteresis delta
        if state_order[target_state] > state_order[prev]:
            threshold_map = {
                DegradationState.HEALTHY: self._config.healthy_threshold,
                DegradationState.WATCH: self._config.watch_threshold,
                DegradationState.CAUTION: self._config.caution_threshold,
                DegradationState.WARNING: self._config.warning_threshold,
            }
            req_hi = threshold_map[target_state] + self._config.hysteresis_delta
            if hi < req_hi:
                return prev

        return target_state

    def _evaluate_trend(
        self,
        ts: datetime,
        current_hi: float,
        is_valid: bool,
        op: Any = None,
    ) -> tuple[str, float | None, dict[str, Any]]:
        """Health-index trend from the Theil-Sen slope over a rolling window
        (shared TrendWindow). DEGRADATION / IMPROVING only when the slope
        exceeds trend_min_rate_per_min AND its confidence interval excludes
        zero; otherwise STABLE. INSUFFICIENT_DATA until the window is full.
        An operating-point step restarts the window."""
        cfg = self._config
        win = self._trend_window
        if self._trend_steps.changed(op):
            win.clear()
        if is_valid:
            win.add(ts.timestamp(), current_hi)
        info: dict[str, Any] = {"window_s": cfg.trend_window_s, "samples": len(win), "span_s": win.span_s}
        full = len(win) >= cfg.trend_min_samples and win.span_s >= cfg.trend_window_s - win.median_dt_s()
        if not full:
            info["reason"] = (f"window has {len(win)} valid samples over {win.span_s:.0f} s "
                              f"(needs {cfg.trend_min_samples} over {cfg.trend_window_s:.0f} s)")
            return "INSUFFICIENT_DATA", None, info
        slope, lo, hi = win.fit(0, cfg.trend_confidence)
        slope_m, lo_m, hi_m = slope * 60.0, lo * 60.0, hi * 60.0
        info.update(slope_per_min=slope_m, ci_per_min=(lo_m, hi_m))
        if hi_m < 0.0 and -slope_m >= cfg.trend_min_rate_per_min:
            trend = "RAPID_DEGRADATION" if -slope_m >= cfg.trend_rapid_rate_per_min else "DEGRADATION"
        elif lo_m > 0.0 and slope_m >= cfg.trend_min_rate_per_min:
            trend = "IMPROVING"
        else:
            trend = "STABLE"
        return trend, slope, info
