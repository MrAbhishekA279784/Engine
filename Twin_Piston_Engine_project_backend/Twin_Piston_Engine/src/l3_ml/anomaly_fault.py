"""
Anomaly Detection and Fault Classifier — Original Module 13.

Second stage of L3 ML & Supervision Layer.
Provides generic anomaly detection and fault classification over the unified
14-class taxonomy (FaultClass 0-13, Prompt 15; docs/FAULT_TAXONOMY.md),
integrating directly with Original Module 12 ML Inference Infrastructure, plus
multi-label condition flags (symptoms that co-occur with any class).

STRICT BOUNDARY CONSTRAINTS:
    - ML MUST NOT consume RawSignalRecord directly. Features MUST come from Modules 5-11 / Module 12 feature vectors.
    - IDs 0-8 are the original nine classes; a legacy nine-class model's
      output (9 probabilities) is still accepted.
    - Reuses Module 12 model registry/infrastructure; does NOT implement a duplicate model loader.
    - Zero fabricated model predictions, zero fake accuracy claims.
    - Reports MODEL_UNAVAILABLE when trained model artifacts are missing.
    - Rule-based diagnostic fallbacks are explicitly flagged with is_ml=False and Provenance.DERIVED.
    - Zero advisory recommendations or flight control logic.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    CONDITION_FLAGS,
    FAULT_CLASS_COUNT,
    LEGACY_FAULT_CLASS_COUNT,
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
    FaultClassificationResult,
    LubricationState,
    ModelMetadata,
    ProvenanceTaggedValue,
    ResidualState,
    VibrationState,
)
from src.l2_digital_twin.egt_diagnostics import EGTDiagnosticResult
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l3_ml.ml_infrastructure import (
    MLFeatureVector,
    MLInferenceService,
)

logger = get_logger(__name__)

# Authoritative taxonomy map (FAULT_CLASS_COUNT classes)
FAULT_CLASS_MAP: dict[int, FaultClass] = {fc.value: fc for fc in FaultClass}


class AnomalyDetector:
    """Anomaly detection service operating over approved Module 12 feature vectors."""

    def __init__(
        self,
        inference_service: MLInferenceService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._inference_service = inference_service or MLInferenceService(settings=settings)
        self._settings = settings or get_settings()

    def detect_anomaly(
        self,
        feature_vector: MLFeatureVector,
        model_name: str = "anomaly_detector",
    ) -> AnomalyResult:
        """Evaluate anomaly detection model via Module 12 inference infrastructure.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        if isinstance(feature_vector, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: AnomalyDetector MUST NOT consume RawSignalRecord directly.")

        ts = feature_vector.timestamp

        # Check input validity
        if not feature_vector.valid:
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=InferenceStatus.INVALID_INPUT,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=0.5,
                evidence={"reason": "Invalid or non-finite feature vector"},
                quality=0.0,
                is_ml=True,
            )

        inf_result = self._inference_service.predict(model_name, feature_vector)

        if inf_result.status != InferenceStatus.SUCCESS:
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=inf_result.status,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=0.5,
                evidence={"error": inf_result.error_message or str(inf_result.status)},
                quality=inf_result.quality,
                model_metadata=inf_result.model_metadata,
                is_ml=True,
            )

        # Interpret output prediction from model
        pred = inf_result.prediction
        is_anomaly = False
        score = 0.0
        threshold = 0.5
        evidence: dict[str, Any] = {}

        if isinstance(pred, dict):
            score = float(pred.get("score", 0.0))
            threshold = float(pred.get("threshold", 0.5))
            is_anomaly = bool(pred.get("is_anomaly", score >= threshold))
            evidence = pred.get("evidence", {})
        elif isinstance(pred, (int, float)):
            score = float(pred)
            threshold = 0.5
            is_anomaly = score >= threshold
            evidence = {"raw_score": score}
        elif isinstance(pred, (list, tuple)) and len(pred) > 0:
            score = float(pred[0])
            threshold = 0.5
            is_anomaly = score >= threshold
            evidence = {"raw_score": score}

        # Finite check on score
        if math.isnan(score) or math.isinf(score):
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=InferenceStatus.INFERENCE_ERROR,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=threshold,
                evidence={"reason": "Non-finite anomaly score returned by model"},
                quality=0.0,
                model_metadata=inf_result.model_metadata,
                is_ml=True,
            )

        return AnomalyResult(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            status=InferenceStatus.SUCCESS,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            threshold=threshold,
            evidence=evidence,
            quality=inf_result.quality,
            model_metadata=inf_result.model_metadata,
            is_ml=True,
        )

    def detect_anomaly_rule_fallback(
        self,
        residual_state: ResidualState,
        threshold: float = 3.0,
    ) -> AnomalyResult:
        """Deterministic rule/threshold-based anomaly detector.

        Used when ML model is unavailable or for deterministic validation.
        Clearly tags output as is_ml=False and Provenance.DERIVED.
        """
        if isinstance(residual_state, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: AnomalyDetector MUST NOT consume RawSignalRecord directly.")

        ts = residual_state.timestamp
        max_residual = 0.0
        evidence: dict[str, Any] = {}
        invalid_count = 0

        for key, ptv in residual_state.residuals.items():
            if not ptv.valid or ptv.value is None or math.isnan(ptv.value) or math.isinf(ptv.value):
                invalid_count += 1
                continue
            abs_val = abs(ptv.value)
            evidence[key] = abs_val
            if abs_val > max_residual:
                max_residual = abs_val

        if invalid_count == len(residual_state.residuals) and len(residual_state.residuals) > 0:
            return AnomalyResult(
                timestamp=ts,
                provenance=Provenance.DERIVED,
                status=InferenceStatus.INVALID_INPUT,
                is_anomaly=False,
                anomaly_score=0.0,
                threshold=threshold,
                evidence={"reason": "All input residuals are invalid"},
                quality=0.0,
                is_ml=False,
            )

        is_anomaly = max_residual >= threshold
        # Normalize score relative to threshold (cap at 10.0 for safety)
        norm_score = min(max_residual / max(threshold, 1e-6), 10.0)

        return AnomalyResult(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            status=InferenceStatus.SUCCESS,
            is_anomaly=is_anomaly,
            anomaly_score=norm_score,
            threshold=threshold,
            evidence=evidence,
            quality=1.0 if invalid_count == 0 else max(0.0, 1.0 - (invalid_count / len(residual_state.residuals))),
            is_ml=False,
        )


_FLAGGED = (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)


def _tv(tv) -> float | None:
    return tv.value if (tv is not None and getattr(tv, "valid", False) and tv.value is not None) else None


def evaluate_condition_flags(
    overheat_state: Any = None,
    comb_state: CombustionStabilityState | None = None,
    lub_state: LubricationState | None = None,
    settings: AppSettings | None = None,
) -> tuple[dict[str, bool | None], dict[str, Any]]:
    """Multi-label condition flags (symptoms; independent of the class).

        overheating_trend       a Prompt 14 time-to-limit is valid and within the
                                alert horizon (overheat.alert_horizon_s)
        combustion_instability  CSI band WARNING or ALARM (M-09)
        lubrication_degraded    LHI band WARNING or ALARM (M-10)

    None means "not evaluable" (input absent or band UNKNOWN), never a
    substituted False.
    """
    settings = settings or get_settings()
    flags: dict[str, bool | None] = {}
    evidence: dict[str, Any] = {}

    if overheat_state is None:
        flags["overheating_trend"] = None
        evidence["overheating_trend"] = "no overheat trend state"
    else:
        horizon = settings.overheat.alert_horizon_s
        hits = {name: t.time_to_limit_s.value for name, t in overheat_state.channels.items()
                if t.time_to_limit_s.valid and t.time_to_limit_s.value <= horizon}
        evaluable = any(t.current_c.valid for t in overheat_state.channels.values())
        flags["overheating_trend"] = bool(hits) if evaluable else None
        evidence["overheating_trend"] = ({"time_to_limit_s": hits, "horizon_s": horizon} if evaluable
                                         else "no temperature channel valid")

    if comb_state is None or comb_state.csi_band in ("UNKNOWN", ""):
        flags["combustion_instability"] = None
        evidence["combustion_instability"] = "CSI not available" if comb_state is None else "CSI band UNKNOWN"
    else:
        flags["combustion_instability"] = comb_state.csi_band in ("WARNING", "ALARM")
        evidence["combustion_instability"] = {"csi": _tv(comb_state.csi_value), "csi_band": comb_state.csi_band}

    if lub_state is None or lub_state.lhi_band in ("UNKNOWN", ""):
        flags["lubrication_degraded"] = None
        evidence["lubrication_degraded"] = "LHI not available" if lub_state is None else "LHI band UNKNOWN"
    else:
        flags["lubrication_degraded"] = lub_state.lhi_band in ("WARNING", "ALARM")
        evidence["lubrication_degraded"] = {"lhi": _tv(lub_state.lhi), "lhi_band": lub_state.lhi_band}

    assert set(flags) == set(CONDITION_FLAGS)
    return flags, evidence


class FaultClassifier:
    """Fault classification service (unified 14-class taxonomy) integrating with
    Module 12 ML infrastructure, with a physics/diagnostic rule fallback."""

    def __init__(
        self,
        inference_service: MLInferenceService | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self._inference_service = inference_service or MLInferenceService(settings=settings)
        self._settings = settings or get_settings()

    def classify_fault(
        self,
        feature_vector: MLFeatureVector,
        model_name: str = "fault_classifier",
    ) -> FaultClassificationResult:
        """Classify fault into the unified taxonomy using Module 12 ML infrastructure.

        Raises TypeError if caller attempts to pass RawSignalRecord directly.
        """
        if isinstance(feature_vector, RawSignalRecord):  # type: ignore[unreachable]
            raise TypeError("STRICT BOUNDARY VIOLATION: FaultClassifier MUST NOT consume RawSignalRecord directly.")

        ts = feature_vector.timestamp

        # Validate feature vector input
        if not feature_vector.valid:
            return FaultClassificationResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=InferenceStatus.INVALID_INPUT,
                predicted_class=FaultClass.NOMINAL,
                class_id=FaultClass.NOMINAL.value,
                class_name=FaultClass.NOMINAL.name,
                confidence=None,
                probabilities=None,
                quality=0.0,
                is_ml=True,
            )

        inf_result = self._inference_service.predict(model_name, feature_vector)

        if inf_result.status != InferenceStatus.SUCCESS:
            return FaultClassificationResult(
                timestamp=ts,
                provenance=Provenance.MODEL_OUTPUT,
                status=inf_result.status,
                predicted_class=FaultClass.NOMINAL,
                class_id=FaultClass.NOMINAL.value,
                class_name=FaultClass.NOMINAL.name,
                confidence=None,
                probabilities=None,
                quality=inf_result.quality,
                model_metadata=inf_result.model_metadata,
                is_ml=True,
            )

        pred = inf_result.prediction
        predicted_class = FaultClass.NOMINAL
        confidence: float | None = None
        probabilities: dict[str, float] | None = None
        evidence: dict[str, Any] | None = None

        # Format A: Dictionary of class probabilities or dict with "predicted_class" / "probabilities"
        if isinstance(pred, dict):
            if "probabilities" in pred and isinstance(pred["probabilities"], (dict, list)):
                raw_probs = pred["probabilities"]
                prob_dict, conf, p_class = self._parse_probabilities(raw_probs)
                if prob_dict is not None:
                    probabilities = prob_dict
                    confidence = conf
                    predicted_class = p_class
                else:
                    return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)
                # Trained model (Prompt 16): the probabilities are calibrated on full
                # feature coverage, so confidence is scaled by the importance-weighted
                # share of features present (sensor dropout -> reduced confidence).
                cov = pred.get("feature_coverage")
                if cov is not None and confidence is not None:
                    confidence = confidence * max(0.0, min(1.0, float(cov)))
                if isinstance(pred.get("evidence"), dict):
                    evidence = dict(pred["evidence"])
                    evidence["max_probability"] = conf
            elif "class_id" in pred:
                cid = int(pred["class_id"])
                if cid in FAULT_CLASS_MAP:
                    predicted_class = FAULT_CLASS_MAP[cid]
                    confidence = float(pred.get("confidence")) if pred.get("confidence") is not None else None
                else:
                    return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)
            else:
                # Direct dict of class_name -> float
                prob_dict, conf, p_class = self._parse_probabilities(pred)
                if prob_dict is not None:
                    probabilities = prob_dict
                    confidence = conf
                    predicted_class = p_class
                else:
                    return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)

        # Format B: List/Sequence of 9 probabilities
        elif isinstance(pred, (list, tuple)):
            prob_dict, conf, p_class = self._parse_probabilities(pred)
            if prob_dict is not None:
                probabilities = prob_dict
                confidence = conf
                predicted_class = p_class
            else:
                return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)

        # Format C: Scalar integer class_id
        elif isinstance(pred, int):
            if pred in FAULT_CLASS_MAP:
                predicted_class = FAULT_CLASS_MAP[pred]
                confidence = None
                probabilities = None
            else:
                return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)
        else:
            return self._error_result(ts, InferenceStatus.SCHEMA_MISMATCH, inf_result.model_metadata)

        return FaultClassificationResult(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            status=InferenceStatus.SUCCESS,
            predicted_class=predicted_class,
            class_id=predicted_class.value,
            class_name=predicted_class.name,
            confidence=confidence,
            probabilities=probabilities,
            feature_schema_version=inf_result.feature_schema_version,
            quality=inf_result.quality,
            evidence=evidence,
            model_metadata=inf_result.model_metadata,
            is_ml=True,
        )

    def classify_fault_rule_fallback(
        self,
        residual_state: ResidualState | None = None,
        egt_diag: EGTDiagnosticResult | DiagnosticState | None = None,
        vib_state: VibrationState | None = None,
        comb_state: CombustionStabilityState | None = None,
        lub_state: LubricationState | None = None,
        injection_state: Any = None,
        elec_state: Any = None,
        coolant_state: Any = None,
        overheat_state: Any = None,
        norm_record: Any = None,
        timestamp: datetime | None = None,
    ) -> FaultClassificationResult:
        """Deterministic rule fallback over the L2 evidence (is_ml=False,
        Provenance.DERIVED), for every class of the unified taxonomy.

        Rules are evaluated in a fixed order of root-cause specificity; the
        first that fires is the predicted class, the others that also fired
        are listed in evidence["also_consistent_with"]. Each rule uses an L2
        status or band, not a new threshold:

            SENSOR_FAULT        an instrumented channel is invalid in this record
            MISFIRE             M-06 dual-channel gate CONFIRMED
            INJECTOR_FAULT      injector flow ratio identifies a cylinder (Prompt 13)
            FUEL_SYSTEM_FAULT   fuel-system status WARNING+ (rail residual / delivery ratio)
            DETONATION_KNOCK    consistent ignition retard on a cylinder (Prompt 13)
            CHARGING_FAULT      ripple status WARNING+, or charging status WARNING+
                                while the regulator holds the bus (Prompt 12)
            BATTERY_DEGRADATION battery (R_int) status WARNING+, or charging status
                                WARNING+ below cut-in (regime BATTERY)
            IMBALANCE           propeller 1X dominates lateral velocity, lateral > vertical (M-12)
            BEARING_WEAR        VHI WARNING+ with the envelope term dominant (M-04, M-10)
            OIL_DEGRADATION     LHI WARNING+ (M-10)
            COOLING_FAULT       coolant status WARNING+ (Prompt 14)
            EXHAUST_VALVE_LEAK  EGT diagnosis WARNING+ with a cold cylinder
            INTAKE_BOOST_LEAK   MAP residual below -10 kPa; the pipeline has no
                                MAP expectation (no boost reference: wastegate or
                                throttle position not instrumented), so it is not
                                evaluable there

        Legacy rules from the pre-Prompt 15 fallback are kept for callers that
        supply only their inputs (overall vibration RMS > 15 m/s^2, oil pressure
        < 1.5 bar, oil-temperature residual > 15 K, MAP residual < -10 kPa).
        Each applies only when the modern evidence for its class is absent (VHI
        band UNKNOWN, LHI band UNKNOWN, no coolant state): with the pipeline's
        states the oil-temperature rule fired on a healthy engine because of the
        uncalibrated oil-temperature expectation (OI-5).

        confidence is None: a rule gives no calibrated probability.
        """
        cfg = self._settings
        fired: list[tuple[FaultClass, dict[str, Any]]] = []
        not_evaluable: dict[str, str] = {}

        def fire(fc: FaultClass, ev: dict[str, Any]) -> None:
            fired.append((fc, ev))

        if norm_record is not None:
            if norm_record.invalid_channels:
                fire(FaultClass.SENSOR_FAULT, {"invalid_channels": list(norm_record.invalid_channels)})
        else:
            not_evaluable["SENSOR_FAULT"] = "no normalized record"

        # misfire_detected is set only when the M-06 gate confirms, so either is the confirmation
        if comb_state is not None and (comb_state.misfire_verdict == "CONFIRMED" or any(comb_state.misfire_detected)):
            fire(FaultClass.MISFIRE, {
                "misfire_cylinders": [i + 1 for i, m in enumerate(comb_state.misfire_detected) if m],
                "crank_cov_pct": _tv(comb_state.crank_cov_pct),
                "half_order_fraction": _tv(comb_state.half_order_fraction)})

        if injection_state is not None:
            if injection_state.identified_injectors:
                fire(FaultClass.INJECTOR_FAULT, {
                    "cylinders": list(injection_state.identified_injectors),
                    "injector_flow_ratio": [_tv(t) for t in injection_state.injector_flow_ratio_median]})
            if injection_state.fuel_system_status in _FLAGGED:
                fire(FaultClass.FUEL_SYSTEM_FAULT, {
                    "rail_pressure_residual_kpa": _tv(injection_state.rail_pressure_residual_median_kpa),
                    "fuel_delivery_ratio": _tv(injection_state.fuel_delivery_ratio_median)})
            if any(injection_state.knock_suspected):
                fire(FaultClass.DETONATION_KNOCK, {
                    "cylinders": [i + 1 for i, k in enumerate(injection_state.knock_suspected) if k],
                    "ignition_retard_deg": [_tv(t) for t in injection_state.ign_timing_residual_median_deg],
                    "knock_status": injection_state.knock_status.value})
        else:
            not_evaluable.update({c: "no injection state" for c in
                                  ("INJECTOR_FAULT", "FUEL_SYSTEM_FAULT", "DETONATION_KNOCK")})

        if elec_state is not None:
            # The charging residual judges the alternator only while it regulates
            # the bus; below cut-in (regime BATTERY) it measures the battery.
            regulated = elec_state.charging_regime == "REGULATED"
            charging_dev = elec_state.charging_status in _FLAGGED
            if (charging_dev and regulated) or elec_state.ripple_status in _FLAGGED:
                fire(FaultClass.CHARGING_FAULT, {
                    "charging_residual_v": _tv(elec_state.charging_residual_median_v),
                    "voltage_ripple_pct": _tv(elec_state.voltage_ripple_pct),
                    "charging_status": elec_state.charging_status.value,
                    "ripple_status": elec_state.ripple_status.value,
                    "charging_regime": elec_state.charging_regime})
            if elec_state.battery_status in _FLAGGED or (charging_dev and elec_state.charging_regime == "BATTERY"):
                fire(FaultClass.BATTERY_DEGRADATION, {
                    "battery_resistance_mohm": _tv(elec_state.battery_resistance_mohm),
                    "battery_status": elec_state.battery_status.value,
                    "charging_residual_v": _tv(elec_state.charging_residual_median_v),
                    "charging_regime": elec_state.charging_regime})
        else:
            not_evaluable.update({c: "no electrical state" for c in ("CHARGING_FAULT", "BATTERY_DEGRADATION")})

        if vib_state is not None:
            vc = cfg.vibration
            frac, ratio = _tv(vib_state.prop_1x_velocity_fraction_lateral), _tv(vib_state.prop_1x_lateral_vertical_ratio)
            if frac is not None and ratio is not None and frac >= vc.imbalance_dominance_fraction \
                    and ratio > vc.imbalance_lateral_ratio_min:
                fire(FaultClass.IMBALANCE, {"prop_1x_hz": _tv(vib_state.prop_1x_hz),
                                            "prop_1x_velocity_fraction_lateral": frac,
                                            "prop_1x_lateral_vertical_ratio": ratio})
            rms = _tv(vib_state.overall_rms_m_s2)
            if vib_state.vhi_band in ("WARNING", "ALARM") and vib_state.vhi_dominant_factor == "envelope":
                fire(FaultClass.BEARING_WEAR, {"vhi": _tv(vib_state.vibration_health_index),
                                               "vhi_band": vib_state.vhi_band,
                                               "envelope_rms_m_s2": _tv(vib_state.envelope_rms_m_s2)})
            elif vib_state.vhi_band == "UNKNOWN" and rms is not None and rms > 15.0:  # legacy rule, no VHI
                fire(FaultClass.BEARING_WEAR, {"vibration_rms": rms, "legacy_rule": "overall RMS > 15 m/s^2"})
        else:
            not_evaluable.update({c: "no vibration state" for c in ("IMBALANCE", "BEARING_WEAR")})

        if lub_state is not None:
            p_oil = _tv(lub_state.oil_pressure_pa)
            if lub_state.lhi_band in ("WARNING", "ALARM"):
                fire(FaultClass.OIL_DEGRADATION, {"lhi": _tv(lub_state.lhi), "lhi_band": lub_state.lhi_band})
            elif lub_state.lhi_band == "UNKNOWN" and p_oil is not None and p_oil < 150000.0:  # legacy rule, no LHI
                fire(FaultClass.OIL_DEGRADATION, {"oil_pressure_bar": p_oil / 1e5, "legacy_rule": "oil pressure < 1.5 bar"})

        res = residual_state.residuals if residual_state is not None else {}
        oil_t = _tv(res.get("oil_temp"))
        if coolant_state is not None and coolant_state.status in _FLAGGED:
            fire(FaultClass.COOLING_FAULT, {"coolant_residual_k": _tv(coolant_state.coolant_residual_median_k),
                                            "coolant_cht_delta_k": _tv(coolant_state.coolant_cht_delta_k)})
        elif coolant_state is None and oil_t is not None and oil_t > 15.0:  # legacy rule, no coolant state
            fire(FaultClass.COOLING_FAULT, {"oil_temp_residual_c": oil_t, "legacy_rule": "oil-temperature residual > 15 K"})

        if isinstance(egt_diag, EGTDiagnosticResult) and egt_diag.overall_status in _FLAGGED \
                and norm_record is not None:
            egts = [c for c in (norm_record.egt_cyl_1, norm_record.egt_cyl_2, norm_record.egt_cyl_3,
                                norm_record.egt_cyl_4)]
            vals = [c.value for c in egts if c.valid]
            if len(vals) >= 3:
                devs = [(i + 1, c.value - (sum(vals) - c.value) / (len(vals) - 1))
                        for i, c in enumerate(egts) if c.valid]
                cyl, dev = min(devs, key=lambda d: d[1])
                if dev <= -cfg.egt_diagnostics.egt_dev_warning_k:
                    fire(FaultClass.EXHAUST_VALVE_LEAK, {"cold_cylinder": cyl, "egt_deviation_k": dev,
                                                         "egt_spread_k": egt_diag.spread_egt_k})
        map_res = _tv(res.get("map_pressure"))
        if map_res is not None and map_res < -10000.0:  # legacy rule: needs a MAP expectation (boost reference)
            fire(FaultClass.INTAKE_BOOST_LEAK, {"map_residual_pa": map_res})
        elif map_res is None:
            not_evaluable["INTAKE_BOOST_LEAK"] = (
                "no MAP residual: no boost reference (wastegate or throttle position) is instrumented, so a "
                "low MAP cannot be told from a lower power setting")

        predicted, evidence = (fired[0] if fired else (FaultClass.NOMINAL, {}))
        evidence = dict(evidence)
        evidence["rule"] = predicted.name
        evidence["also_consistent_with"] = [fc.name for fc, _ in fired[1:]]
        evidence["not_evaluable"] = not_evaluable

        flags, flag_evidence = evaluate_condition_flags(overheat_state, comb_state, lub_state, cfg)
        ts = timestamp or next((x.timestamp for x in (residual_state, comb_state, lub_state, vib_state,
                                                      injection_state, elec_state) if x is not None
                                and getattr(x, "timestamp", None) is not None), datetime.now(timezone.utc))
        return FaultClassificationResult(
            timestamp=ts,
            provenance=Provenance.DERIVED,
            status=InferenceStatus.SUCCESS,
            predicted_class=predicted,
            class_id=predicted.value,
            class_name=predicted.name,
            confidence=None,       # rules give no calibrated probability
            probabilities=None,
            quality=1.0,
            evidence=evidence,
            is_ml=False,
            condition_flags=flags,
            condition_evidence=flag_evidence,
        )

    def _parse_probabilities(
        self,
        probs: Any,
    ) -> tuple[dict[str, float] | None, float | None, FaultClass]:
        """Validate and format a class probability distribution: FAULT_CLASS_COUNT
        classes, or the LEGACY_FAULT_CLASS_COUNT (0-8) of a pre-Prompt 15 model."""
        prob_dict: dict[str, float] = {}

        if isinstance(probs, dict):
            # Map by class name or integer string
            for k, v in probs.items():
                val = float(v)
                if math.isnan(val) or math.isinf(val) or val < 0.0:
                    return None, None, FaultClass.NOMINAL
                if k in FAULT_CLASS_MAP:
                    fc = FAULT_CLASS_MAP[int(k)]
                    prob_dict[fc.name] = val
                elif hasattr(FaultClass, str(k)):
                    prob_dict[str(k)] = val
                else:
                    return None, None, FaultClass.NOMINAL

            if len(prob_dict) not in (LEGACY_FAULT_CLASS_COUNT, FAULT_CLASS_COUNT):
                return None, None, FaultClass.NOMINAL

        elif isinstance(probs, (list, tuple)):
            if len(probs) not in (LEGACY_FAULT_CLASS_COUNT, FAULT_CLASS_COUNT):
                return None, None, FaultClass.NOMINAL
            for i, p in enumerate(probs):
                val = float(p)
                if math.isnan(val) or math.isinf(val) or val < 0.0:
                    return None, None, FaultClass.NOMINAL
                fc = FAULT_CLASS_MAP[i]
                prob_dict[fc.name] = val
        else:
            return None, None, FaultClass.NOMINAL

        # Find max probability class
        best_class_name = max(prob_dict, key=lambda k: prob_dict[k])
        best_prob = prob_dict[best_class_name]
        best_fc = getattr(FaultClass, best_class_name)

        return prob_dict, best_prob, best_fc

    def _error_result(
        self,
        ts: datetime,
        status: InferenceStatus,
        metadata: ModelMetadata | None,
    ) -> FaultClassificationResult:
        return FaultClassificationResult(
            timestamp=ts,
            provenance=Provenance.MODEL_OUTPUT,
            status=status,
            predicted_class=FaultClass.NOMINAL,
            class_id=FaultClass.NOMINAL.value,
            class_name=FaultClass.NOMINAL.name,
            confidence=None,
            probabilities=None,
            quality=0.0,
            model_metadata=metadata,
            is_ml=True,
        )


# Deprecated alias (Prompt 15): the classifier is no longer nine-class.
NineClassFaultClassifier = FaultClassifier
