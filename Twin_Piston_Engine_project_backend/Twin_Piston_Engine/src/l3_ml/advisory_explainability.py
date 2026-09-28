"""
Advisory and Explainability Engine — Original Module 19.

Eighth stage of L3 Machine Learning & Supervision Layer.
Converts diagnostic/supervisory outputs (Modules 7–18) into:
    - Evidence-backed human-readable explanations
    - Structured evidence item models and ranked contributing factors
    - Fault, Health, RUL, Mission Risk, and What-if Scenario explanations
    - Deterministic decision-support advisories with priority and category classifications
    - Operator-facing diagnostic query service ("What is health?", "Why degraded?", "What changed?")

STRICT SAFETY & BOUNDARY CONSTRAINTS:
    - MUST NOT consume RawSignalRecord directly (raises TypeError).
    - MUST NOT access simulator internal ground-truth state.
    - ZERO autonomous control, actuator commands, or UAV flight trajectory alterations.
    - ZERO fabricated evidence, false causality, or unbacked maintenance/certification claims.
    - 100% deterministic, testable logic (zero LLM / external unvetted text generators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import (
    DiagnosticStatus,
    FaultClass,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import (
    ElectricalState,
    AnomalyResult,
    CombustionStabilityState,
    DegradationState,
    DerivedEngineState,
    DiagnosticState,
    FaultClassificationResult,
    HealthState,
    LubricationState,
    ModelMetadata,
    ProvenanceTaggedValue,
    RULState,
    ResidualState,
    VibrationState,
)
from src.l1_data.raw_signal_record import RawSignalRecord

logger = get_logger(__name__)


class AdvisoryCategory(str, Enum):
    """Authoritative prototype advisory action categories."""

    MONITOR = "MONITOR"
    INSPECT = "INSPECT"
    INVESTIGATE = "INVESTIGATE"
    MAINTENANCE_REVIEW = "MAINTENANCE_REVIEW"
    DATA_QUALITY_CHECK = "DATA_QUALITY_CHECK"
    CONTINUE_MONITORING = "CONTINUE_MONITORING"


class AdvisoryPriority(str, Enum):
    """Deterministic advisory priority levels."""

    INFORMATION = "INFORMATION"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class QuestionType(str, Enum):
    """Supported operator diagnostic query question types."""

    HEALTH_STATUS = "HEALTH_STATUS"
    HEALTH_REASON = "HEALTH_REASON"
    FAULT_STATUS = "FAULT_STATUS"
    FAULT_EVIDENCE = "FAULT_EVIDENCE"
    RUL_ESTIMATE = "RUL_ESTIMATE"
    MISSION_RISK = "MISSION_RISK"
    WHAT_CHANGED = "WHAT_CHANGED"
    RECOMMENDED_ACTION = "RECOMMENDED_ACTION"


class EvidenceItem(BaseModel):
    """Container for a single structured diagnostic evidence item."""

    source: str = Field(description="Originating module or diagnostic subsystem")
    metric: str = Field(description="Signal or feature name evaluated")
    observed_value: float | None = Field(default=None, description="Observed engineering value")
    reference_value: float | None = Field(default=None, description="Healthy baseline or expected value")
    deviation: float | None = Field(default=None, description="Residual or deviation magnitude")
    status: str = Field(default="NORMAL", description="Status code or channel status flag")
    contribution: float = Field(default=0.0, ge=0.0, description="Normalized severity/contribution weight")
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Signal or feature quality score")
    provenance: Provenance = Field(default=Provenance.DERIVED, description="Data origin provenance")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    explanation: str = Field(default="", description="Concise textual summary of evidence item")

    model_config = ConfigDict(frozen=True)


class Explanation(BaseModel):
    """Structured human-readable explanation container."""

    explanation_id: str = Field(description="Unique explanation ID")
    finding: str = Field(description="Primary summary finding statement")
    observation: str = Field(description="Objective measurement or inference observation")
    interpretation: str = Field(description="Technical interpretation of evidence")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="List of supporting evidence items")
    uncertainty: str = Field(default="", description="Signal or model uncertainty disclaimer")
    limitation: str = Field(default="", description="Prototype and analytical boundary limitation statement")
    provenance: Provenance = Field(default=Provenance.DERIVED)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)


class Advisory(BaseModel):
    """Deterministic prototype decision-support advisory container."""

    advisory_id: str = Field(description="Unique advisory notification identifier")
    category: AdvisoryCategory = Field(description="Action category")
    priority: AdvisoryPriority = Field(description="Deterministic priority level")
    title: str = Field(description="Advisory header title")
    message: str = Field(description="Concise advisory text message")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Triggering evidence items")
    subsystem: str = Field(default="GENERAL", description="Affected engine subsystem")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Diagnostic confidence score")
    provenance: Provenance = Field(default=Provenance.DERIVED)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    limitations: str = Field(default="Prototype decision support guidance only; not certified maintenance instruction.")

    model_config = ConfigDict(frozen=True)


class DiagnosticAnswer(BaseModel):
    """Response structure for operator diagnostic queries."""

    question_type: str = Field(description="Requested query category")
    answer: str = Field(description="Direct textual answer to the query")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Supporting evidence items")
    limitations: str = Field(default="", description="Data or model limitations")
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Provenance = Field(default=Provenance.DERIVED)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)


class ExplainabilityEngine:
    """Engine providing evidence extraction and structured diagnostic explanations."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def _check_boundary(self, *inputs: Any) -> None:
        for arg in inputs:
            if isinstance(arg, RawSignalRecord):
                raise TypeError("STRICT BOUNDARY VIOLATION: ExplainabilityEngine MUST NOT consume RawSignalRecord directly.")

    def extract_evidence_items(
        self,
        residual_state: ResidualState | None = None,
        anomaly_result: AnomalyResult | None = None,
        fault_result: FaultClassificationResult | None = None,
        lub_state: LubricationState | None = None,
        vib_state: VibrationState | None = None,
        comb_state: CombustionStabilityState | None = None,
    ) -> list[EvidenceItem]:
        """Extract and rank structured evidence items from available L2/L3 diagnostic outputs."""
        self._check_boundary(residual_state, anomaly_result, fault_result, lub_state, vib_state, comb_state)

        items: list[EvidenceItem] = []
        ts = datetime.now(timezone.utc)

        # 1. Residual evidence
        if residual_state:
            ts = residual_state.timestamp
            for key, ptv in residual_state.residuals.items():
                if ptv.valid and ptv.value is not None and not math.isnan(ptv.value):
                    dev = abs(ptv.value)
                    if dev > 1e-4:
                        items.append(
                            EvidenceItem(
                                source="Module 11 Residual Engine",
                                metric=key,
                                observed_value=None,
                                reference_value=0.0,
                                deviation=ptv.value,
                                status="ELEVATED_RESIDUAL" if dev > 50.0 else "NORMAL",
                                contribution=min(1.0, dev / 100.0),
                                quality=ptv.quality,
                                provenance=ptv.provenance,
                                timestamp=ts,
                                explanation=f"Residual deviation of {ptv.value:.2f} observed for {key}.",
                            )
                        )

        # 2. Anomaly evidence
        if anomaly_result:
            ts = anomaly_result.timestamp
            items.append(
                EvidenceItem(
                    source="Module 13 Anomaly Detector",
                    metric="anomaly_score",
                    observed_value=anomaly_result.anomaly_score,
                    reference_value=anomaly_result.threshold,
                    deviation=max(0.0, anomaly_result.anomaly_score - anomaly_result.threshold),
                    status="ANOMALY_DETECTED" if anomaly_result.is_anomaly else "NORMAL",
                    contribution=min(1.0, anomaly_result.anomaly_score),
                    quality=anomaly_result.quality,
                    provenance=anomaly_result.provenance,
                    timestamp=ts,
                    explanation=f"Anomaly score {anomaly_result.anomaly_score:.3f} (threshold {anomaly_result.threshold:.2f}).",
                )
            )

        # 3. Fault Classification evidence
        if fault_result and fault_result.predicted_class != FaultClass.NOMINAL:
            ts = fault_result.timestamp
            conf = fault_result.confidence or 1.0
            items.append(
                EvidenceItem(
                    source="Module 13 Fault Classifier",
                    metric="fault_classification",
                    observed_value=float(fault_result.class_id),
                    reference_value=0.0,
                    deviation=float(fault_result.class_id),
                    status=fault_result.class_name,
                    contribution=conf,
                    quality=fault_result.quality,
                    provenance=fault_result.provenance,
                    timestamp=ts,
                    explanation=f"Classified fault '{fault_result.class_name}' with confidence {conf:.2f}.",
                )
            )

        # 4. Lubrication evidence
        if lub_state:
            ts = lub_state.timestamp
            if lub_state.oil_pressure_pa.valid and lub_state.oil_pressure_pa.value is not None:
                p_bar = lub_state.oil_pressure_pa.value / 100000.0
                if p_bar < 2.5:
                    items.append(
                        EvidenceItem(
                            source="Module 8 Lubrication Model",
                            metric="oil_pressure_bar",
                            observed_value=p_bar,
                            reference_value=4.0,
                            deviation=4.0 - p_bar,
                            status="LOW_PRESSURE",
                            contribution=min(1.0, max(0.0, (2.5 - p_bar) / 1.5)),
                            quality=lub_state.oil_pressure_pa.quality,
                            provenance=lub_state.provenance,
                            timestamp=ts,
                            explanation=f"Low oil pressure {p_bar:.2f} bar below 2.5 bar margin.",
                        )
                    )

        # Sort evidence items by contribution magnitude (highest contribution first)
        items.sort(key=lambda x: x.contribution, reverse=True)
        return items

    def explain_fault(
        self,
        fault_result: FaultClassificationResult | None,
        anomaly_result: AnomalyResult | None = None,
        residual_state: ResidualState | None = None,
    ) -> Explanation:
        """Generate human-readable explanation of fault classification results."""
        self._check_boundary(fault_result, anomaly_result, residual_state)

        ts = fault_result.timestamp if fault_result else datetime.now(timezone.utc)
        evidence_items = self.extract_evidence_items(residual_state=residual_state, anomaly_result=anomaly_result, fault_result=fault_result)

        if not fault_result or fault_result.status != InferenceStatus.SUCCESS:
            status_desc = str(fault_result.status) if fault_result else "UNKNOWN"
            return Explanation(
                explanation_id=f"exp_fault_{int(ts.timestamp())}",
                finding="No definitive fault detected (Classifier Status: " + status_desc + ").",
                observation=f"Fault classifier status is {status_desc}.",
                interpretation="ML fault classification unavailable or unconfirmed. System operating under baseline rules.",
                evidence=evidence_items,
                uncertainty="Lack of validated ML inference result or active model artifact.",
                limitation="Rule-based diagnostic fallback active; ML classifier not loaded.",
                provenance=Provenance.DERIVED,
                quality=0.5,
                timestamp=ts,
            )

        f_class = fault_result.predicted_class
        conf_str = f"{fault_result.confidence:.2f}" if fault_result.confidence is not None else "N/A"

        if f_class == FaultClass.NOMINAL:
            return Explanation(
                explanation_id=f"exp_fault_{int(ts.timestamp())}",
                finding="Engine operating in NOMINAL condition.",
                observation="All multi-signal physics features and residuals remain within healthy baseline bounds.",
                interpretation="No abnormal combustion, valve leak, boost drop, or lubrication fault patterns detected.",
                evidence=evidence_items,
                uncertainty="Residual noise floor ±5%.",
                limitation="Diagnostic inference constrained to the prototype fault taxonomy (docs/FAULT_TAXONOMY.md).",
                provenance=fault_result.provenance,
                quality=fault_result.quality,
                timestamp=ts,
            )

        state_label = "DETECTED" if (fault_result.confidence or 1.0) >= 0.7 else "POSSIBLE"

        return Explanation(
            explanation_id=f"exp_fault_{int(ts.timestamp())}",
            finding=f"Fault {state_label}: {fault_result.class_name} (Confidence: {conf_str}).",
            observation=f"Fault classifier identified pattern matching {fault_result.class_name}.",
            interpretation=f"Multi-signal residual vectors exhibit characteristic signatures of {fault_result.class_name}.",
            evidence=evidence_items,
            uncertainty="Statistical classifier prediction subject to sensor noise boundaries.",
            limitation="Prototype classifier recommendation only; requires physical ground inspection.",
            provenance=fault_result.provenance,
            quality=fault_result.quality,
            timestamp=ts,
        )

    def explain_health(self, health_state: HealthState | None) -> Explanation:
        """Generate human-readable explanation of HealthIndex and degradation supervision."""
        self._check_boundary(health_state)

        if not health_state or not health_state.health_index.valid or health_state.health_index.value is None:
            ts = datetime.now(timezone.utc)
            return Explanation(
                explanation_id=f"exp_health_{int(ts.timestamp())}",
                finding="Health Index calculation unavailable.",
                observation="Health state telemetry or inputs missing/invalid.",
                interpretation="Cannot establish health index baseline.",
                evidence=[],
                uncertainty="High uncertainty due to invalid inputs.",
                limitation="Health Supervision Engine requires valid residual or diagnostic inputs.",
                provenance=Provenance.DERIVED,
                quality=0.0,
                timestamp=ts,
            )

        ts = health_state.timestamp
        hi_val = health_state.health_index.value
        deg_state = health_state.degradation_state.name

        comp_reasons = []
        for c_name, c_score in health_state.component_health.items():
            if c_score < 0.9:
                comp_reasons.append(f"{c_name} (score {c_score:.2f})")

        reasons_str = f" Degraded components: {', '.join(comp_reasons)}." if comp_reasons else " Component scores healthy."

        return Explanation(
            explanation_id=f"exp_health_{int(ts.timestamp())}",
            finding=f"Engine Health Index is {hi_val:.2f} ({deg_state} state, trend: {health_state.trend}).",
            observation=f"Composite health score is {hi_val * 100.0:.1f}%.{reasons_str}",
            interpretation=f"Supervisory engine evaluated physical residuals and component degradation state as {deg_state}.",
            evidence=[],
            uncertainty="Health Index is a relative diagnostic index, NOT a remaining life percentage.",
            limitation="Non-certified analytical health supervision score.",
            provenance=health_state.provenance,
            quality=health_state.health_index.quality,
            timestamp=ts,
        )

    def explain_rul(self, rul_state: RULState | None) -> Explanation:
        """Generate human-readable explanation of RUL estimates and uncertainty boundaries."""
        self._check_boundary(rul_state)

        if not rul_state or rul_state.hours_remaining <= 0.0 or rul_state.status != InferenceStatus.SUCCESS:
            ts = rul_state.timestamp if rul_state else datetime.now(timezone.utc)
            status_name = str(rul_state.status) if rul_state else "NO_DATA"
            return Explanation(
                explanation_id=f"exp_rul_{int(ts.timestamp())}",
                finding="Remaining Useful Life (RUL) estimate UNAVAILABLE.",
                observation=f"RUL estimator status: {status_name}.",
                interpretation="Insufficient historical health degradation trend to compute reliable RUL trajectory.",
                evidence=[],
                uncertainty="RUL trend calculation requires monotonic multi-point health history.",
                limitation="RUL unavailable during initial baseline or steady nominal operation.",
                provenance=Provenance.DERIVED,
                quality=0.0,
                timestamp=ts,
            )

        ts = rul_state.timestamp
        r_hrs = rul_state.hours_remaining
        lower = f"{rul_state.lower_bound_hours:.1f}" if rul_state.lower_bound_hours is not None else "N/A"
        upper = f"{rul_state.upper_bound_hours:.1f}" if rul_state.upper_bound_hours is not None else "N/A"

        return Explanation(
            explanation_id=f"exp_rul_{int(ts.timestamp())}",
            finding=f"Estimated Remaining Useful Life: {r_hrs:.1f} hours (Bounds: [{lower}, {upper}] hrs).",
            observation=f"Degradation trend '{rul_state.trend}' projects RUL of {r_hrs:.1f} {rul_state.unit}.",
            interpretation=f"Based on assumption '{rul_state.operating_assumption}' and historical health progression.",
            evidence=[],
            uncertainty=f"Uncertainty bounds [{lower} - {upper}] hours reflect trend extrapolation tolerance.",
            limitation="Analytical estimate assuming constant future operating profiles; not a certified maintenance limit.",
            provenance=rul_state.provenance,
            quality=rul_state.quality,
            timestamp=ts,
        )

    def explain_mission_risk(self, mission_state: Any) -> Explanation:
        """Generate human-readable explanation of Mission Risk scores and phase context."""
        self._check_boundary(mission_state)

        if not mission_state:
            ts = datetime.now(timezone.utc)
            return Explanation(
                explanation_id=f"exp_risk_{int(ts.timestamp())}",
                finding="Mission Risk evaluation unavailable.",
                observation="Mission state missing.",
                interpretation="Cannot compute mission risk.",
                evidence=[],
                uncertainty="High",
                limitation="No data",
                provenance=Provenance.DERIVED,
                quality=0.0,
                timestamp=ts,
            )

        ts = getattr(mission_state, "timestamp", datetime.now(timezone.utc))
        risk_score = getattr(mission_state, "risk_score", 0.0)
        risk_level = getattr(mission_state, "risk_level", "LOW")
        phase = getattr(mission_state, "flight_phase", "UNKNOWN")

        return Explanation(
            explanation_id=f"exp_risk_{int(ts.timestamp())}",
            finding=f"Mission Risk is {risk_level} (Score: {risk_score:.3f}) during phase {phase}.",
            observation=f"Analytical risk engine derived relative risk score of {risk_score:.3f}.",
            interpretation=f"Evaluated during flight phase {phase} under risk level {risk_level}.",
            evidence=[],
            uncertainty="Risk score is a relative analytical index, NOT an absolute probability of failure.",
            limitation="Decision-support advisory score; no autonomous flight control or mission abort command.",
            provenance=getattr(mission_state, "provenance", Provenance.DERIVED),
            quality=getattr(mission_state, "quality", 1.0),
            timestamp=ts,
        )


class AdvisoryEngine:
    """Deterministic advisory generation engine producing decision-support guidance."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._explainer = ExplainabilityEngine(settings=self._settings)

    def _check_boundary(self, *inputs: Any) -> None:
        for arg in inputs:
            if isinstance(arg, RawSignalRecord):
                raise TypeError("STRICT BOUNDARY VIOLATION: AdvisoryEngine MUST NOT consume RawSignalRecord directly.")

    def generate_advisories(
        self,
        health_state: HealthState | None = None,
        fault_result: FaultClassificationResult | None = None,
        anomaly_result: AnomalyResult | None = None,
        rul_state: RULState | None = None,
        mission_state: Any | None = None,
        residual_state: ResidualState | None = None,
        lub_state: LubricationState | None = None,
        elec_state: ElectricalState | None = None,
    ) -> list[Advisory]:
        """Generate deterministic decision-support advisories based on diagnostic outputs."""
        self._check_boundary(health_state, fault_result, anomaly_result, rul_state, mission_state, residual_state,
                             lub_state, elec_state)

        advisories: list[Advisory] = []
        ts = datetime.now(timezone.utc)

        evidence_items = self._explainer.extract_evidence_items(
            residual_state=residual_state,
            anomaly_result=anomaly_result,
            fault_result=fault_result,
            lub_state=lub_state,
        )

        # 1. Data Quality Check
        if health_state and health_state.health_index.quality < 0.7:
            advisories.append(
                Advisory(
                    advisory_id=f"adv_qual_{int(ts.timestamp())}",
                    category=AdvisoryCategory.DATA_QUALITY_CHECK,
                    priority=AdvisoryPriority.MEDIUM,
                    title="Telemetry Data Quality Degraded",
                    message=f"Signal quality score ({health_state.health_index.quality:.2f}) is below standard threshold 0.70.",
                    evidence=evidence_items,
                    subsystem="SENSORS",
                    confidence=0.9,
                    provenance=health_state.provenance,
                    timestamp=ts,
                    limitations="Verify sensor connections, wiring, and ADC calibration.",
                )
            )

        # 2. Fault Classification Advisory
        if fault_result and fault_result.predicted_class != FaultClass.NOMINAL and fault_result.status == InferenceStatus.SUCCESS:
            priority = AdvisoryPriority.CRITICAL if fault_result.predicted_class in (FaultClass.MISFIRE, FaultClass.DETONATION_KNOCK) else AdvisoryPriority.HIGH
            advisories.append(
                Advisory(
                    advisory_id=f"adv_fault_{int(ts.timestamp())}",
                    category=AdvisoryCategory.INSPECT,
                    priority=priority,
                    title=f"Inspect Engine: {fault_result.class_name} Detected",
                    message=f"Fault classifier detected {fault_result.class_name} pattern.",
                    evidence=evidence_items,
                    subsystem="CYLINDER_COMBUSTION" if "MISFIRE" in fault_result.class_name else "ENGINE_MECHANICAL",
                    confidence=fault_result.confidence or 0.8,
                    provenance=fault_result.provenance,
                    timestamp=ts,
                    limitations="Prototype diagnostic classifier finding; perform manual physical inspection according to maintenance manual.",
                )
            )

        # 2b. US-801 class templates (classes 9-13) and condition flags (Prompt 15)
        if fault_result is not None:
            class_adv = fault_class_advisory(fault_result)
            if class_adv is not None:
                advisories.append(class_adv)
            advisories.extend(condition_flag_advisories(fault_result))

        # 3. Anomaly Advisory
        if anomaly_result and anomaly_result.is_anomaly:
            advisories.append(
                Advisory(
                    advisory_id=f"adv_anom_{int(ts.timestamp())}",
                    category=AdvisoryCategory.INVESTIGATE,
                    priority=AdvisoryPriority.MEDIUM,
                    title="Investigate Engine Signal Anomaly",
                    message=f"Multi-feature anomaly score ({anomaly_result.anomaly_score:.2f}) exceeded threshold ({anomaly_result.threshold:.2f}).",
                    evidence=evidence_items,
                    subsystem="DIGITAL_TWIN",
                    confidence=0.85,
                    provenance=anomaly_result.provenance,
                    timestamp=ts,
                )
            )

        # 4. Health Index Degradation Advisory
        if health_state and health_state.health_index.valid and health_state.health_index.value is not None:
            hi_val = health_state.health_index.value
            if hi_val < 0.8:
                priority = AdvisoryPriority.HIGH if hi_val < 0.6 else AdvisoryPriority.MEDIUM
                advisories.append(
                    Advisory(
                        advisory_id=f"adv_hi_{int(ts.timestamp())}",
                        category=AdvisoryCategory.MAINTENANCE_REVIEW,
                        priority=priority,
                        title=f"Health Index Degraded ({health_state.degradation_state.name})",
                        message=f"Overall Health Index degraded to {hi_val * 100.0:.1f}% (Trend: {health_state.trend}).",
                        evidence=evidence_items,
                        subsystem="SUPERVISION",
                        confidence=0.9,
                        provenance=health_state.provenance,
                        timestamp=ts,
                    )
                )

        # 5. Electrical (charging system and battery)
        if elec_state is not None:
            advisories.extend(electrical_advisories(elec_state))

        # 6. Nominal / Default Advisory if no issues
        if not advisories:
            advisories.append(
                Advisory(
                    advisory_id=f"adv_nom_{int(ts.timestamp())}",
                    category=AdvisoryCategory.CONTINUE_MONITORING,
                    priority=AdvisoryPriority.INFORMATION,
                    title="Normal Engine Operation",
                    message="All physics residuals, health indices, and diagnostic signals remain within healthy limits.",
                    evidence=evidence_items,
                    subsystem="GENERAL",
                    confidence=1.0,
                    provenance=Provenance.DERIVED,
                    timestamp=ts,
                )
            )

        # Sort advisories by priority order
        prio_order = {
            AdvisoryPriority.CRITICAL: 4,
            AdvisoryPriority.HIGH: 3,
            AdvisoryPriority.MEDIUM: 2,
            AdvisoryPriority.LOW: 1,
            AdvisoryPriority.INFORMATION: 0,
        }
        advisories.sort(key=lambda a: prio_order.get(a.priority, 0), reverse=True)
        return advisories


_ELEC_LIMITATION = (
    "Expectations use placeholder charging-system values (regulator setpoint, cut-in, battery OCV and "
    "internal resistance) pending verification against the Rotax 915 iS Operators Manual; see docs/OPEN_ITEMS.md."
)


def _electrical_advisory(key: str, status: DiagnosticStatus, title: str, message: str, metric: str,
                         observed: float, reference: float, quality: float, ts: datetime) -> Advisory:
    priority = AdvisoryPriority.HIGH if status == DiagnosticStatus.CRITICAL else AdvisoryPriority.MEDIUM
    item = EvidenceItem(source="L2 Electrical Model", metric=metric, observed_value=observed,
                        reference_value=reference, deviation=observed - reference, status=status.value,
                        contribution=1.0 if status == DiagnosticStatus.CRITICAL else 0.5, quality=quality,
                        provenance=Provenance.DERIVED, timestamp=ts, explanation=message)
    return Advisory(advisory_id=f"adv_elec_{key}_{int(ts.timestamp())}", category=AdvisoryCategory.INSPECT,
                    priority=priority, title=title, message=message, evidence=[item], subsystem="ELECTRICAL",
                    confidence=min(0.9, quality), provenance=Provenance.DERIVED, timestamp=ts,
                    limitations=_ELEC_LIMITATION)


def electrical_advisories(elec_state: ElectricalState) -> list[Advisory]:
    """Advisories for the charging system and battery (SRD-FUN-142: cause and
    action, worded as consistent-with, never as a diagnosis)."""
    out: list[Advisory] = []
    ts = elec_state.timestamp
    flagged = (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL)
    cfg = get_settings().electrical

    res, exp = elec_state.charging_residual_median_v, elec_state.expected_bus_voltage_v
    if elec_state.charging_status in flagged and res.valid and exp.valid:
        d = res.value
        if elec_state.charging_regime == "REGULATED" and d < 0.0:
            msg = (f"Charging voltage {abs(d):.1f} V below expectation for this rpm, consistent with regulator or "
                   f"alternator drive failure. Inspect the regulator and the alternator drive.")
        elif elec_state.charging_regime == "REGULATED":
            msg = (f"Charging voltage {d:.1f} V above expectation for this rpm, consistent with a regulator "
                   f"over-voltage fault. Inspect the regulator; sustained over-voltage can damage the battery "
                   f"and avionics.")
        elif d < 0.0:
            msg = (f"Bus voltage {abs(d):.1f} V below the expected battery voltage with the alternator below "
                   f"cut-in, consistent with a discharged or degraded battery. Check battery state of charge "
                   f"and condition.")
        else:
            msg = (f"Bus voltage {d:.1f} V above the expected battery voltage with the alternator below cut-in, "
                   f"consistent with an external charging source or a bus-voltage sensing fault. Check the "
                   f"voltage sensing divider.")
        out.append(_electrical_advisory("charging", elec_state.charging_status, "Charging Voltage Deviation", msg,
                                        "charging_residual_v", exp.value + d, exp.value, res.quality, ts))

    rip = elec_state.voltage_ripple_pct
    if elec_state.ripple_status in flagged and rip.valid:
        healthy = cfg.ripple_healthy_pct
        msg = (f"Bus voltage ripple {rip.value:.2f} % ({rip.value / healthy:.1f}x the healthy level), consistent "
               f"with an open rectifier diode or a failing regulator. Inspect the rectifier and regulator unit.")
        if elec_state.ripple_order.valid:
            msg += (f" The ripple is locked to generator speed (order {elec_state.ripple_order.value:.0f}), which "
                    f"points to the charging circuit rather than an electrical load.")
        out.append(_electrical_advisory("ripple", elec_state.ripple_status, "Bus Voltage Ripple Elevated", msg,
                                        "voltage_ripple_pct", rip.value, healthy, rip.quality, ts))

    r = elec_state.battery_resistance_mohm
    if elec_state.battery_status in flagged and r.valid:
        nominal = cfg.battery_r_int_nominal_mohm
        msg = (f"Battery internal resistance {r.value:.0f} mOhm ({r.value / nominal:.1f}x nominal), consistent "
               f"with battery ageing, sulphation or a high-resistance connection. Perform a battery capacity "
               f"test and check the terminal connections.")
        out.append(_electrical_advisory("battery", elec_state.battery_status, "Battery Internal Resistance High",
                                        msg, "battery_resistance_mohm", r.value, nominal, r.quality, ts))
    return out


class DiagnosticQueryService:
    """Service providing operator-facing diagnostic question answering."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._explainer = ExplainabilityEngine(settings=self._settings)
        self._adviser = AdvisoryEngine(settings=self._settings)

    def _check_boundary(self, *inputs: Any) -> None:
        for arg in inputs:
            if isinstance(arg, RawSignalRecord):
                raise TypeError("STRICT BOUNDARY VIOLATION: DiagnosticQueryService MUST NOT consume RawSignalRecord directly.")

    def answer_query(
        self,
        question_type: QuestionType | str,
        health_state: HealthState | None = None,
        fault_result: FaultClassificationResult | None = None,
        anomaly_result: AnomalyResult | None = None,
        rul_state: RULState | None = None,
        mission_state: Any | None = None,
        previous_health_state: HealthState | None = None,
        comparison_result: Any | None = None,
    ) -> DiagnosticAnswer:
        """Answer structured operator diagnostic questions using available L2/L3 outputs."""
        self._check_boundary(health_state, fault_result, anomaly_result, rul_state, mission_state, previous_health_state, comparison_result)

        q_type_str = question_type.value if isinstance(question_type, QuestionType) else str(question_type).upper()
        ts = datetime.now(timezone.utc)

        evidence_items = self._explainer.extract_evidence_items(
            anomaly_result=anomaly_result,
            fault_result=fault_result,
        )

        if q_type_str == QuestionType.HEALTH_STATUS.value:
            exp = self._explainer.explain_health(health_state)
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=exp.finding,
                evidence=exp.evidence,
                limitations=exp.limitation,
                quality=exp.quality,
                provenance=exp.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.HEALTH_REASON.value:
            exp = self._explainer.explain_health(health_state)
            ans = f"{exp.finding} {exp.observation} {exp.interpretation}"
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=ans,
                evidence=evidence_items,
                limitations=exp.limitation,
                quality=exp.quality,
                provenance=exp.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.FAULT_STATUS.value:
            exp = self._explainer.explain_fault(fault_result, anomaly_result)
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=exp.finding,
                evidence=exp.evidence,
                limitations=exp.limitation,
                quality=exp.quality,
                provenance=exp.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.FAULT_EVIDENCE.value:
            exp = self._explainer.explain_fault(fault_result, anomaly_result)
            ans = f"{exp.observation} {exp.interpretation}"
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=ans,
                evidence=evidence_items,
                limitations=exp.limitation,
                quality=exp.quality,
                provenance=exp.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.RUL_ESTIMATE.value:
            exp = self._explainer.explain_rul(rul_state)
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=exp.finding,
                evidence=[],
                limitations=exp.limitation,
                quality=exp.quality,
                provenance=exp.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.MISSION_RISK.value:
            exp = self._explainer.explain_mission_risk(mission_state)
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=exp.finding,
                evidence=[],
                limitations=exp.limitation,
                quality=exp.quality,
                provenance=exp.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.WHAT_CHANGED.value:
            if not health_state or not previous_health_state:
                return DiagnosticAnswer(
                    question_type=q_type_str,
                    answer="INSUFFICIENT_HISTORY: Historical state comparison requires both current and previous health states.",
                    evidence=[],
                    limitations="Insufficient historical telemetry window.",
                    quality=0.0,
                    provenance=Provenance.DERIVED,
                    timestamp=ts,
                )

            curr_hi = health_state.health_index.value if (health_state.health_index.valid and health_state.health_index.value is not None) else 0.0
            prev_hi = previous_health_state.health_index.value if (previous_health_state.health_index.valid and previous_health_state.health_index.value is not None) else 0.0
            hi_delta = curr_hi - prev_hi

            ans = (
                f"Health Index changed from {prev_hi:.2f} to {curr_hi:.2f} (Delta: {hi_delta:+.2f}). "
                f"Degradation state transition: {previous_health_state.degradation_state.name} → {health_state.degradation_state.name}."
            )

            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=ans,
                evidence=evidence_items,
                limitations="Historical comparison limited to available snapshot memory.",
                quality=health_state.health_index.quality,
                provenance=health_state.provenance,
                timestamp=ts,
            )

        elif q_type_str == QuestionType.RECOMMENDED_ACTION.value:
            advisories = self._adviser.generate_advisories(
                health_state=health_state,
                fault_result=fault_result,
                anomaly_result=anomaly_result,
                rul_state=rul_state,
                mission_state=mission_state,
            )
            top_adv = advisories[0]
            ans = f"[{top_adv.category.value}] Priority {top_adv.priority.value}: {top_adv.title}. {top_adv.message}"
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=ans,
                evidence=top_adv.evidence,
                limitations=top_adv.limitations,
                quality=top_adv.confidence,
                provenance=top_adv.provenance,
                timestamp=ts,
            )

        else:
            return DiagnosticAnswer(
                question_type=q_type_str,
                answer=f"Unsupported question type '{q_type_str}'. Supported queries: {[q.value for q in QuestionType]}",
                evidence=[],
                limitations="Unsupported query category.",
                quality=0.0,
                provenance=Provenance.DERIVED,
                timestamp=ts,
            )


_OVERHEAT_LIMITATION = (
    "Projection from the slope of (measured - expected) over the last few minutes; alarm limits and the "
    "healthy expectations are placeholders pending verification against the Rotax 915 iS Operators Manual "
    "(docs/OPEN_ITEMS.md OI-5, OI-22)."
)


def overheat_advisories(overheat_state) -> list[Advisory]:
    """Overheating-trend advisories (Prompt 14): one per channel whose projected
    time to its alarm limit is within the alert horizon. Worded as a projection
    with its range, never as a certainty (SRD-FUN-142)."""
    from src.l2_digital_twin.overheat_trend import overheat_message  # L2 wording, no L2 state

    out: list[Advisory] = []
    if overheat_state is None:
        return out
    ts = overheat_state.timestamp
    for name, trend in overheat_state.channels.items():
        if trend.status not in (DiagnosticStatus.WARNING, DiagnosticStatus.CRITICAL):
            continue
        msg = (overheat_message(trend) + " Consistent with reduced cooling for this operating point; "
               "reduce power and check the cooling system.")
        cur = trend.current_c.value if trend.current_c.valid else 0.0
        item = EvidenceItem(source="L2 Overheat Trend", metric=f"{name}_time_to_limit_s",
                            observed_value=cur, reference_value=trend.limit_c or 0.0,
                            deviation=cur - (trend.limit_c or 0.0), status=trend.status.value,
                            contribution=1.0 if trend.status == DiagnosticStatus.CRITICAL else 0.5,
                            quality=trend.time_to_limit_s.quality, provenance=Provenance.DERIVED, timestamp=ts,
                            explanation=msg)
        out.append(Advisory(
            advisory_id=f"adv_overheat_{name}_{int(ts.timestamp())}", category=AdvisoryCategory.INSPECT,
            priority=AdvisoryPriority.HIGH if trend.status == DiagnosticStatus.CRITICAL else AdvisoryPriority.MEDIUM,
            title="Overheating Trend", message=msg, evidence=[item], subsystem="COOLING",
            confidence=0.8, provenance=Provenance.DERIVED, timestamp=ts, limitations=_OVERHEAT_LIMITATION))
    return out


# ---------------------------------------------------------------------------
# US-801 templates (Prompt 15): physical cause, evidence parameters, action.
# Worded "consistent with" (SRD-FUN-142); the evidence values come from the
# classifier result, never from the template.
# ---------------------------------------------------------------------------

FAULT_ADVISORY_TEMPLATES: dict[FaultClass, dict[str, Any]] = {
    FaultClass.INJECTOR_FAULT: {
        "cause": "an injector delivering a different fuel quantity than commanded (clogged: lean and hot; "
                 "leaking or dripping: rich and cold)",
        "evidence": ("cylinders", "injector_flow_ratio"),
        "action": "Inspect and flow-test the injector of the identified cylinder; check its connector and spray pattern.",
        "subsystem": "FUEL_INJECTION"},
    FaultClass.FUEL_SYSTEM_FAULT: {
        "cause": "fuel pump delivery or pressure regulation insufficient for the demand, so all cylinders run lean",
        "evidence": ("rail_pressure_residual_kpa", "fuel_delivery_ratio"),
        "action": "Check the fuel pumps, filter and pressure regulator; avoid high power until rail pressure is restored.",
        "subsystem": "FUEL_SYSTEM"},
    FaultClass.IMBALANCE: {
        "cause": "a rotating imbalance at propeller-shaft speed (propeller mass or pitch imbalance, spinner, ice)",
        "evidence": ("prop_1x_hz", "prop_1x_velocity_fraction_lateral", "prop_1x_lateral_vertical_ratio"),
        "action": "Inspect the propeller and spinner for damage or ice; perform a dynamic propeller balance.",
        "subsystem": "PROPELLER"},
    FaultClass.CHARGING_FAULT: {
        "cause": "the regulator or alternator (stator, rectifier diode, drive) not holding the bus voltage",
        "evidence": ("charging_residual_v", "voltage_ripple_pct", "charging_status", "ripple_status"),
        "action": "Inspect the regulator/rectifier and the alternator drive; check the bus voltage under load.",
        "subsystem": "ELECTRICAL"},
    FaultClass.BATTERY_DEGRADATION: {
        "cause": "battery internal resistance above nominal (ageing, sulphation, low electrolyte)",
        "evidence": ("battery_resistance_mohm", "battery_status"),
        "action": "Load-test the battery and replace it if the test confirms the loss of capacity.",
        "subsystem": "ELECTRICAL"},
}

CONDITION_FLAG_TEMPLATES: dict[str, dict[str, Any]] = {
    "overheating_trend": {
        "cause": "a temperature rising faster than the operating point explains, projected to reach its alarm limit",
        "evidence": ("time_to_limit_s", "horizon_s"),
        "action": "Reduce power, increase airspeed if possible, and check the cooling system.",
        "subsystem": "COOLING"},
    "combustion_instability": {
        "cause": "cycle-to-cycle combustion variability (misfire, knock, mixture or ignition problem)",
        "evidence": ("csi", "csi_band"),
        "action": "Check ignition and injection; review the misfire and knock evidence.",
        "subsystem": "CYLINDER_COMBUSTION"},
    "lubrication_degraded": {
        "cause": "oil pressure, viscosity or temperature outside the healthy envelope (degradation, dilution, "
                 "leak, pump wear)",
        "evidence": ("lhi", "lhi_band"),
        "action": "Check the oil level and pressure, take an oil sample and inspect for leaks.",
        "subsystem": "LUBRICATION"},
}


def _evidence_text(values: dict[str, Any], keys: tuple[str, ...]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3g}"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {fmt(x)}" for k, x in v.items()) + "}"
        return "n/a" if v is None else str(v)
    return "; ".join(f"{k} = {fmt(values.get(k))}" for k in keys)


def fault_class_advisory(fault_result: FaultClassificationResult) -> Advisory | None:
    """US-801 advisory for a class with a template (classes 9-13)."""
    tpl = FAULT_ADVISORY_TEMPLATES.get(fault_result.predicted_class)
    if tpl is None or fault_result.status != InferenceStatus.SUCCESS:
        return None
    ev = fault_result.evidence or {}
    source = "rule-based diagnosis (is_ml=False)" if not fault_result.is_ml else "ML classifier"
    msg = (f"{fault_result.class_name}: consistent with {tpl['cause']}. Evidence: "
           f"{_evidence_text(ev, tpl['evidence'])}. Action: {tpl['action']}")
    ts = fault_result.timestamp
    return Advisory(
        advisory_id=f"adv_class_{fault_result.class_name.lower()}_{int(ts.timestamp())}",
        category=AdvisoryCategory.INSPECT, priority=AdvisoryPriority.HIGH,
        title=fault_result.class_name.replace("_", " ").title(), message=msg, evidence=[],
        subsystem=tpl["subsystem"],
        confidence=fault_result.confidence if fault_result.confidence is not None else 0.5,
        provenance=fault_result.provenance, timestamp=ts,
        limitations=f"From the {source}; placeholder thresholds pending verification (docs/FAULT_TAXONOMY.md).")


def condition_flag_advisories(fault_result: FaultClassificationResult) -> list[Advisory]:
    """US-801 advisories for the condition flags that are set (True)."""
    out: list[Advisory] = []
    ts = fault_result.timestamp
    for flag, tpl in CONDITION_FLAG_TEMPLATES.items():
        if fault_result.condition_flags.get(flag) is not True:
            continue
        ev = fault_result.condition_evidence.get(flag)
        ev = ev if isinstance(ev, dict) else {}
        msg = (f"Condition {flag}: consistent with {tpl['cause']}. Evidence: {_evidence_text(ev, tpl['evidence'])}. "
               f"Action: {tpl['action']}")
        out.append(Advisory(
            advisory_id=f"adv_flag_{flag}_{int(ts.timestamp())}", category=AdvisoryCategory.MONITOR,
            priority=AdvisoryPriority.MEDIUM, title=f"Condition: {flag.replace('_', ' ')}", message=msg,
            evidence=[], subsystem=tpl["subsystem"], confidence=0.5, provenance=Provenance.DERIVED,
            timestamp=ts, limitations="Condition flag (symptom), independent of the fault class."))
    return out
