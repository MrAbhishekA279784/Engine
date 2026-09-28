"""
API Transport Schemas — Pydantic Request, Response, and Event Models (Module 20).

Contains clean serialization DTOs for REST endpoints and WebSocket events.
Keeps transport concerns decoupled from domain models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from src.core.provenance import (
    DiagnosticStatus,
    FaultClass,
    FlightPhase,
    InferenceStatus,
    Provenance,
)
from src.core.schemas import SignalQuality

T = TypeVar("T")


class APIErrorResponse(BaseModel):
    """Standardized API error response envelope."""

    error_code: str = Field(description="Machine-readable error category code")
    message: str = Field(description="Human-readable error description")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional context or validation details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthCheckResponse(BaseModel):
    """REST API service availability health response."""

    status: str = Field(default="ok", description="Service status (ok / degraded)")
    service_name: str = Field(default="Piston Engine Digital Twin API")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    """Overall system operational status response."""

    service_status: str = Field(description="API service health status")
    telemetry_pipeline_status: str = Field(description="L1 Ingestion pipeline status")
    digital_twin_status: str = Field(description="L2 Digital Twin status")
    ml_supervision_status: str = Field(description="L3 ML Supervision status")
    active_connections: int = Field(default=0, description="Active WebSocket clients")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryIngestRequest(BaseModel):
    """Request DTO for raw telemetry ingestion."""

    sequence_number: int = Field(ge=0)
    source_type: Provenance = Field(default=Provenance.SIMULATED)
    egt_cyl1_hot_uv: float | None
    egt_cyl2_hot_uv: float | None
    egt_cyl3_hot_uv: float | None
    egt_cyl4_hot_uv: float | None
    egt_cold_c: float | None
    cht_hot_uv: float | None
    cht_cold_c: float | None
    oil_rtd_ohms: float | None
    oil_p_counts: int | None = Field(ge=0, le=4095)
    map_counts: int | None = Field(ge=0, le=4095)
    adc_vref_counts: int | None = Field(ge=0, le=4095)
    crank_period_us: float | None = Field(gt=0.0)
    fuel_pulse_hz: float | None = Field(ge=0.0)
    accel_counts_xyz: tuple[int, int, int] | None
    ambient_temp_c: float | None
    ambient_press_pa: float | None
    accel_burst_counts_x: list[int] = Field(default_factory=list)
    accel_burst_counts_y: list[int] = Field(default_factory=list)
    accel_burst_counts_z: list[int] = Field(default_factory=list)
    accel_burst_fs_hz: float = Field(default=0.0, ge=0.0)
    crank_period_burst_us: list[float] = Field(default_factory=list)
    bus_v_counts: int | None = None
    alt_i_counts: int | None = None
    batt_i_counts: int | None = None
    bus_v_burst_counts: list[int] | None = None
    bus_v_burst_fs_hz: float | None = None
    inj_pw_us: list[float] | None = None
    inj_soi_delay_us: list[float] | None = None
    ign_delay_us: list[float] | None = None
    fuel_press_counts: int | None = None
    coolant_ntc_ohms: float | None = None
    timestamp: datetime | None = None
    integrity_hash: str | None = None
    signature: str | None = None


class TelemetryIngestResponse(BaseModel):
    """Response DTO for raw telemetry ingestion."""

    accepted: bool
    sequence_number: int
    rejection_reason: str = ""
    security_verified: bool = False
    invalid_channels: list[str] = Field(default_factory=list)
    quality: SignalQuality
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EngineHealthResponse(BaseModel):
    """Engine Health Index & degradation state response DTO."""

    # From the L3 HealthSupervision output of the running pipeline (OI-20).
    # health_index is None (with health_index_reason) when the evidence is
    # insufficient. component_health lists every configured component; a
    # component without valid data is None and its reason is in
    # component_health_reasons.
    health_index: float | None
    health_index_reason: str | None = None
    degradation_state: str
    trend: str
    health_rate_per_s: float | None = None
    component_health: dict[str, float | None]
    component_health_reasons: dict[str, str] = Field(default_factory=dict)
    coverage: float | None = None
    records_in_window: int = 0
    # Versioning (Prompt 17)
    baseline_version: str | None = None
    adapted_baseline: bool = False
    model_version: str | None = None
    hours_since_baseline: float | None = None
    status: str
    quality: float
    provenance: Provenance
    timestamp: datetime
    # L2-derived performance context. Values are None when not derivable
    # (e.g. SFC at idle); bands are "UNKNOWN" then.
    sfc_kg_kwh: float | None = None
    sfc_band: str = "UNKNOWN"
    sfc_degradation_pct: float | None = None
    engine_load: float | None = None
    load_band: str = "UNKNOWN"
    # Environment (ISA). None when ambient pressure/temperature is invalid.
    pressure_altitude_m: float | None = None
    isa_deviation_k: float | None = None
    density_altitude_m: float | None = None
    # Electrical (Prompt 12). None / "UNKNOWN" / INVALID when not derivable
    # (e.g. electrical signals not instrumented, or R_int not observable).
    bus_voltage_v: float | None = None
    alternator_current_a: float | None = None
    battery_current_a: float | None = None
    charging_residual_v: float | None = None
    battery_resistance_mohm: float | None = None
    voltage_ripple_pct: float | None = None
    ripple_dominant_frequency_hz: float | None = None
    electrical_health_index: float | None = None
    electrical_health_band: str = "UNKNOWN"
    electrical_status: str = "INVALID"
    electrical_invalid_reasons: dict[str, str] = Field(default_factory=dict)
    electrical_advisories: list[str] = Field(default_factory=list)
    # Coolant and overheating trends (Prompt 14). None / INVALID when not derivable
    # (no coolant circuit, NTC absent, too little history for a trend).
    coolant_temp_c: float | None = None
    coolant_residual_k: float | None = None
    coolant_cht_delta_k: float | None = None
    coolant_status: str = "INVALID"
    coolant_invalid_reasons: dict[str, str] = Field(default_factory=dict)
    overheat_status: str = "INVALID"
    overheat_trends: dict[str, dict] = Field(default_factory=dict)
    overheat_advisories: list[str] = Field(default_factory=list)


class DiagnosticSummaryResponse(BaseModel):
    """Summary DTO of all L2 physics diagnostics."""

    rpm: float
    map_pa: float
    egt_mean_k: float
    egt_spread_k: float
    oil_temp_k: float
    oil_pressure_pa: float
    vibration_rms_m_s2: float
    misfire_detected: list[bool]
    overall_status: str
    timestamp: datetime
    provenance: Provenance


class AnomalyResponse(BaseModel):
    """Anomaly detection response DTO."""

    is_anomaly: bool | None
    anomaly_score: float | None
    threshold: float | None
    status: str
    evidence: dict[str, Any]
    quality: float
    provenance: Provenance
    timestamp: datetime
    records_in_window: int = 0  # records the running pipeline has processed (OI-20)


class FaultClassificationResponse(BaseModel):
    """Fault classification response DTO (unified taxonomy, FAULT_CLASS_COUNT classes)."""

    predicted_class: str
    class_id: int
    confidence: float | None
    probabilities: dict[str, float] | None
    status: str
    quality: float
    provenance: Provenance
    timestamp: datetime
    records_in_window: int = 0  # records the running pipeline has processed (OI-20)
    # Versioning (Prompt 17)
    baseline_version: str | None = None
    adapted_baseline: bool = False
    model_version: str | None = None
    hours_since_baseline: float | None = None
    # Condition flags (symptoms; None = not evaluable) and the rule-based
    # diagnosis over the L2 evidence (is_ml=False), reported alongside the ML result.
    condition_flags: dict[str, bool | None] = Field(default_factory=dict)
    condition_evidence: dict[str, Any] = Field(default_factory=dict)
    rule_based: dict[str, Any] | None = None


class RULResponse(BaseModel):
    """Remaining Useful Life estimation response DTO."""

    hours_remaining: float | None
    lower_bound_hours: float | None
    upper_bound_hours: float | None
    unit: str = "hours"
    trend: str
    operating_assumption: str
    status: str
    quality: float
    provenance: Provenance
    timestamp: datetime
    records_in_window: int = 0  # records the running pipeline has processed (OI-20)
    # Prompt 17: False unless a lifetime-validated RUL method produced the value.
    # When False, status is "UNVALIDATED" and the number must not be trusted.
    validated: bool = False
    validation_note: str | None = None
    inference_status: str | None = None
    interval_calibrated: bool = False    # False: lower/upper bounds are NOT a calibrated 90 % interval
    interval_note: str | None = None


class MissionRiskResponse(BaseModel):
    """Mission phase and risk assessment response DTO."""

    flight_phase: str
    risk_score: float | None
    risk_level: str
    risk_trend: str
    health_index: float | None
    rul_hours: float | None
    contributing_fault: str | None
    quality: float
    provenance: Provenance
    timestamp: datetime
    records_in_window: int = 0  # records the running pipeline has processed (OI-20)
    rul_validated: bool = False  # rul_hours is unvalidated unless True (Prompt 17)


class AdvisoryResponse(BaseModel):
    """Decision-support advisory response DTO."""

    advisory_id: str
    category: str
    priority: str
    title: str
    message: str
    subsystem: str
    confidence: float
    provenance: Provenance
    timestamp: datetime
    limitations: str


class ExplanationResponse(BaseModel):
    """Human-readable explanation response DTO."""

    explanation_id: str
    finding: str
    observation: str
    interpretation: str
    evidence: list[dict[str, Any]]
    uncertainty: str
    limitation: str
    provenance: Provenance
    quality: float
    timestamp: datetime


class DiagnosticQueryRequest(BaseModel):
    """Operator diagnostic query request DTO."""

    question_type: str = Field(description="Query category (e.g. HEALTH_STATUS, FAULT_STATUS, WHAT_CHANGED)")
    time_window_s: float | None = Field(default=None, description="Optional time window in seconds")


class DiagnosticQueryResponse(BaseModel):
    """Operator diagnostic query answer DTO."""

    question_type: str
    answer: str
    evidence: list[dict[str, Any]]
    limitations: str
    quality: float
    provenance: Provenance
    timestamp: datetime


class ReplayActionRequest(BaseModel):
    """Request DTO for scenario replay control."""

    scenario_id: str = Field(default="default_scenario")
    position: int | None = Field(default=None, ge=0)
    target_time: datetime | None = None


class ReplayStateResponse(BaseModel):
    """Current state of scenario replay response DTO."""

    scenario_id: str
    replay_position: int
    total_records: int
    status: str
    current_timestamp: datetime | None
    start_time: datetime | None
    end_time: datetime | None
    sequence_number: int
    speed_multiplier: float
    provenance: Provenance


class WhatIfRequest(BaseModel):
    """Request DTO for creating and executing a what-if scenario."""

    baseline_scenario_id: str = Field(default="baseline_01")
    modifications: dict[str, Any] = Field(description="Allowed modification keys (rpm_profile, fault_scenarios, seed, etc.)")
    duration_s: float = Field(default=30.0, gt=0.0, le=600.0)
    dt_s: float = Field(default=1.0, ge=0.01, le=10.0)


class WhatIfResponse(BaseModel):
    """Response DTO for what-if scenario execution."""

    what_if_id: str
    parent_scenario_id: str
    sample_count: int
    seed: int
    provenance: Provenance
    created_at: datetime


class ScenarioCompareRequest(BaseModel):
    """Request DTO for comparing baseline and what-if scenario pipeline outputs."""

    baseline_scenario_id: str
    what_if_id: str


class ScenarioCompareResponse(BaseModel):
    """Response DTO for scenario comparison results."""

    baseline_id: str
    what_if_id: str
    metric_deltas: dict[str, Any]
    state_changes: dict[str, Any]
    timestamp_alignment_info: dict[str, Any]
    ground_truth_validation: dict[str, Any] | None
    provenance: Provenance


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response container."""

    items: list[T]
    total_items: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=1000)
    total_pages: int


class EventEnvelope(BaseModel):
    """Real-time WebSocket event message envelope."""

    event_type: str = Field(description="Event category (e.g. telemetry_update, health_update, fault_update)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = Field(default=0)
    payload: dict[str, Any]
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Provenance = Field(default=Provenance.DERIVED)
