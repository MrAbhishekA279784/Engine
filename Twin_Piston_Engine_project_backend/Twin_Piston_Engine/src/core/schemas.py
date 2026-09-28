"""
Schemas — Base Pydantic models used across all layers.

Provides generic wrappers for provenance-tagged values, timestamped records,
and operating point definitions. All domain schemas inherit from these.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.core.provenance import (
    AlertLevel,
    ChannelValidity,
    DegradationState,
    FAULT_CLASS_COUNT,
    DiagnosticStatus,
    FaultClass,
    FlightPhase,
    InferenceStatus,
    ModelLifecycleStatus,
    Provenance,
)

T = TypeVar("T")


class ProvenanceTaggedValue(BaseModel, Generic[T]):
    """A value paired with its provenance, validity, and quality metadata.

    Every numeric output in the system must be wrapped in this container
    so that downstream consumers always know:
      1. Where the value came from (provenance)
      2. Whether it should be trusted (valid / quality)
      3. Why it might be untrustworthy (fault_flag)
    """

    value: T
    provenance: Provenance
    valid: bool = True
    validity_reason: ChannelValidity = ChannelValidity.VALID
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    fault_flag: str | None = None

    model_config = ConfigDict(frozen=True)

    def as_invalid(self, reason: ChannelValidity, fault_flag: str) -> ProvenanceTaggedValue[T]:
        """Return a copy marked invalid with the given reason."""
        return self.model_copy(
            update={
                "valid": False,
                "validity_reason": reason,
                "quality": 0.0,
                "fault_flag": fault_flag,
            }
        )


class TimestampedRecord(BaseModel):
    """Base class for all time-indexed records."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)


class OperatingPoint(BaseModel):
    """Engine operating point — the independent variables that define the
    current engine state. Used as the lookup key for baseline expectation
    models in L2.

    All values in SI units.
    """

    rpm: float = Field(ge=0, le=6500, description="Engine speed [rev/min]")
    map_pressure_pa: float = Field(ge=20000, le=200000, description="Manifold absolute pressure [Pa]")
    altitude_m: float = Field(ge=0, le=15000, description="Altitude [m]")
    ambient_temp_k: float = Field(ge=210, le=330, description="Ambient temperature [K]")
    ambient_pressure_pa: float = Field(ge=30000, le=110000, description="Ambient pressure [Pa]")
    throttle_pct: float | None = Field(
        default=None, ge=0, le=100, description="Throttle position [%]; None = not instrumented"
    )

    model_config = ConfigDict(frozen=True)


class ChannelRange(BaseModel):
    """Valid range for a telemetry channel, used in packet integrity checks."""

    min: float
    max: float

    def contains(self, value: float) -> bool:
        """Check if value falls within [min, max]."""
        return self.min <= value <= self.max


class ModelMetadata(BaseModel):
    """Metadata for a serialized ML model."""

    name: str
    version: str
    trained_at: datetime
    input_shape: list[int]
    output_shape: list[int]
    accuracy_metrics: dict[str, float] = Field(default_factory=dict)
    description: str = ""

    model_config = ConfigDict(frozen=True)


class SignalQuality(BaseModel):
    """Signal quality metadata for telemetric measurements."""

    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall quality score [0..1]")
    valid: bool = Field(default=True, description="Whether overall packet signals are valid")
    invalid_channels: list[str] = Field(default_factory=list, description="List of channels marked invalid")
    invalid_reasons: dict[str, str] = Field(default_factory=dict, description="Channel to invalidity reason map")
    noise_level: float = Field(default=0.0, ge=0.0, description="Estimated signal noise level")

    # extra="forbid": unknown keywords (e.g. quality=, fault_flag=) used to be
    # silently dropped, leaving a fault record looking healthy.
    model_config = ConfigDict(frozen=True, extra="forbid")


class DerivedEngineState(TimestampedRecord):
    """L2 Physics digital twin derived engineering parameters."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    bmep_pa: ProvenanceTaggedValue[float]
    imep_pa: ProvenanceTaggedValue[float | None]
    eta_thermal: ProvenanceTaggedValue[float]
    # Model-expected volumetric efficiency from the breathing model, not a
    # measurement. Not a health indicator: measured deviation shows in lambda.
    eta_volumetric: ProvenanceTaggedValue[float | None]
    afr: ProvenanceTaggedValue[float | None]
    brake_torque_nm: ProvenanceTaggedValue[float]
    brake_power_kw: ProvenanceTaggedValue[float]
    fmep_pa: ProvenanceTaggedValue[float]
    eta_mechanical: ProvenanceTaggedValue[float | None]
    mean_piston_speed_m_s: ProvenanceTaggedValue[float]
    # lambda = m_dot_air / (m_dot_fuel * AFR_stoich); None/valid=False when underivable.
    lambda_derived: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(
            value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0
        )
    )
    combustion_efficiency: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(
            value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0
        )
    )
    # Brake SFC (M-07); None/valid=False below 1 kW or without measured fuel flow.
    sfc_kg_kwh: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(
            value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0
        )
    )
    sfc_band: str = "UNKNOWN"
    sfc_degradation_pct: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(
            value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0
        )
    )
    # Normalised engine load P_brake / P_rated, clipped at 1.2 (M-08).
    engine_load: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(
            value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0
        )
    )
    load_band: str = "UNKNOWN"
    # Environment (ISO 2533 ISA): pressure altitude from ambient pressure,
    # ISA deviation = OAT - ISA temperature at that altitude, density altitude
    # from rho = p / (R T). valid=False when ambient p or T is invalid.
    pressure_altitude_m: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)
    )
    isa_deviation_k: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)
    )
    density_altitude_m: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)
    )

    model_config = ConfigDict(frozen=True)


class DiagnosticState(TimestampedRecord):
    """Subsystem & per-cylinder diagnostic status."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    per_cylinder_egt_dev_k: list[float] = Field(default_factory=list)
    per_cylinder_cht_dev_k: list[float] = Field(default_factory=list)
    misfire_detected: list[bool] = Field(default_factory=lambda: [False, False, False, False])
    vibration_rms_m_s2: float = 0.0
    lubrication_pressure_ok: bool = True
    cooling_temp_ok: bool = True

    # Extended Module 7 EGT fields
    per_cylinder_egt_k: list[float | None] = Field(default_factory=list)
    per_cylinder_egt_status: list[DiagnosticStatus] = Field(default_factory=list)
    egt_mean_k: float | None = None
    egt_spread_k: float | None = None
    max_egt_cylinder: int | None = None
    min_egt_cylinder: int | None = None
    egt_rate_of_change_k_s: list[float | None] = Field(default_factory=list)
    egt_overall_status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)

    model_config = ConfigDict(frozen=True)


def _not_computed() -> ProvenanceTaggedValue[float | None]:
    return ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)


class LubricationState(TimestampedRecord):
    """L2 Physics digital twin derived lubrication system state."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    oil_temperature_k: ProvenanceTaggedValue[float]
    oil_pressure_pa: ProvenanceTaggedValue[float]
    dynamic_viscosity_pa_s: ProvenanceTaggedValue[float]
    pressure_margin_pa: ProvenanceTaggedValue[float]
    temperature_margin_k: ProvenanceTaggedValue[float]
    status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)
    # Lubrication Health Index (M-10), expectation-normalised. Coverage is the
    # fraction of component weight that was valid (viscosity never is).
    lhi: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    lhi_band: str = "UNKNOWN"
    lhi_coverage: float = 0.0
    lhi_evidence: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class ElectricalState(TimestampedRecord):
    """L2 derived charging-system and battery state (Prompt 12).

    Every value is provenance-tagged; a value that cannot be derived is
    valid=False with value None and the reason in fault_flag.
    """

    provenance: Provenance = Field(default=Provenance.DERIVED)
    bus_voltage_v: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    alternator_current_a: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    battery_current_a: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    expected_bus_voltage_v: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    charging_regime: str = "UNKNOWN"  # BATTERY (below cut-in) | REGULATED (above cut-in) | UNKNOWN
    charging_residual_v: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    charging_residual_median_v: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    battery_resistance_mohm: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    r_int_step_count: int = 0
    voltage_ripple_pct: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    ripple_dominant_frequency_hz: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    ripple_order: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    charging_status: DiagnosticStatus = DiagnosticStatus.INVALID
    ripple_status: DiagnosticStatus = DiagnosticStatus.INVALID
    battery_status: DiagnosticStatus = DiagnosticStatus.INVALID
    status: DiagnosticStatus = DiagnosticStatus.INVALID
    ehi: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    ehi_band: str = "UNKNOWN"
    ehi_coverage: float = 0.0
    ehi_evidence: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


def _four_not_computed() -> list[ProvenanceTaggedValue[float | None]]:
    return [_not_computed() for _ in range(4)]


class InjectionState(TimestampedRecord):
    """L2 injection, ignition and fuel-system state (Prompt 13).

    Per-cylinder lists are ordered cylinder 1..4. Every value is provenance-
    tagged; a value that cannot be derived is valid=False, value None, with
    the reason in fault_flag.
    """

    provenance: Provenance = Field(default=Provenance.DERIVED)
    # Fuel rail
    fuel_rail_pressure_kpa: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)  # gauge
    rail_dp_kpa: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)  # rail - manifold
    expected_rail_dp_kpa: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    rail_pressure_residual_kpa: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    rail_pressure_residual_median_kpa: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    rail_mixture_factor: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)  # sqrt(dP/dP_ref)
    # Commanded fuel (injector model) and fuel delivery consistency (SRD-FUN-084)
    fuel_cmd_per_cycle_mg: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    injector_duty_pct: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    injector_duty_saturated: list[bool] = Field(default_factory=lambda: [False] * 4)
    fuel_cmd_total_kg_s: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    fuel_delivery_ratio: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    fuel_delivery_ratio_median: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    # Injector flow from EGT (SRD-FUN-044 separates lean-hot from non-firing-cold)
    lambda_operating: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    egt_differential_k: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    injector_flow_ratio: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    injector_flow_ratio_median: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    injector_status: list[DiagnosticStatus] = Field(default_factory=lambda: [DiagnosticStatus.INVALID] * 4)
    identified_injectors: list[int] = Field(default_factory=list)
    # Ignition timing vs the ECU schedule; retard is evidence for knock control
    scheduled_advance_deg: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    ign_timing_residual_deg: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    ign_timing_residual_median_deg: list[ProvenanceTaggedValue[float | None]] = Field(default_factory=_four_not_computed)
    knock_suspected: list[bool] = Field(default_factory=lambda: [False] * 4)
    fuel_system_status: DiagnosticStatus = DiagnosticStatus.INVALID
    knock_status: DiagnosticStatus = DiagnosticStatus.INVALID
    status: DiagnosticStatus = DiagnosticStatus.INVALID
    evidence: list[str] = Field(default_factory=list)


class CoolantState(TimestampedRecord):
    """L2 coolant-circuit state (Prompt 14). Unavailable (valid=False with the
    reason) when the engine has no liquid coolant circuit or the NTC is absent."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    coolant_temp_c: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    expected_coolant_c: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    coolant_residual_k: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    coolant_residual_median_k: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    # CHT - coolant: rises when head-to-coolant heat transfer degrades
    coolant_cht_delta_k: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    expected_cht_delta_k: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    cht_delta_residual_k: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    status: DiagnosticStatus = DiagnosticStatus.INVALID
    evidence: list[str] = Field(default_factory=list)


class ChannelTrend(BaseModel):
    """Overheating trend for one temperature channel (Prompt 14)."""

    channel: str
    current_c: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    limit_c: float | None = None
    measured_slope_k_min: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    expected_slope_k_min: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    # measured slope - expected slope (Theil-Sen of the residual), with its CI
    slope_residual_k_min: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    slope_residual_ci_k_min: tuple[float, float] | None = None
    time_to_limit_s: ProvenanceTaggedValue[float | None] = Field(default_factory=_not_computed)
    time_to_limit_range_s: tuple[float, float] | None = None
    samples: int = 0
    status: DiagnosticStatus = DiagnosticStatus.INVALID

    model_config = ConfigDict(frozen=True)


class OverheatTrendState(TimestampedRecord):
    """Overheating trend prediction for CHT, each EGT, oil and coolant (Prompt 14)."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    channels: dict[str, ChannelTrend] = Field(default_factory=dict)
    status: DiagnosticStatus = DiagnosticStatus.INVALID
    alerts: list[str] = Field(default_factory=list)


class ChannelDrift(BaseModel):
    """PSI drift of one residual channel vs its baseline reference (Prompt 17)."""

    channel: str
    psi: float | None = None
    band: str = "UNKNOWN"          # STABLE / MODERATE / SIGNIFICANT / UNKNOWN
    window_records: int = 0
    reason: str | None = None

    model_config = ConfigDict(frozen=True)


class DriftState(TimestampedRecord):
    """Residual distribution drift (Prompt 17): per channel and overall (max PSI)."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    channels: dict[str, ChannelDrift] = Field(default_factory=dict)
    overall_psi: float | None = None
    overall_band: str = "UNKNOWN"
    reference_version: str | None = None
    reference_status: str = "NO_REFERENCE"   # NO_REFERENCE / BUILDING / READY


class VibrationState(TimestampedRecord):
    """L2 Physics digital twin derived vibration feature state."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    # value is None (valid=False) when no full-window accelerometer burst exists.
    rms_x_m_s2: ProvenanceTaggedValue[float | None]
    rms_y_m_s2: ProvenanceTaggedValue[float | None]
    rms_z_m_s2: ProvenanceTaggedValue[float | None]
    overall_rms_m_s2: ProvenanceTaggedValue[float | None]
    peak_m_s2: ProvenanceTaggedValue[float | None]
    crest_factor: ProvenanceTaggedValue[float | None]
    dominant_freq_hz: ProvenanceTaggedValue[float | None]
    dominant_amplitude_m_s2: ProvenanceTaggedValue[float | None]
    dominant_order: ProvenanceTaggedValue[float | None]
    # Condition features, overall = max over X/Y/Z. valid=False / None when not
    # computable (no window, rpm unknown, band above Nyquist). Defaults keep
    # older constructors working and report "not computed", never 0.0.
    excess_kurtosis: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    half_order_fraction: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    firing_order_fraction: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    envelope_rms_m_s2: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    # Vibration Health Index (M-10): (RMS/ref) x (crest/ref) x (envelope/ref).
    vibration_health_index: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    vhi_band: str = "UNKNOWN"
    vhi_dominant_factor: str = "none"
    # Rotating imbalance at propeller-shaft 1X (M-12, SRD-FUN-070; Prompt 15)
    prop_1x_hz: ProvenanceTaggedValue[float | None] = Field(default_factory=lambda: _not_computed())
    prop_1x_velocity_fraction_lateral: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: _not_computed())
    prop_1x_lateral_vertical_ratio: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: _not_computed())

    model_config = ConfigDict(frozen=True)


class CombustionStabilityState(TimestampedRecord):
    """L2 Physics digital twin derived combustion stability & misfire diagnostic state."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    misfire_detected: list[bool] = Field(default_factory=lambda: [False, False, False, False])
    misfire_status: list[DiagnosticStatus] = Field(default_factory=lambda: [DiagnosticStatus.NORMAL]*4)
    evidence_score: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    overall_combustion_status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)
    # Dual-channel misfire gate (M-06): NORMAL / UNCONFIRMED / CONFIRMED /
    # INVALID, or NOT_EVALUATED for states built without the gate.
    misfire_verdict: str = "NOT_EVALUATED"
    misfire_gate_channel: str | None = None  # channel that fired when UNCONFIRMED
    misfire_gate_evidence: str = ""
    crank_cov_pct: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)
    )
    half_order_fraction: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)
    )
    # Combustion Stability Index (M-09); CHT term is a slope residual.
    csi_value: ProvenanceTaggedValue[float | None] = Field(
        default_factory=lambda: ProvenanceTaggedValue(value=None, provenance=Provenance.DERIVED, valid=False, quality=0.0)
    )
    csi_band: str = "UNKNOWN"
    csi_dominant_term: str = "none"
    csi_terms: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ResidualState(TimestampedRecord):
    """Physical actuals vs baseline expected residuals."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    operating_point: OperatingPoint
    residuals: dict[str, ProvenanceTaggedValue[float | None]] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class HealthState(TimestampedRecord):
    """L3 Supervision aggregate engine health metrics and degradation supervision."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    health_index: ProvenanceTaggedValue[float]
    degradation_state: DegradationState = Field(default=DegradationState.HEALTHY)
    # Components with valid evidence only (quality > 0). A component with no
    # valid data is absent here and listed in component_unavailable with its
    # reason; it is never given a default score.
    component_health: dict[str, float] = Field(default_factory=dict)
    component_unavailable: dict[str, str] = Field(default_factory=dict)
    # Versioning (Prompt 17): which per-engine baseline and which model produced this
    baseline_version: str = "fleet-0"
    adapted_baseline: bool = False
    model_version: str = "rules-only"
    hours_since_baseline: float | None = None
    health_status: DiagnosticStatus = Field(default=DiagnosticStatus.NORMAL)
    trend: str = Field(default="STABLE")
    # Theil-Sen dHI/dt [1/s] over the trend window; None while INSUFFICIENT_DATA
    health_rate: float | None = Field(default=None)
    evidence: dict[str, Any] = Field(default_factory=dict)
    contributing_fault: FaultClass | None = Field(default=None)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    model_version: str = Field(default="1.0.0")

    model_config = ConfigDict(frozen=True)


class RULState(TimestampedRecord):
    """Remaining Useful Life estimation with uncertainty boundaries."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    hours_remaining: float = Field(default=0.0, ge=0.0)
    lower_bound_hours: float | None = Field(default=None)
    upper_bound_hours: float | None = Field(default=None)
    confidence: float | None = Field(default=None)
    unit: str = Field(default="hours")
    status: InferenceStatus = Field(default=InferenceStatus.SUCCESS)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    uncertainty_available: bool = Field(default=False)
    uncertainty_representation: dict[str, Any] | None = Field(default=None)
    degradation_state: DegradationState = Field(default=DegradationState.HEALTHY)
    trend: str = Field(default="STABLE")
    operating_assumption: str = Field(default="constant_operating_profile")
    model_name: str | None = Field(default=None)
    model_version: str = Field(default="1.0.0")
    feature_schema_version: str = Field(default="1.0.0")
    evidence: dict[str, Any] = Field(default_factory=dict)
    is_ml: bool = Field(default=False)
    # Prompt 17: True only for a RUL method validated on held-out engines at
    # lifetime scale (reports/rul_metrics.md). The baseline trend estimator and
    # the single-record ML path are NOT validated: consumers must not treat
    # their number as trustworthy.
    validated: bool = Field(default=False)
    validation_note: str = Field(default="")
    # True only where the 90 % interval's coverage was measured at >= ~0.87 on
    # held-out engines (lifetime model, RUL below 150 h). Otherwise the bounds
    # are shown but labelled not calibrated.
    interval_calibrated: bool = Field(default=False)
    interval_note: str = Field(default="")

    model_config = ConfigDict(frozen=True)


class Alert(TimestampedRecord):
    """L4 Advisory alert notification."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    alert_level: AlertLevel = Field(default=AlertLevel.NOMINAL)
    fault_class: FaultClass = Field(default=FaultClass.NOMINAL)
    summary: str
    message: str
    recommended_action: str

    model_config = ConfigDict(frozen=True)


class AnomalyResult(TimestampedRecord):
    """Result of anomaly detection evaluation."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    status: InferenceStatus = Field(default=InferenceStatus.SUCCESS)
    is_anomaly: bool = Field(default=False)
    anomaly_score: float = Field(default=0.0, ge=0.0)
    threshold: float = Field(default=0.5, ge=0.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    model_metadata: ModelMetadata | None = Field(default=None)
    is_ml: bool = Field(default=True)

    model_config = ConfigDict(frozen=True)


class FaultClassificationResult(TimestampedRecord):
    """Result of fault classification (unified taxonomy, FAULT_CLASS_COUNT classes)."""

    provenance: Provenance = Field(default=Provenance.MODEL_OUTPUT)
    status: InferenceStatus = Field(default=InferenceStatus.SUCCESS)
    predicted_class: FaultClass = Field(default=FaultClass.NOMINAL)
    class_id: int = Field(default=0, ge=0, le=FAULT_CLASS_COUNT - 1)
    class_name: str = Field(default="NOMINAL")
    confidence: float | None = Field(default=None)
    probabilities: dict[str, float] | None = Field(default=None)
    feature_schema_version: str = Field(default="1.0.0")
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] | None = Field(default=None)
    model_metadata: ModelMetadata | None = Field(default=None)
    is_ml: bool = Field(default=True)
    # Multi-label condition flags (symptoms, co-occur with any class; Prompt 15):
    # overheating_trend, combustion_instability, lubrication_degraded.
    # None = not evaluable (input absent / band UNKNOWN), never a substituted False.
    condition_flags: dict[str, bool | None] = Field(default_factory=dict)
    condition_evidence: dict[str, Any] = Field(default_factory=dict)
    # Versioning (Prompt 17): which per-engine baseline and which model produced this
    baseline_version: str = "fleet-0"
    adapted_baseline: bool = False
    model_version: str = "rules-only"
    hours_since_baseline: float | None = None

    model_config = ConfigDict(frozen=True)


class MissionState(TimestampedRecord):
    """Mission phase and analytical risk assessment."""

    provenance: Provenance = Field(default=Provenance.DERIVED)
    flight_phase: FlightPhase = Field(default=FlightPhase.GROUND)
    mission_hours: float = Field(default=0.0, ge=0.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    go_no_go_recommendation: str = "GO"
    mission_id: str | None = Field(default=None)
    phase_quality: float = Field(default=1.0, ge=0.0, le=1.0)
    previous_phase: FlightPhase | str | None = Field(default=None)
    phase_transition: str | None = Field(default=None)
    risk_level: str = Field(default="LOW")
    risk_trend: str = Field(default="STABLE")
    health_index: float | None = Field(default=None)
    degradation_state: DegradationState | None = Field(default=None)
    rul_hours: float | None = Field(default=None)
    contributing_fault: FaultClass | None = Field(default=None)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


def make_tagged(
    value: Any,
    provenance: Provenance,
    valid: bool = True,
    quality: float = 1.0,
) -> ProvenanceTaggedValue[Any]:
    """Convenience factory for creating provenance-tagged values."""
    return ProvenanceTaggedValue(
        value=value,
        provenance=provenance,
        valid=valid,
        quality=quality,
    )

