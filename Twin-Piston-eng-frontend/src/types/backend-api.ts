export type Provenance =
  | "REAL"
  | "SIMULATED"
  | "CSV_REPLAY"
  | "DERIVED"
  | "MODEL_OUTPUT";

export type FaultClass =
  | "NOMINAL"
  | "MISFIRE"
  | "DETONATION_KNOCK"
  | "EXHAUST_VALVE_LEAK"
  | "INTAKE_BOOST_LEAK"
  | "OIL_DEGRADATION"
  | "COOLING_FAULT"
  | "BEARING_WEAR"
  | "SENSOR_FAULT";

export type FlightPhase =
  | "GROUND"
  | "TAKEOFF"
  | "CLIMB"
  | "CRUISE"
  | "DESCENT"
  | "LANDING";

export type AdvisoryCategory =
  | "MONITOR"
  | "INSPECT"
  | "INVESTIGATE"
  | "MAINTENANCE_REVIEW"
  | "DATA_QUALITY_CHECK"
  | "CONTINUE_MONITORING";

export type AdvisoryPriority =
  | "INFORMATION"
  | "LOW"
  | "MEDIUM"
  | "HIGH"
  | "CRITICAL";

export interface HealthCheckResponse {
  status: string;
  service_name: string;
  version: string;
  timestamp: string;
  details: Record<string, any>;
}

export interface SystemStatusResponse {
  service_status: string;
  telemetry_pipeline_status: string;
  digital_twin_status: string;
  ml_supervision_status: string;
  active_connections: number;
  timestamp: string;
}

export interface EngineHealthResponse {
  health_index: number | null;
  health_index_reason?: string | null;
  degradation_state: string;
  trend: string;
  health_rate_per_s?: number | null;
  component_health: Record<string, number | null>;
  component_health_reasons?: Record<string, string>;
  coverage?: number | null;
  records_in_window?: number;
  status: string;
  quality: number;
  provenance: Provenance;
  timestamp: string;
  sfc_kg_kwh?: number | null;
  sfc_band?: string;
  engine_load?: number | null;
  load_band?: string;
  bus_voltage_v?: number | null;
  alternator_current_a?: number | null;
  battery_current_a?: number | null;
  battery_resistance_mohm?: number | null;
  voltage_ripple_pct?: number | null;
  electrical_health_index?: number | null;
  electrical_status?: string;
  coolant_temp_c?: number | null;
  coolant_status?: string;
  overheat_status?: string;
  overheat_advisories?: string[];
}

export interface DiagnosticSummaryResponse {
  rpm: number;
  map_pa: number;
  egt_mean_k: number;
  egt_spread_k: number;
  oil_temp_k: number;
  oil_pressure_pa: number;
  vibration_rms_m_s2: number;
  misfire_detected: boolean[];
  overall_status: string;
  timestamp: string;
  provenance: Provenance;
}

export interface AnomalyResponse {
  is_anomaly: boolean | null;
  anomaly_score: number | null;
  threshold: number | null;
  status: string;
  evidence: Record<string, any>;
  quality: number;
  provenance: Provenance;
  timestamp: string;
  records_in_window?: number;
}

export interface FaultClassificationResponse {
  predicted_class: string;
  class_id: number;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  status: string;
  quality: number;
  provenance: Provenance;
  timestamp: string;
  records_in_window?: number;
  condition_flags?: Record<string, boolean | null>;
  rule_based?: Record<string, any> | null;
}

export interface RULResponse {
  hours_remaining: number | null;
  lower_bound_hours: number | null;
  upper_bound_hours: number | null;
  unit: string;
  trend: string;
  operating_assumption: string;
  status: string;
  quality: number;
  provenance: Provenance;
  timestamp: string;
  validated: boolean;
  validation_note?: string | null;
}

export interface MissionRiskResponse {
  flight_phase: string;
  risk_score: number | null;
  risk_level: string;
  risk_trend: string;
  health_index: number | null;
  rul_hours: number | null;
  contributing_fault: string | null;
  quality: number;
  provenance: Provenance;
  timestamp: string;
  rul_validated: boolean;
}

export interface AdvisoryResponse {
  advisory_id: string;
  category: string;
  priority: AdvisoryPriority;
  title: string;
  message: string;
  subsystem: string;
  confidence: number;
  provenance: Provenance;
  timestamp: string;
  limitations: string;
}

export interface ExplanationResponse {
  explanation_id: string;
  finding: string;
  observation: string;
  interpretation: string;
  evidence: Array<Record<string, any>>;
  uncertainty: string;
  limitation: string;
  provenance: Provenance;
  quality: number;
  timestamp: string;
}

export interface DiagnosticQueryRequest {
  question_type: string;
  time_window_s?: number | null;
}

export interface DiagnosticQueryResponse {
  question_type: string;
  answer: string;
  evidence: Array<Record<string, any>>;
  limitations: string;
  quality: number;
  provenance: Provenance;
  timestamp: string;
}

export interface ReplayStateResponse {
  scenario_id: string;
  replay_position: number;
  total_records: number;
  status: string;
  current_timestamp: string | null;
  sequence_number: number;
  speed_multiplier: number;
  provenance: Provenance;
}

export interface WhatIfRequest {
  baseline_scenario_id?: string;
  modifications: Record<string, any>;
  duration_s?: number;
  dt_s?: number;
}

export interface WhatIfResponse {
  what_if_id: string;
  parent_scenario_id: string;
  sample_count: number;
  seed: number;
  provenance: Provenance;
  created_at: string;
}

export interface ScenarioCompareResponse {
  baseline_id: string;
  what_if_id: string;
  metric_deltas: Record<string, any>;
  state_changes: Record<string, any>;
  timestamp_alignment_info: Record<string, any>;
  ground_truth_validation?: Record<string, any> | null;
  provenance: Provenance;
}

export interface EventEnvelope {
  event_type: "telemetry_update" | "health_update" | "fault_update" | "advisory_update" | "replay_update";
  timestamp: string;
  sequence_number: number;
  payload: Record<string, any>;
  quality: number;
  provenance: Provenance;
}
