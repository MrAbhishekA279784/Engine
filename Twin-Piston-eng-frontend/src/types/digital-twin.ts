export type EngineStatus = "normal" | "caution" | "critical";
export type AnomalyStatus = "NORMAL" | "WARNING" | "ANOMALY" | "CRITICAL";
export type DegradationState = "Stable" | "Gradual" | "Accelerating" | "Critical";
export type MaintenanceStatus = "MONITOR" | "SCHEDULE" | "GROUND";
export type TrendDirection = "Stable" | "Increasing" | "Decreasing";

export type Parameter = {
  label: string;
  value: string;
  unit?: string;
  status?: EngineStatus;
};

export type ExpectedVsObserved = {
  parameter: string;
  expected: string;
  observed: string;
  delta: string;
  unit: string;
  status: EngineStatus;
};

export type XAIFactor = {
  signal: string;
  delta: string;
  direction: "up" | "down" | "neutral";
};

export type EngineComponent = {
  id: string;
  number?: number;
  name: string;
  shortName: string;
  category?: string;
  description: string;
  function: string;
  parameters: Parameter[];
  expectedVsObserved?: ExpectedVsObserved[];
  health: number;
  status: EngineStatus;
  faultProbability: number;
  // ML diagnostics
  anomalyScore: number;
  anomalyStatus: AnomalyStatus;
  degradationState: DegradationState;
  rul: number;
  rulConfidence: number;
  rulRange: [number, number];
  predictionConfidence: number;
  xaiFactors: XAIFactor[];
  xaiAssessment: string;
  maintenanceStatus: MaintenanceStatus;
  maintenanceNote: string;
  nextInspectionHours: number;
  healthHistory: number[]; // health % at [0, 25, 50, 75, 100, 125, 150, 175, 200] hours
  operatingHours?: number;
  thermalCycles?: number;
};

export type TelemetryMetric = {
  id: string;
  label: string;
  value: string;
  unit: string;
  status: EngineStatus;
  trend: number[];
  expected?: string;
  observed?: string;
  deviation?: string;
};

export type MLPrediction = {
  healthIndex: number;
  anomalyScore: number;
  faultProbability: number;
  predictedFault: string;
  confidence: number;
  degradationState: DegradationState;
  rul: number;
  rulConfidence: number;
  rulRange: [number, number];
  rulTrend: TrendDirection;
  anomalyStatus: AnomalyStatus;
  anomalyComponent: string | null;
  anomalySeverity: string | null;
  contributingSignals: string[];
};

export type MissionData = {
  successProbability: number;
  risk: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  safeOperatingTime: number;
  engineCondition: string;
  health: number;
  faultProbability: number;
  anomalyScore: number;
  degradationRate: string;
  rul: string;
  status: EngineStatus;
};
