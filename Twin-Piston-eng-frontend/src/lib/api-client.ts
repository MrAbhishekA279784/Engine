import type {
  AdvisoryResponse,
  AnomalyResponse,
  DiagnosticQueryRequest,
  DiagnosticQueryResponse,
  DiagnosticSummaryResponse,
  EngineHealthResponse,
  ExplanationResponse,
  FaultClassificationResponse,
  HealthCheckResponse,
  MissionRiskResponse,
  ReplayStateResponse,
  RULResponse,
  ScenarioCompareResponse,
  SystemStatusResponse,
  WhatIfRequest,
  WhatIfResponse,
} from "@/types/backend-api";

const BASE_URL = "http://localhost:8000/api/v1";

async function fetchJSON<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status} on ${endpoint}: ${errorText}`);
  }

  return response.json();
}

export const apiClient = {
  // System Endpoints
  getSystemHealth: () => fetchJSON<HealthCheckResponse>("/system/health"),
  getSystemStatus: () => fetchJSON<SystemStatusResponse>("/system/status"),

  // Telemetry Endpoints
  getLatestTelemetry: () => fetchJSON<Record<string, any>>("/telemetry/latest"),
  getTelemetryHistory: (page = 1, pageSize = 50) =>
    fetchJSON<any>(`/telemetry/history?page=${page}&page_size=${pageSize}`),

  // Diagnostics & Engine Health Endpoints
  getEngineHealth: () => fetchJSON<EngineHealthResponse>("/engine/health"),
  getDiagnosticSummary: () => fetchJSON<DiagnosticSummaryResponse>("/diagnostics"),
  getEgtDiagnostics: () => fetchJSON<Record<string, any>>("/diagnostics/egt"),
  getLubricationDiagnostics: () => fetchJSON<Record<string, any>>("/diagnostics/lubrication"),
  getVibrationDiagnostics: () => fetchJSON<Record<string, any>>("/diagnostics/vibration"),
  getCombustionDiagnostics: () => fetchJSON<Record<string, any>>("/diagnostics/combustion"),

  // ML Supervision & Risk Endpoints
  getAnomalyStatus: () => fetchJSON<AnomalyResponse>("/diagnostics/anomaly"),
  getFaultClassification: () => fetchJSON<FaultClassificationResponse>("/diagnostics/fault"),
  getRUL: () => fetchJSON<RULResponse>("/engine/rul"),
  getMissionRisk: () => fetchJSON<MissionRiskResponse>("/mission/risk"),

  // Advisories, Explanations & Operator Q&A
  getAdvisories: () => fetchJSON<AdvisoryResponse[]>("/advisories"),
  getAdvisoryById: (id: string) => fetchJSON<AdvisoryResponse>(`/advisories/${id}`),
  getExplanations: () => fetchJSON<ExplanationResponse[]>("/explanations"),
  queryDiagnostics: (query: DiagnosticQueryRequest) =>
    fetchJSON<DiagnosticQueryResponse>("/diagnostic/query", {
      method: "POST",
      body: JSON.stringify(query),
    }),

  // Scenario Replay
  getReplayState: () => fetchJSON<ReplayStateResponse>("/replay/state"),
  startReplay: (scenario_id = "default_scenario", position = 0) =>
    fetchJSON<ReplayStateResponse>("/replay/start", {
      method: "POST",
      body: JSON.stringify({ scenario_id, position }),
    }),
  pauseReplay: () => fetchJSON<ReplayStateResponse>("/replay/pause", { method: "POST" }),
  resumeReplay: () => fetchJSON<ReplayStateResponse>("/replay/resume", { method: "POST" }),
  stopReplay: () => fetchJSON<ReplayStateResponse>("/replay/stop", { method: "POST" }),
  seekReplay: (position: number) =>
    fetchJSON<ReplayStateResponse>("/replay/seek", {
      method: "POST",
      body: JSON.stringify({ position }),
    }),

  // What-If Scenarios
  executeWhatIf: (request: WhatIfRequest) =>
    fetchJSON<WhatIfResponse>("/scenarios/what-if", {
      method: "POST",
      body: JSON.stringify(request),
    }),
  compareScenarios: (baseline_scenario_id: string, what_if_id: string) =>
    fetchJSON<ScenarioCompareResponse>("/scenarios/compare", {
      method: "POST",
      body: JSON.stringify({ baseline_scenario_id, what_if_id }),
    }),
};
