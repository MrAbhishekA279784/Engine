import { useState } from "react";
import type { EngineComponent } from "@/types/digital-twin";
import {
  Thermometer,
  Activity,
  Gauge,
  Clock,
  ShieldCheck,
  AlertTriangle,
  ChevronLeft,
  X,
  FileText,
  TrendingDown,
  Wrench,
  CheckCircle2,
} from "lucide-react";

export function ComponentInfoPanel({
  component,
  components,
  onSelect,
  onClose,
}: {
  component: EngineComponent;
  components: EngineComponent[];
  onSelect: (id: string) => void;
  onClose?: () => void;
}) {
  const [tab, setTab] = useState<"Overview" | "Parameters" | "History">("Overview");
  const health = component.health;
  const isCaution = component.status === "caution";
  const isCritical = component.status === "critical";

  return (
    <aside className="detail-panel" aria-labelledby="ci-title">
      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="ci-header">
        <div className="ci-header-title">
          <ChevronLeft className="ci-back-icon" onClick={onClose} />
          <h2 id="ci-title">COMPONENT INTELLIGENCE</h2>
        </div>
        {onClose && (
          <button type="button" className="ci-close-btn" onClick={onClose} aria-label="Close component panel">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* ── Component Preview Card ─────────────────────────────────────── */}
      <div className="ci-preview-card">
        <div className="ci-preview-graphic">
          <div className="component-schematic-icon">
            <div className="schematic-ring" />
            <span className="schematic-num">#{component.number ?? 1}</span>
          </div>
        </div>
        <div className="ci-preview-info">
          <div className="ci-name-row">
            <h3>{component.name.toUpperCase()}</h3>
            <span className={`status-badge-chip ${isCaution ? "status-caution" : isCritical ? "status-critical" : "status-normal"}`}>
              {component.status.toUpperCase()}
            </span>
          </div>
          <span className="ci-subsystem">{component.category ?? "Engine Subsystem"}</span>
          <div className="ci-health-bar-wrap">
            <div className="ci-health-label">
              <span>Health</span>
              <strong className={isCaution ? "value-caution" : isCritical ? "value-critical" : "value-normal"}>
                {health}%
              </strong>
            </div>
            <div className="ci-health-track">
              <div
                className={`ci-health-fill ${isCaution ? "health-caution" : isCritical ? "health-critical" : "health-normal"}`}
                style={{ width: `${health}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Navigation Tabs ────────────────────────────────────────────── */}
      <div className="ci-tabs-bar">
        {(["Overview", "Parameters", "History"] as const).map((name) => (
          <button
            type="button"
            key={name}
            className={`ci-tab-btn ${tab === name ? "ci-tab-active" : ""}`}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>

      {/* ── Tab Contents ───────────────────────────────────────────────── */}
      <div className="ci-scroll-content">
        {tab === "Overview" && (
          <>
            {/* Function */}
            <div className="ci-section-box">
              <div className="ci-section-head">
                <FileText className="ci-sec-icon" />
                <span>FUNCTION</span>
              </div>
              <p className="ci-function-text">{component.function}</p>
            </div>

            {/* Key Parameters */}
            <div className="ci-section-box">
              <div className="ci-section-head">
                <Gauge className="ci-sec-icon" />
                <span>KEY PARAMETERS</span>
              </div>
              <div className="ci-params-grid">
                {component.parameters && component.parameters.length > 0 ? (
                  component.parameters.map((p) => {
                    const l = p.label.toLowerCase();
                    const IconComp = l.includes("temp")
                      ? Thermometer
                      : l.includes("press") || l.includes("flow") || l.includes("angle")
                      ? Gauge
                      : l.includes("vibr") || l.includes("speed") || l.includes("torque") || l.includes("volt")
                      ? Activity
                      : l.includes("rul") || l.includes("hour")
                      ? Clock
                      : Gauge;
                    return (
                      <div className="ci-param-row" key={p.label}>
                        <div className="param-label-group">
                          <IconComp className="p-icon" />
                          <span>{p.label}</span>
                        </div>
                        <strong>{p.value}</strong>
                      </div>
                    );
                  })
                ) : (
                  <div className="ci-param-row">
                    <span className="text-gray-400">Telemetry</span>
                    <strong>N/A</strong>
                  </div>
                )}
                {!component.parameters?.some((p) => p.label.toLowerCase().includes("rul")) && (
                  <div className="ci-param-row">
                    <div className="param-label-group">
                      <Clock className="p-icon" />
                      <span>Predicted RUL</span>
                    </div>
                    <strong className="value-cyan">{component.rul} h</strong>
                  </div>
                )}
              </div>
            </div>

            {/* AI Diagnostics */}
            <div className="ci-section-box">
              <div className="ci-section-head">
                <ShieldCheck className="ci-sec-icon" />
                <span>AI DIAGNOSTICS</span>
              </div>
              <div className="ci-params-grid">
                <div className="ci-param-row">
                  <span>Anomaly Score</span>
                  <strong className={component.anomalyScore > 0.35 ? "value-caution" : "value-normal"}>
                    {component.anomalyScore.toFixed(2)}
                  </strong>
                </div>
                <div className="ci-param-row">
                  <span>Fault Probability</span>
                  <strong className={component.faultProbability > 10 ? "value-caution" : ""}>
                    {component.faultProbability}%
                  </strong>
                </div>
                <div className="ci-param-row">
                  <span>Prediction Confidence</span>
                  <strong>{component.predictionConfidence}%</strong>
                </div>
              </div>
            </div>

            {/* Degradation Trend */}
            <div className="ci-section-box">
              <div className="ci-section-head">
                <TrendingDown className="ci-sec-icon" />
                <span>DEGRADATION TREND</span>
                <span className={`ci-mini-badge ${component.degradationState === "Stable" ? "badge-stable" : "badge-caution"}`}>
                  {component.degradationState ?? "Stable"}
                </span>
                <span className="ci-mini-cond">
                  Condition:{" "}
                  <strong>
                    {component.status === "normal"
                      ? "Healthy"
                      : component.status === "caution"
                      ? "Caution"
                      : "Critical"}
                  </strong>
                </span>
              </div>
              <div className="ci-degrad-chart">
                <svg viewBox="0 0 200 48" className="ci-svg-chart" aria-hidden="true">
                  <line x1="0" y1="12" x2="200" y2="12" className="chart-grid-line" />
                  <line x1="0" y1="24" x2="200" y2="24" className="chart-grid-line" />
                  <line x1="0" y1="36" x2="200" y2="36" className="chart-grid-line" />
                  <polyline
                    points={component.healthHistory
                      .map((val, i) => `${(i / (component.healthHistory.length - 1)) * 195 + 2},${46 - ((val - 70) / 30) * 36}`)
                      .join(" ")}
                    className="degrad-trend-line"
                  />
                  <circle
                    cx="197"
                    cy={46 - ((health - 70) / 30) * 36}
                    r="2.5"
                    className="degrad-trend-dot"
                  />
                </svg>
                <div className="chart-axis-hours">
                  <span>0</span>
                  <span>50</span>
                  <span>100</span>
                  <span>150</span>
                  <span>200 h</span>
                </div>
              </div>
            </div>

            {/* AI Explanation */}
            <div className="ci-section-box">
              <div className="ci-section-head">
                <AlertTriangle className="ci-sec-icon text-amber-400" />
                <span>AI EXPLANATION</span>
              </div>
              <p className="ci-explanation-text">{component.xaiAssessment}</p>
            </div>

            {/* Recommendation */}
            <div className="ci-section-box ci-rec-box">
              <div className="ci-section-head">
                <CheckCircle2 className="ci-sec-icon text-emerald-400" />
                <span>RECOMMENDATION</span>
              </div>
              <p className="ci-rec-text">{component.maintenanceNote}</p>
              <span className="ci-rec-sub">
                Next scheduled inspection in <strong>{component.nextInspectionHours} operating hours</strong>.
              </span>
            </div>
          </>
        )}

        {tab === "Parameters" && (
          <div className="ci-section-box">
            <div className="ci-section-head">
              <Gauge className="ci-sec-icon" />
              <span>EXPECTED VS OBSERVED TELEMETRY</span>
            </div>
            <div className="evo-table-wrap">
              <table className="evo-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Expected</th>
                    <th>Observed</th>
                    <th>Deviation</th>
                  </tr>
                </thead>
                <tbody>
                  {component.expectedVsObserved && component.expectedVsObserved.length > 0 ? (
                    component.expectedVsObserved.map((row) => (
                      <tr key={row.parameter}>
                        <td>{row.parameter}</td>
                        <td>{row.expected}</td>
                        <td>{row.observed}</td>
                        <td className={row.status === "caution" ? "value-caution" : "value-normal"}>
                          {row.delta}
                        </td>
                      </tr>
                    ))
                  ) : component.parameters && component.parameters.length > 0 ? (
                    component.parameters.map((p) => (
                      <tr key={p.label}>
                        <td>{p.label}</td>
                        <td>Nominal</td>
                        <td>{p.value}</td>
                        <td className="value-normal">0.0%</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={4} className="text-center py-2 text-gray-400">
                        No telemetry channels configured for this component
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {tab === "History" && (
          <div className="ci-section-box">
            <div className="ci-section-head">
              <Clock className="ci-sec-icon" />
              <span>HISTORICAL LOG & CYCLES</span>
            </div>
            <div className="history-meta-list">
              <div className="ci-param-row">
                <span>Total Operating Hours</span>
                <strong>{component.operatingHours ?? 200} h</strong>
              </div>
              <div className="ci-param-row">
                <span>Completed Thermal Cycles</span>
                <strong>{component.thermalCycles ?? 142} cycles</strong>
              </div>
              <div className="ci-param-row">
                <span>Degradation Rate</span>
                <strong>0.18% / 10h (Stable)</strong>
              </div>
              <div className="ci-param-row">
                <span>Last Inspection</span>
                <strong>50 h ago (Passed)</strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
