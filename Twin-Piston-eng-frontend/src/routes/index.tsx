import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState, useEffect } from "react";
import { Navigation, type NavItem } from "@/components/digital-twin/Navigation";
import { EngineViewer } from "@/components/digital-twin/EngineViewer";
import { ComponentInfoPanel } from "@/components/digital-twin/ComponentInfoPanel";
import { ComponentHealthGauge } from "@/components/digital-twin/ComponentHealthGauge";
import { ComponentsListModal } from "@/components/digital-twin/ComponentsListModal";
import { FaultAnalysisView } from "@/components/digital-twin/FaultAnalysisView";
import { RULView } from "@/components/digital-twin/RULView";
import { MissionTimelineView } from "@/components/digital-twin/MissionTimelineView";
import { EngineTrendCharts } from "@/components/digital-twin/EngineTrendCharts";
import { AdvisoriesPanel } from "@/components/digital-twin/AdvisoriesPanel";
import { WhatIfReplayPanel } from "@/components/digital-twin/WhatIfReplayPanel";
import { ElectricalDiagnosticsCard } from "@/components/digital-twin/ElectricalDiagnosticsCard";
import { EgtCombustionCard } from "@/components/digital-twin/EgtCombustionCard";
import { useEngineWebSocket } from "@/hooks/use-engine-websocket";
import {
  engineComponents,
  telemetry as mockTelemetry,
  mlPrediction as mockMlPrediction,
  mission as mockMission,
} from "@/data/mock-engine-data";
import { Shield, Activity, Compass, Cpu, Clock, CheckCircle } from "lucide-react";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Aeronoix | AeroTwin Sight Digital Twin" },
      {
        name: "description",
        content:
          "Aeronoix Twin-Piston Aircraft Engine real-time digital twin, telemetry health monitoring, and AI predictive maintenance.",
      },
    ],
  }),
});

function Index() {
  const [activeNav, setActiveNav] = useState<NavItem>("3D View");
  const [selectedId, setSelectedId] = useState<string>("cylinder-head-1");
  const [showComponentPanel, setShowComponentPanel] = useState<boolean>(true);

  // Live WebSocket connection to FastAPI backend
  const streamState = useEngineWebSocket();

  const [time, setTime] = useState({
    timeStr: "14:32:18",
    dateStr: "16 Sep 2026",
  });

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime({
        timeStr: now.toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
        dateStr: now.toLocaleDateString("en-GB", {
          day: "2-digit",
          month: "short",
          year: "numeric",
        }),
      });
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const healthCounts = useMemo(() => {
    let normal = 0;
    let caution = 0;
    let critical = 0;
    engineComponents.forEach((c) => {
      if (c.status === "normal") normal++;
      else if (c.status === "caution") caution++;
      else if (c.status === "critical") critical++;
    });
    return { normal, caution, critical };
  }, []);

  const rpmVal = useMemo(
    () => streamState.liveTelemetry?.rpm?.toString() ?? mockTelemetry.find((m) => m.id === "rpm")?.value ?? "2,480",
    [streamState.liveTelemetry]
  );
  const oilTempVal = useMemo(
    () => streamState.liveTelemetry?.oil_temp_c?.toString() ?? mockTelemetry.find((m) => m.id === "oil-temp")?.value ?? "92",
    [streamState.liveTelemetry]
  );

  const selected = useMemo(() => {
    return (
      engineComponents.find(
        (item) => item.id === selectedId || item.shortName.toLowerCase() === selectedId.toLowerCase()
      ) ?? engineComponents[3]!
    );
  }, [selectedId]);

  const handleComponentSelect = (id: string) => {
    setSelectedId(id);
    setShowComponentPanel(true);
  };

  return (
    <main className="chassis-root-frame">
      {/* ── Outer Industrial Corner Hex Screws ───────────────────────────── */}
      <div className="chassis-screw screw-top-left" />
      <div className="chassis-screw screw-top-right" />
      <div className="chassis-screw screw-bottom-left" />
      <div className="chassis-screw screw-bottom-right" />

      {/* ── Top Brushed Aluminium Aerospace Header ───────────────────────── */}
      <header className="aerospace-topbar">
        {/* Left: Aeronoix Branding */}
        <div className="topbar-brand-cluster">
          <div className="aeronoix-logo-shield">
            <svg viewBox="0 0 36 36" className="aeronoix-wings-svg">
              <path
                d="M18 4 L30 14 L26 26 L18 32 L10 26 L6 14 Z"
                fill="none"
                stroke="#00d4ff"
                strokeWidth="1.8"
              />
              <path
                d="M10 16 L18 8 L26 16 L18 24 Z"
                fill="rgba(0, 212, 255, 0.25)"
                stroke="#00d4ff"
                strokeWidth="1.2"
              />
              <circle cx="18" cy="16" r="3" fill="#00d4ff" />
            </svg>
          </div>
          <div className="topbar-brand-text">
            <div className="brand-primary-row">
              <strong>AERONOIX</strong>
            </div>
            <small>TWIN-PISTON AIRCRAFT ENGINE &nbsp;|&nbsp; DIGITAL TWIN</small>
          </div>
        </div>

        {/* Center: Mission Metadata Capsule */}
        <div className="topbar-capsule-bezel">
          <div className="capsule-pill">SIH 2026</div>
          <div className="capsule-divider" />
          <div className="capsule-pill">MALE UAV</div>
          <div className="capsule-divider" />
          <div className="capsule-pill">Twin-Piston</div>
          <div className="capsule-divider" />
          <div
            className={`capsule-pill font-mono tracking-wider font-semibold ${
              streamState.isConnected ? "text-emerald-400" : "text-cyan-400"
            }`}
          >
            DATA MODE: {streamState.isConnected ? "LIVE WEBSOCKET" : "SIMULATION (MOCK)"}
          </div>
        </div>

        {/* Right: Engine Status & Telemetry Readouts */}
        <div className="topbar-status-cluster">
          <div className="status-indicator-module">
            <i className={`status-led ${streamState.isConnected ? "led-green pulse-led" : "led-amber"}`} />
            <div className="status-module-text">
              <span>ENGINE STATUS</span>
              <strong className="status-label-healthy">
                {streamState.liveHealth?.degradation_state ?? "HEALTHY"}
              </strong>
            </div>
          </div>

          <div className="topbar-metric-box">
            <span>RPM</span>
            <strong>
              {rpmVal} <small>rpm</small>
            </strong>
          </div>

          <div className="topbar-metric-box">
            <span>OIL TEMP</span>
            <strong>
              {oilTempVal} <small>°C</small>
            </strong>
          </div>

          <div className="topbar-clock-box">
            <strong>{time.timeStr}</strong>
            <small>{time.dateStr}</small>
          </div>
        </div>
      </header>

      {/* ── Main Workspace ──────────────────────────────────────────────── */}
      <div className="chassis-inner-body">
        {/* Left Navigation Rail */}
        <Navigation active={activeNav} onChange={setActiveNav} />

        {/* Central & Right Content Area */}
        <div className="workspace-main-bay">
          {/* Main 3D Twin View */}
          {(activeNav === "3D View" || activeNav === "Exploded View") && (
            <div className="main-viewport-split">
              {/* 55–60% Hero Central Engine Viewport */}
              <div className="hero-engine-viewport">
                <EngineViewer
                  selectedId={selectedId}
                  onSelectComponent={handleComponentSelect}
                />
              </div>

              {/* 20–25% Right Component Intelligence Panel */}
              {showComponentPanel && selected && (
                <ComponentInfoPanel
                  component={selected}
                  components={engineComponents}
                  onSelect={handleComponentSelect}
                  onClose={() => setShowComponentPanel(false)}
                />
              )}
            </div>
          )}

          {/* Full Components Registry View */}
          {activeNav === "Components" && (
            <div className="dedicated-subview-panel">
              <ComponentsListModal
                components={engineComponents}
                selectedId={selectedId}
                onSelect={(id) => {
                  handleComponentSelect(id);
                  setActiveNav("3D View");
                }}
              />
            </div>
          )}

          {/* Dedicated Telemetry View */}
          {activeNav === "Telemetry" && (
            <div className="dedicated-subview-panel space-y-4">
              <div className="subview-header">
                <Activity className="subview-icon text-cyan-400" />
                <h3>HIGH-FREQUENCY SENSOR TELEMETRY, ELECTRICAL & COMBUSTION BUS</h3>
              </div>

              <ElectricalDiagnosticsCard data={streamState.liveHealth} />
              <EgtCombustionCard data={null} />

              <div className="telemetry-deep-grid">
                <div className="bottom-col">
                  <div className="telem-section">
                    <div className="telem-section-header">
                      <h2>LIVE TELEMETRY BUS</h2>
                      <span className="live-badge">
                        <i className="live-dot" /> Real-time
                      </span>
                    </div>
                    <div className="telemetry-grid">
                      {mockTelemetry.map((m) => (
                        <div className="telemetry-cell" key={m.id}>
                          <div className="metric-head">
                            <span>{m.label}</span>
                            <i className={`status-dot status-${m.status}`} />
                          </div>
                          <div className="metric-value">
                            <strong>{m.value}</strong>
                            <span>{m.unit}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="bottom-col">
                  <EngineTrendCharts telemetry={mockTelemetry} />
                </div>
              </div>
            </div>
          )}

          {/* Dedicated Fault Analysis View */}
          {activeNav === "Fault Analysis" && (
            <div className="dedicated-subview-panel">
              <FaultAnalysisView data={mockMlPrediction} liveFault={streamState.liveFault} />
            </div>
          )}

          {/* Dedicated RUL View */}
          {activeNav === "RUL" && (
            <div className="dedicated-subview-panel">
              <RULView data={mockMlPrediction} liveRul={streamState.liveRul} />
            </div>
          )}

          {/* Dedicated Mission Reliability View */}
          {activeNav === "Mission" && (
            <div className="dedicated-subview-panel">
              <MissionTimelineView data={mockMission} liveMission={streamState.liveMission} />
            </div>
          )}

          {/* Advisories & Q&A Panel */}
          {activeNav === "Advisories & Q&A" && <AdvisoriesPanel />}

          {/* Replay & What-If Panel */}
          {activeNav === "Replay & What-If" && <WhatIfReplayPanel />}

          {/* Settings View */}
          {activeNav === "Settings" && (
            <div className="dedicated-subview-panel">
              <div className="subview-header">
                <Shield className="subview-icon text-cyan-400" />
                <h3>SYSTEM SETTINGS & TELEMETRY STREAM PROTOCOLS</h3>
              </div>
              <div className="settings-options-list">
                <div className="settings-row">
                  <div>
                    <strong>FastAPI Backend Real-Time Stream URL</strong>
                    <small>ws://localhost:8000/api/v1/ws/engine</small>
                  </div>
                  <span
                    className={
                      streamState.isConnected
                        ? "bg-emerald-950 text-emerald-400 border border-emerald-500/40 px-2 py-1 rounded text-xs"
                        : "bg-amber-950 text-amber-300 border border-amber-500/40 px-2 py-1 rounded text-xs"
                    }
                  >
                    {streamState.isConnected ? "CONNECTED" : "RECONNECTING"}
                  </span>
                </div>
                <div className="settings-row">
                  <div>
                    <strong>FADEC Dual Channel Redundancy</strong>
                    <small>Enable auto fail-over from Lane A to Lane B on CRC error</small>
                  </div>
                  <span className="badge-normal">ENABLED</span>
                </div>
                <div className="settings-row">
                  <div>
                    <strong>Edge AI Model Precision</strong>
                    <small>Quantized INT8 inference at 1,000 Hz sample rate</small>
                  </div>
                  <span className="badge-normal">FP16 HIGH</span>
                </div>
                <div className="settings-row">
                  <div>
                    <strong>Telemetry Replay Buffer</strong>
                    <small>Retain 200 hours of continuous multi-spectral sensor logs</small>
                  </div>
                  <span className="badge-normal">ACTIVE (5.8 GB)</span>
                </div>
              </div>
            </div>
          )}

          {/* ── Bottom Information Strip ── */}
          <section className="dashboard-bottom-strip">
            {/* Bay 1: Live Telemetry */}
            <div className="instrument-bay bay-telemetry">
              <div className="bay-header">
                <div className="bay-title">
                  <span className="bay-bracket">&lt;</span>
                  <h4>LIVE TELEMETRY</h4>
                </div>
                <span className="live-pill">
                  <i className="live-dot pulse-dot" /> Real-time
                </span>
              </div>
              <div className="telem-values-grid">
                {mockTelemetry.map((m) => (
                  <div className="telem-item" key={m.id}>
                    <span>{m.label}</span>
                    <strong>
                      {m.value} <small>{m.unit}</small>
                    </strong>
                  </div>
                ))}
              </div>
            </div>

            {/* Bay 2: AI Prediction */}
            <div className="instrument-bay bay-ai">
              <div className="bay-header">
                <h4>AI PREDICTION</h4>
                <span className="model-chip">Module 13/14</span>
              </div>
              <div className="ai-values-grid">
                <div className="ai-item">
                  <span>Engine Health</span>
                  <strong className="value-cyan">
                    {streamState.liveHealth?.health_index !== undefined && streamState.liveHealth?.health_index !== null
                      ? (streamState.liveHealth.health_index * 100).toFixed(0)
                      : mockMlPrediction.healthIndex}
                    %
                  </strong>
                </div>
                <div className="ai-item">
                  <span>Anomaly Score</span>
                  <strong>
                    {streamState.liveFault?.confidence !== undefined && streamState.liveFault?.confidence !== null
                      ? streamState.liveFault.confidence.toFixed(2)
                      : mockMlPrediction.anomalyScore.toFixed(2)}
                  </strong>
                </div>
                <div className="ai-item">
                  <div className="ai-label-w-dot">
                    <i className="dot-cyan" />
                    <span>Fault Probability</span>
                  </div>
                  <strong>{mockMlPrediction.faultProbability}%</strong>
                </div>
                <div className="ai-item">
                  <span>Predicted Fault</span>
                  <strong
                    className={
                      (streamState.liveFault?.predicted_class ?? mockMlPrediction.predictedFault) === "NOMINAL" ||
                      (streamState.liveFault?.predicted_class ?? mockMlPrediction.predictedFault) === "None"
                        ? "value-normal"
                        : "value-caution"
                    }
                  >
                    {streamState.liveFault?.predicted_class ?? mockMlPrediction.predictedFault}
                  </strong>
                </div>
                <div className="ai-item">
                  <div className="ai-label-w-dot">
                    <Clock className="w-3 h-3 text-cyan-400 inline mr-1" />
                    <span>Predicted RUL</span>
                  </div>
                  <strong className="value-cyan">
                    {streamState.liveRul?.hours_remaining ?? mockMlPrediction.rul} h
                  </strong>
                </div>
                <div className="ai-item">
                  <span>Confidence</span>
                  <strong>{mockMlPrediction.confidence}%</strong>
                </div>
              </div>
            </div>

            {/* Bay 3: Mission Reliability */}
            <div className="instrument-bay bay-mission">
              <div className="bay-header">
                <h4>MISSION RELIABILITY</h4>
              </div>
              <div className="mission-values-grid">
                <div className="mission-row">
                  <span>Mission Success Probability</span>
                  <strong className="value-cyan">
                    {streamState.liveMission?.risk_score !== undefined && streamState.liveMission?.risk_score !== null
                      ? ((1 - streamState.liveMission.risk_score) * 100).toFixed(1)
                      : mockMission.successProbability}
                    %
                  </strong>
                </div>
                <div className="mission-progress-bar">
                  <div
                    className="m-fill"
                    style={{
                      width: `${
                        streamState.liveMission?.risk_score !== undefined && streamState.liveMission?.risk_score !== null
                          ? (1 - streamState.liveMission.risk_score) * 100
                          : mockMission.successProbability
                      }%`,
                    }}
                  />
                </div>
                <div className="mission-meta-row">
                  <div className="m-meta-item">
                    <i className="dot-green" />
                    <span>Mission Risk</span>
                    <strong className="text-emerald-400">
                      {streamState.liveMission?.risk_level ?? mockMission.risk}
                    </strong>
                  </div>
                  <div className="m-meta-item">
                    <i className="dot-green" />
                    <span>Est. Safe Operating Time</span>
                    <strong>{streamState.liveMission?.rul_hours ?? mockMission.safeOperatingTime} h</strong>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ── Lower Row: Engine Health Donut, Trends, Fault Analysis ── */}
          <section className="dashboard-lower-row">
            {/* Card 1: Engine Health Donut */}
            <ComponentHealthGauge
              health={
                streamState.liveHealth?.health_index !== undefined && streamState.liveHealth?.health_index !== null
                  ? Math.round(streamState.liveHealth.health_index * 100)
                  : mockMlPrediction.healthIndex
              }
              normalCount={healthCounts.normal}
              cautionCount={healthCounts.caution}
              criticalCount={healthCounts.critical}
            />

            {/* Card 2: Key Telemetry Trends Sparklines */}
            <div className="lower-trends-bay">
              <EngineTrendCharts telemetry={mockTelemetry} />
            </div>

            {/* Card 3: Fault Analysis Mini Breakdown */}
            <div className="lower-fault-bay">
              <div className="fault-bay-head">
                <h4>FAULT ANALYSIS</h4>
                <button
                  type="button"
                  className="view-all-link"
                  onClick={() => setActiveNav("Fault Analysis")}
                >
                  View All
                </button>
              </div>
              <div className="fault-bay-body">
                <div className="fault-big-stat">
                  <strong>{mockMlPrediction.faultProbability}%</strong>
                  <small>Fault Probability</small>
                </div>
                <div className="fault-mini-list">
                  <div className="f-mini-row">
                    <i className="f-dot" />
                    <span>Injector</span>
                    <strong>2%</strong>
                  </div>
                  <div className="f-mini-row">
                    <i className="f-dot" />
                    <span>Valve</span>
                    <strong>1%</strong>
                  </div>
                  <div className="f-mini-row">
                    <i className="f-dot" />
                    <span>Sensor</span>
                    <strong>1%</strong>
                  </div>
                  <div className="f-mini-row">
                    <i className="f-dot" />
                    <span>Others</span>
                    <strong>1%</strong>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
