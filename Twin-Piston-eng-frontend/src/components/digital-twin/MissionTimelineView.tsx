import { useState } from "react";
import type { MissionData } from "@/types/digital-twin";
import type { MissionRiskResponse, FlightPhase } from "@/types/backend-api";
import { Compass, ShieldCheck, Plane, CheckCircle2, AlertTriangle } from "lucide-react";

export function MissionTimelineView({
  data,
  liveMission,
}: {
  data: MissionData;
  liveMission?: MissionRiskResponse | null;
}) {
  const [activePhase, setActivePhase] = useState<FlightPhase>(
    (liveMission?.flight_phase as FlightPhase) || "CRUISE"
  );

  const phases: { name: FlightPhase; label: string; duration: string; stress: string }[] = [
    { name: "GROUND", label: "GROUND", duration: "0:15", stress: "Low" },
    { name: "TAKEOFF", label: "TAKEOFF", duration: "0:10", stress: "High" },
    { name: "CLIMB", label: "CLIMB", duration: "0:25", stress: "High" },
    { name: "CRUISE", label: "CRUISE", duration: "2:45", stress: "Moderate" },
    { name: "DESCENT", label: "DESCENT", duration: "0:30", stress: "Low" },
    { name: "LANDING", label: "LANDING", duration: "0:15", stress: "Moderate" },
  ];

  const riskScore = liveMission?.risk_score ?? 0.03;
  const riskLevel = liveMission?.risk_level ?? data.risk;
  const safeTime = liveMission?.rul_hours ?? data.safeOperatingTime;

  return (
    <div className="mission-full-view space-y-6" aria-labelledby="mission-title">
      <div className="mission-header flex justify-between items-center">
        <div className="m-title-group flex items-center gap-3">
          <Compass className="m-icon text-cyan-400" />
          <div>
            <h3 id="mission-title">MISSION RELIABILITY & PHASE RISK MATRIX</h3>
            <small className="text-cyan-400/80">
              Module 16 & 20 Flight Envelope Risk Assessment & Mission Continuity Analysis
            </small>
          </div>
        </div>
        <div className="m-risk-badge flex items-center gap-2">
          <span>Mission Success Probability:</span>
          <strong className="text-emerald-400 font-extrabold text-sm">
            {((1 - riskScore) * 100).toFixed(1)}%
          </strong>
        </div>
      </div>

      {/* Mission Timeline Horizontal Track */}
      <div className="timeline-track-card fa-card">
        <h4>MISSION FLIGHT PHASES (CLICK TO SELECT PHASE)</h4>
        <div className="timeline-phases-list flex items-center justify-between mt-4 overflow-x-auto gap-2">
          {phases.map((p, i) => {
            const isSelected = activePhase === p.name;
            return (
              <button
                type="button"
                key={p.name}
                onClick={() => setActivePhase(p.name)}
                className={`timeline-phase-node flex-1 min-w-[100px] p-2.5 rounded border transition text-left cursor-pointer ${
                  isSelected
                    ? "bg-cyan-950/80 border-cyan-400 text-cyan-100 shadow-[0_0_12px_rgba(0,212,255,0.3)]"
                    : "bg-black/50 border-cyan-500/20 text-gray-400 hover:border-cyan-500/50"
                }`}
              >
                <div className="phase-indicator flex items-center gap-1 mb-1 text-xs">
                  <span className="phase-num font-bold text-cyan-400">{i + 1}.</span>
                  <span className="font-semibold text-xs">{p.label}</span>
                </div>
                <div className="phase-info text-[11px]">
                  <span>{p.duration} ({p.stress})</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="mission-kpis-grid grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="m-card bg-black/40 border border-cyan-500/20 p-4 rounded text-xs space-y-1">
          <span className="text-gray-400 block">Safe Operating Flight Time</span>
          <strong className="m-big-val text-2xl text-cyan-400 block font-bold">{safeTime} h</strong>
          <small className="text-gray-400 italic">Fuel reserve & thermal margin verified</small>
        </div>
        <div className="m-card bg-black/40 border border-emerald-500/20 p-4 rounded text-xs space-y-1">
          <span className="text-gray-400 block">Active Phase Risk ({activePhase})</span>
          <strong className="m-big-val text-2xl text-emerald-400 block font-bold">{riskLevel}</strong>
          <small className="text-emerald-400/80 italic">Risk Score: {(riskScore * 100).toFixed(1)}%</small>
        </div>
        <div className="m-card bg-black/40 border border-cyan-500/20 p-4 rounded text-xs space-y-1">
          <span className="text-gray-400 block">Engine Condition Rating</span>
          <strong className="m-big-val text-2xl text-cyan-300 block font-bold">{data.engineCondition}</strong>
          <small className="text-gray-400 italic">Continuous telemetry monitoring active</small>
        </div>
      </div>
    </div>
  );
}
