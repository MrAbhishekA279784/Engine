import type { MissionData } from "@/types/digital-twin";

const RISK_CLASS: Record<string, string> = {
  LOW: "value-normal",
  MEDIUM: "value-caution",
  HIGH: "value-critical",
  CRITICAL: "value-critical",
};

export function MissionReliability({ data }: { data: MissionData }) {
  return (
    <section className="ml-panel mission-reliability-panel" aria-labelledby="mission-title">
      <div className="ml-panel-header">
        <h3 id="mission-title">MISSION RELIABILITY</h3>
      </div>
      <div className="ml-grid">
        <div className="ml-row ml-row-accent">
          <span>Success Probability</span>
          <strong className="value-normal">{data.successProbability}%</strong>
        </div>
        <div className="ml-row">
          <span>Mission Risk</span>
          <span className={`risk-badge ${RISK_CLASS[data.risk]}`}>{data.risk}</span>
        </div>
        <div className="ml-row">
          <span>Safe Operating Time</span>
          <strong>{data.safeOperatingTime} h</strong>
        </div>
        <div className="ml-row">
          <span>Engine Condition</span>
          <strong className={data.status === "normal" ? "value-normal" : "value-caution"}>{data.engineCondition}</strong>
        </div>
        <div className="ml-row">
          <span>Fault Probability</span>
          <strong className={data.faultProbability > 15 ? "value-caution" : ""}>{data.faultProbability}%</strong>
        </div>
        <div className="ml-row">
          <span>Degradation Rate</span>
          <strong>{data.degradationRate}</strong>
        </div>
      </div>
    </section>
  );
}
