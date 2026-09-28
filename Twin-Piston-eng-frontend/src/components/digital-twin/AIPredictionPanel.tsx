import type { MLPrediction } from "@/types/digital-twin";

export function AIPredictionPanel({ data }: { data: MLPrediction }) {
  return (
    <section className="ml-panel" aria-labelledby="ai-pred-title">
      <div className="ml-panel-header">
        <h3 id="ai-pred-title">AI PREDICTION</h3>
        <span className="model-badge">Model v2.4</span>
      </div>
      <div className="ml-grid">
        <div className="ml-row ml-row-accent">
          <span>Health Index</span>
          <div className="ml-bar-cell">
            <div className="ml-bar-track">
              <div className="ml-bar-fill ml-bar-normal" style={{ width: `${data.healthIndex}%` }} />
            </div>
            <strong className="value-normal">{data.healthIndex}%</strong>
          </div>
        </div>
        <div className="ml-row">
          <span>Anomaly Score</span>
          <strong className={data.anomalyScore > 0.5 ? "value-caution" : ""}>{data.anomalyScore.toFixed(2)}</strong>
        </div>
        <div className="ml-row ml-row-accent">
          <span>Fault Probability</span>
          <div className="ml-bar-cell">
            <div className="ml-bar-track">
              <div className={`ml-bar-fill ${data.faultProbability > 20 ? "ml-bar-caution" : "ml-bar-low"}`} style={{ width: `${Math.min(data.faultProbability * 2, 100)}%` }} />
            </div>
            <strong>{data.faultProbability}%</strong>
          </div>
        </div>
        <div className="ml-row">
          <span>Predicted Fault</span>
          <strong className="value-normal">{data.predictedFault}</strong>
        </div>
        <div className="ml-row">
          <span>Confidence</span>
          <strong>{data.confidence}%</strong>
        </div>
        <div className="ml-row">
          <span>Degradation</span>
          <strong className={data.degradationState === "Stable" ? "value-normal" : "value-caution"}>{data.degradationState}</strong>
        </div>
        <div className="ml-row ml-row-accent">
          <span>RUL</span>
          <strong className="rul-value">{data.rul} h</strong>
        </div>
        <div className="ml-row">
          <span>RUL Confidence</span>
          <strong>{data.rulConfidence}%</strong>
        </div>
      </div>
    </section>
  );
}
