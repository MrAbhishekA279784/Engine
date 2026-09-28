import type { MLPrediction } from "@/types/digital-twin";

const STATUS_COLOR: Record<string, string> = {
  NORMAL: "anomaly-normal",
  WARNING: "anomaly-warning",
  ANOMALY: "anomaly-anomaly",
  CRITICAL: "anomaly-critical",
};

export function AnomalyDetectionPanel({ data }: { data: MLPrediction }) {
  const hasAnomaly = data.anomalyStatus !== "NORMAL";

  return (
    <section className="ml-panel" aria-labelledby="anomaly-title">
      <div className="ml-panel-header">
        <h3 id="anomaly-title">ANOMALY DETECTION</h3>
      </div>
      <div className="anomaly-status-row">
        <span className={`anomaly-badge ${STATUS_COLOR[data.anomalyStatus]}`}>{data.anomalyStatus}</span>
        <span className="anomaly-score-label">Score: <strong>{data.anomalyScore.toFixed(2)}</strong></span>
      </div>

      {!hasAnomaly && (
        <p className="anomaly-clear">No significant deviation detected.</p>
      )}

      {hasAnomaly && data.anomalyComponent && (
        <div className="anomaly-detail">
          <div className="ml-row">
            <span>Component</span>
            <strong>{data.anomalyComponent}</strong>
          </div>
          <div className="ml-row">
            <span>Severity</span>
            <strong className={data.anomalyStatus === "WARNING" ? "value-caution" : "value-critical"}>{data.anomalySeverity}</strong>
          </div>
          {data.contributingSignals.length > 0 && (
            <div className="anomaly-signals">
              <span className="signals-label">Primary signals:</span>
              <ul>
                {data.contributingSignals.map((sig) => (
                  <li key={sig}>{sig}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
