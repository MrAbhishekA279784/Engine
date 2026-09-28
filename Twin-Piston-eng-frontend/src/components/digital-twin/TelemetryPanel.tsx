import type { TelemetryMetric } from "@/types/digital-twin";
import type { MLPrediction } from "@/types/digital-twin";

function Sparkline({ points, caution }: { points: number[]; caution: boolean }) {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const coords = points.map((v, i) => `${(i / (points.length - 1)) * 100},${26 - ((v - min) / span) * 20}`).join(" ");
  return (
    <svg className={caution ? "sparkline sparkline-caution" : "sparkline"} viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true">
      <path d="M0 27 H100" />
      <polyline points={coords} />
    </svg>
  );
}

type Props = { telemetry: TelemetryMetric[]; mlData: MLPrediction };

export function TelemetryPanel({ telemetry, mlData }: Props) {
  const liveMetrics = ["rpm", "egt", "cht", "oil-temp", "oil-pressure", "fuel", "altitude", "vibration"];
  const live = liveMetrics.map(id => telemetry.find(t => t.id === id)).filter(Boolean) as TelemetryMetric[];

  return (
    <section className="telemetry-band" aria-labelledby="telemetry-title">
      {/* Live Telemetry */}
      <div className="telem-section">
        <div className="telem-section-header">
          <h2 id="telemetry-title">LIVE TELEMETRY</h2>
          <span className="live-badge"><i className="live-dot" />Real-time</span>
        </div>
        <div className="telemetry-grid">
          {live.map((metric) => (
            <article className="telemetry-cell" key={metric.id}>
              <div className="metric-head">
                <span>{metric.label}</span>
                <i className={`status-dot status-${metric.status}`} />
              </div>
              <div className="metric-value">
                <strong>{metric.value}</strong>
                <span>{metric.unit}</span>
              </div>
              <Sparkline points={metric.trend} caution={metric.status === "caution"} />
            </article>
          ))}
        </div>
      </div>

      {/* AI Output — separated from live */}
      <div className="telem-section telem-ai-section">
        <div className="telem-section-header">
          <h2>AI OUTPUT</h2>
          <span className="model-badge">Model v2.4</span>
        </div>
        <div className="ai-output-grid">
          <div className="ai-output-row">
            <span>Health Index</span>
            <strong className="value-normal">{mlData.healthIndex}%</strong>
          </div>
          <div className="ai-output-row">
            <span>Anomaly Score</span>
            <strong>{mlData.anomalyScore.toFixed(2)}</strong>
          </div>
          <div className="ai-output-row">
            <span>Fault Probability</span>
            <strong>{mlData.faultProbability}%</strong>
          </div>
          <div className="ai-output-row">
            <span>RUL</span>
            <strong>{mlData.rul} h</strong>
          </div>
          <div className="ai-output-row">
            <span>Confidence</span>
            <strong>{mlData.confidence}%</strong>
          </div>
        </div>
      </div>
    </section>
  );
}
