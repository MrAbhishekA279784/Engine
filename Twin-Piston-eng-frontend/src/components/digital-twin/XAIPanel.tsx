import type { XAIFactor } from "@/types/digital-twin";

type Props = {
  healthIndex?: number;
  factors: XAIFactor[];
  assessment: string;
};

export function XAIPanel({ healthIndex, factors, assessment }: Props) {
  return (
    <section className="ml-panel" aria-labelledby="xai-title">
      <div className="ml-panel-header">
        <h3 id="xai-title">AI EXPLANATION</h3>
      </div>
      {healthIndex !== undefined && (
        <div className="xai-headline">
          <span>Health Index:</span>
          <strong className={healthIndex >= 90 ? "value-normal" : "value-caution"}>{healthIndex}%</strong>
        </div>
      )}
      <div className="xai-factors-label">Main contributing factors:</div>
      <div className="xai-factors-list">
        {factors.map((f) => (
          <div key={f.signal} className="xai-factor-row">
            <span className="xai-signal">{f.signal}</span>
            <span className={`xai-delta ${f.direction === "up" ? "xai-up" : f.direction === "down" ? "xai-down" : ""}`}>
              {f.delta}
            </span>
          </div>
        ))}
      </div>
      <p className="xai-assessment">{assessment}</p>
    </section>
  );
}
