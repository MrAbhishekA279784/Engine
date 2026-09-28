type Props = {
  rul: number;
  rulConfidence: number;
  rulRange: [number, number];
  trend?: string;
};

export function RULPanel({ rul, rulConfidence, rulRange, trend = "Stable" }: Props) {
  return (
    <section className="ml-panel" aria-labelledby="rul-title">
      <div className="ml-panel-header">
        <h3 id="rul-title">REMAINING USEFUL LIFE</h3>
        <span className="model-badge">MODEL PREDICTION</span>
      </div>
      <div className="rul-display">
        <span className="rul-value-large">{rul}</span>
        <span className="rul-unit">h</span>
      </div>
      <div className="ml-grid">
        <div className="ml-row">
          <span>Confidence</span>
          <strong>{rulConfidence}%</strong>
        </div>
        <div className="ml-row">
          <span>Expected Range</span>
          <strong>{rulRange[0]} – {rulRange[1]} h</strong>
        </div>
        <div className="ml-row">
          <span>Trend</span>
          <strong className={trend === "Stable" ? "value-normal" : "value-caution"}>{trend}</strong>
        </div>
      </div>
    </section>
  );
}
