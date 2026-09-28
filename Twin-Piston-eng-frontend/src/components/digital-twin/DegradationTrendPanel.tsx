type Props = {
  healthHistory: number[];
  currentHealth: number;
  degradationState: string;
  operatingHours?: number;
};

function DegradationChart({ healthHistory }: { healthHistory: number[] }) {
  const hours = [0, 25, 50, 75, 100, 125, 150, 175, 200];
  const W = 200;
  const H = 60;
  const pad = { l: 28, r: 8, t: 6, b: 18 };
  const chartW = W - pad.l - pad.r;
  const chartH = H - pad.t - pad.b;

  const points = healthHistory.slice(0, hours.length).map((val, i) => {
    const x = pad.l + (i / (hours.length - 1)) * chartW;
    const y = pad.t + ((100 - val) / 100) * chartH;
    return `${x},${y}`;
  });

  const xLabels = [0, 100, 200];
  const yLabels = [0, 50, 100];

  return (
    <svg className="degradation-chart" viewBox={`0 0 ${W} ${H}`} aria-label="Health degradation trend">
      {/* Grid lines */}
      {yLabels.map((y) => {
        const cy = pad.t + ((100 - y) / 100) * chartH;
        return (
          <g key={y}>
            <line x1={pad.l} y1={cy} x2={W - pad.r} y2={cy} className="chart-grid-line" />
            <text x={pad.l - 3} y={cy + 3} className="chart-label-y">{y}</text>
          </g>
        );
      })}
      {/* X axis */}
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} className="chart-axis-line" />
      {xLabels.map((h) => {
        const cx = pad.l + (h / 200) * chartW;
        return (
          <text key={h} x={cx} y={H - 4} className="chart-label-x">{h}h</text>
        );
      })}
      {/* Trend line */}
      <polyline points={points.join(" ")} className="degrad-trend-line" />
      {/* Current point */}
      {points[points.length - 1] && (() => {
        const last = points[points.length - 1]!.split(",");
        const x = parseFloat(last[0]!);
        const y = parseFloat(last[1]!);
        return <circle cx={x} cy={y} r="2.5" className="degrad-trend-dot" />;
      })()}
    </svg>
  );
}

export function DegradationTrendPanel({ healthHistory, currentHealth, degradationState, operatingHours = 200 }: Props) {
  return (
    <section className="ml-panel" aria-labelledby="degrad-title">
      <div className="ml-panel-header">
        <h3 id="degrad-title">DEGRADATION TREND</h3>
        <span className={`degrad-state-badge ${degradationState === "Stable" ? "ds-stable" : degradationState === "Gradual" ? "ds-gradual" : "ds-critical"}`}>
          {degradationState}
        </span>
      </div>
      <DegradationChart healthHistory={healthHistory} />
      <div className="degrad-stats">
        <div className="ml-row">
          <span>Current Health</span>
          <strong className={currentHealth >= 90 ? "value-normal" : currentHealth >= 80 ? "value-caution" : "value-critical"}>{currentHealth}%</strong>
        </div>
        <div className="ml-row">
          <span>Operating Hours</span>
          <strong>{operatingHours} h</strong>
        </div>
        <div className="ml-row">
          <span>Trend</span>
          <strong className={degradationState === "Stable" ? "value-normal" : "value-caution"}>{degradationState}</strong>
        </div>
      </div>
    </section>
  );
}
