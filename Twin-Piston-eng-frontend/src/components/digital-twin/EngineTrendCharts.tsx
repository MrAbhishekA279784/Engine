import type { TelemetryMetric } from "@/types/digital-twin";

type ChartProps = { points: number[]; label: string; unit: string; caution: boolean; height?: number };

function MiniChart({ points, label, unit, caution, height = 44 }: ChartProps) {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;
  const W = 100;
  const H = height;
  const padL = 0;
  const padB = 14;
  const padT = 4;
  const chartH = H - padB - padT;

  const coords = points.map((v, i) => {
    const x = padL + (i / (points.length - 1)) * (W - padL);
    const y = padT + ((max - v) / span) * chartH;
    return `${x},${y}`;
  }).join(" ");

  const last = points[points.length - 1];
  const firstVal = min < 1000 ? min.toFixed(min < 10 ? 1 : 0) : (min / 1000).toFixed(1) + "k";
  const lastVal = max < 1000 ? max.toFixed(max < 10 ? 1 : 0) : (max / 1000).toFixed(1) + "k";

  return (
    <div className="mini-chart-card">
      <div className="mini-chart-header">
        <span className="mini-chart-label">{label}</span>
        <span className={`mini-chart-value ${caution ? "value-caution" : ""}`}>{typeof last === "number" && last < 1000 ? last.toFixed(last < 10 ? 1 : 0) : last} <em>{unit}</em></span>
      </div>
      <svg className={`mini-chart-svg ${caution ? "mini-chart-caution" : ""}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        {/* Grid */}
        <line x1={0} y1={padT} x2={W} y2={padT} className="chart-grid-line" />
        <line x1={0} y1={padT + chartH / 2} x2={W} y2={padT + chartH / 2} className="chart-grid-line" />
        {/* Baseline */}
        <line x1={0} y1={padT + chartH} x2={W} y2={padT + chartH} className="chart-axis-line" />
        {/* Trend */}
        <polyline points={coords} className="mini-chart-line" />
        {/* Range labels */}
        <text x={1} y={H - 2} className="chart-label-x">{firstVal}</text>
        <text x={W - 1} y={H - 2} textAnchor="end" className="chart-label-x">{lastVal}</text>
      </svg>
    </div>
  );
}

export function EngineTrendCharts({ telemetry }: { telemetry: TelemetryMetric[] }) {
  const chartOrder = ["egt", "cht", "oil-temp", "oil-pressure", "vibration", "fuel"];
  const ordered = chartOrder.map(id => telemetry.find(t => t.id === id)).filter(Boolean) as TelemetryMetric[];

  return (
    <section className="trend-charts-section" aria-labelledby="trends-title">
      <div className="ml-panel-header">
        <h3 id="trends-title">KEY TELEMETRY TRENDS</h3>
      </div>
      <div className="trend-charts-grid">
        {ordered.map(metric => (
          <MiniChart
            key={metric.id}
            points={metric.trend}
            label={`${metric.label} (${metric.unit})`}
            unit={metric.unit}
            caution={metric.status === "caution"}
          />
        ))}
      </div>
    </section>
  );
}
