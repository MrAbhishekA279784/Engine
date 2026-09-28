export function ComponentHealthGauge({
  health = 92,
  normalCount = 22,
  cautionCount = 3,
  criticalCount = 1,
}: {
  health?: number;
  normalCount?: number;
  cautionCount?: number;
  criticalCount?: number;
}) {
  const radius = 38;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (health / 100) * circumference;

  return (
    <div className="engine-health-card">
      <div className="health-card-head">
        <span>ENGINE HEALTH</span>
      </div>
      <div className="health-card-body">
        <div className="donut-gauge-wrap">
          <svg height={radius * 2} width={radius * 2} className="donut-svg">
            <circle
              stroke="rgba(255, 255, 255, 0.08)"
              fill="transparent"
              strokeWidth={stroke}
              r={normalizedRadius}
              cx={radius}
              cy={radius}
            />
            <circle
              stroke="#00d4ff"
              fill="transparent"
              strokeWidth={stroke}
              strokeDasharray={circumference + " " + circumference}
              style={{ strokeDashoffset }}
              strokeLinecap="round"
              r={normalizedRadius}
              cx={radius}
              cy={radius}
              className="donut-progress"
            />
          </svg>
          <div className="donut-center-label">
            <strong>{health}%</strong>
            <small>Overall</small>
          </div>
        </div>

        <div className="health-breakdown-list">
          <div className="health-stat-pill">
            <i className="h-dot dot-normal" />
            <span>Normal</span>
            <strong>{normalCount}</strong>
          </div>
          <div className="health-stat-pill">
            <i className="h-dot dot-caution" />
            <span>Caution</span>
            <strong>{cautionCount}</strong>
          </div>
          <div className="health-stat-pill">
            <i className="h-dot dot-critical" />
            <span>Critical</span>
            <strong>{criticalCount}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}
