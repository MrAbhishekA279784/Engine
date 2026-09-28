import type { MLPrediction } from "@/types/digital-twin";
import type { RULResponse } from "@/types/backend-api";
import { Clock, TrendingDown, Target, Activity, ShieldCheck, AlertTriangle } from "lucide-react";

export function RULView({
  data,
  liveRul,
}: {
  data: MLPrediction;
  liveRul?: RULResponse | null;
}) {
  const hoursRemaining = liveRul?.hours_remaining ?? data.rul;
  const lowerBound = liveRul?.lower_bound_hours ?? data.rulRange[0];
  const upperBound = liveRul?.upper_bound_hours ?? data.rulRange[1];
  const validated = liveRul?.validated ?? true;

  return (
    <div className="rul-full-view space-y-6" aria-labelledby="rul-title">
      <div className="rul-header flex justify-between items-center">
        <div className="rul-title-group flex items-center gap-3">
          <Clock className="rul-icon text-cyan-400" />
          <div>
            <h3 id="rul-title">REMAINING USEFUL LIFE (RUL) PROGNOSIS</h3>
            <small className="text-cyan-400/80">
              Module 15 Weibull Degradation, LSTM Recurrent Neural Network & Lifetime Validation
            </small>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="rul-confidence-badge">
            <span>Prognosis Confidence:</span>
            <strong>{data.rulConfidence}%</strong>
          </div>
          <span
            className={`px-2 py-1 text-xs font-bold rounded flex items-center gap-1 ${
              validated
                ? "bg-emerald-950 text-emerald-400 border border-emerald-500/40"
                : "bg-amber-950 text-amber-300 border border-amber-500/40"
            }`}
          >
            {validated ? <ShieldCheck className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
            {validated ? "VALIDATED PROGNOSIS" : "UNVALIDATED"}
          </span>
        </div>
      </div>

      <div className="rul-grid-layout grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main RUL Display Card */}
        <div className="rul-card rul-hero-card">
          <div className="rul-hero-readout text-center py-4">
            <span className="rul-hero-label text-xs text-gray-400 block mb-1">Estimated Remaining Useful Life</span>
            <div className="rul-hero-number flex items-baseline justify-center gap-2">
              <strong className="text-4xl text-cyan-400 font-extrabold">{hoursRemaining}</strong>
              <small className="text-gray-400 font-mono">FLIGHT HOURS</small>
            </div>
            <div className="rul-range-pill mt-3 inline-block bg-cyan-950/60 border border-cyan-500/30 px-3 py-1 rounded text-xs">
              <span className="text-gray-400">90% Calibrated Interval: </span>
              <strong className="text-cyan-300">{lowerBound} h – {upperBound} h</strong>
            </div>
          </div>

          <div className="rul-hero-meta space-y-2 pt-3 border-t border-cyan-500/20 text-xs">
            <div className="r-stat flex justify-between">
              <span className="text-gray-400">Degradation Trend</span>
              <strong className="value-normal text-emerald-400">{liveRul?.trend ?? "STABLE (0.18% / 10h)"}</strong>
            </div>
            <div className="r-stat flex justify-between">
              <span className="text-gray-400">Operating Assumption</span>
              <strong className="text-cyan-300">{liveRul?.operating_assumption ?? "NOMINAL_CRUISE_PROFILE"}</strong>
            </div>
            <div className="r-stat flex justify-between">
              <span className="text-gray-400">Time to Next Overhaul (TBO)</span>
              <strong className="text-cyan-200">1,800 h</strong>
            </div>
          </div>
        </div>

        {/* Degradation Trajectory Curve */}
        <div className="rul-card lg:col-span-2">
          <h4>HEALTH DEGRADATION & RUL TRAJECTORY</h4>
          <div className="rul-trajectory-chart mt-3">
            <svg viewBox="0 0 500 120" className="rul-svg w-full" aria-label="RUL Trajectory Chart">
              {/* Grid lines */}
              <line x1="40" y1="20" x2="480" y2="20" className="chart-grid-line" stroke="#1e293b" />
              <line x1="40" y1="50" x2="480" y2="50" className="chart-grid-line" stroke="#1e293b" />
              <line x1="40" y1="80" x2="480" y2="80" className="chart-grid-line" stroke="#1e293b" />
              <line x1="40" y1="100" x2="480" y2="100" className="chart-axis-line" stroke="#334155" />

              {/* Confidence Band (Polygon) */}
              <polygon
                points="40,22 140,28 220,38 320,52 400,70 480,98 480,105 400,82 320,62 220,44 140,32 40,24"
                fill="rgba(0, 212, 255, 0.12)"
              />

              {/* Historical Health Curve (Solid) */}
              <polyline
                points="40,22 80,24 110,26 140,29"
                stroke="#00d4ff"
                strokeWidth="2.5"
                fill="none"
              />

              {/* Predicted Degradation Path (Dashed) */}
              <polyline
                points="140,29 220,40 300,54 380,72 440,88 480,100"
                stroke="#00d4ff"
                strokeWidth="2"
                strokeDasharray="4 3"
                fill="none"
              />

              {/* Current Operating Point */}
              <circle cx="140" cy="29" r="4" fill="#00d4ff" />
              <text x="140" y="20" fill="#00d4ff" fontSize="9" textAnchor="middle">Current (200h)</text>

              {/* Axis Labels */}
              <text x="35" y="24" fill="#6a7d8f" fontSize="8" textAnchor="end">100%</text>
              <text x="35" y="54" fill="#6a7d8f" fontSize="8" textAnchor="end">80%</text>
              <text x="35" y="84" fill="#6a7d8f" fontSize="8" textAnchor="end">60%</text>
              <text x="35" y="104" fill="#6a7d8f" fontSize="8" textAnchor="end">EoL</text>

              <text x="40" y="114" fill="#6a7d8f" fontSize="8">0h</text>
              <text x="140" y="114" fill="#6a7d8f" fontSize="8">200h</text>
              <text x="240" y="114" fill="#6a7d8f" fontSize="8">400h</text>
              <text x="340" y="114" fill="#6a7d8f" fontSize="8">600h</text>
              <text x="440" y="114" fill="#6a7d8f" fontSize="8">800h</text>
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
