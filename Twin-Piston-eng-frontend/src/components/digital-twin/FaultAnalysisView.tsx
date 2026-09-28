import type { MLPrediction } from "@/types/digital-twin";
import type { FaultClassificationResponse } from "@/types/backend-api";
import { AlertOctagon, CheckCircle2, Cpu, Wrench, ShieldAlert } from "lucide-react";

export function FaultAnalysisView({
  data,
  liveFault,
}: {
  data: MLPrediction;
  liveFault?: FaultClassificationResponse | null;
}) {
  const faultTaxonomy = [
    { label: "NOMINAL (Normal Operation)", key: "NOMINAL", prob: liveFault?.probabilities?.NOMINAL ?? 94 },
    { label: "CYLINDER MISFIRE", key: "MISFIRE", prob: liveFault?.probabilities?.MISFIRE ?? 2 },
    { label: "DETONATION / KNOCK", key: "DETONATION_KNOCK", prob: liveFault?.probabilities?.DETONATION_KNOCK ?? 0.5 },
    { label: "EXHAUST VALVE LEAK", key: "EXHAUST_VALVE_LEAK", prob: liveFault?.probabilities?.EXHAUST_VALVE_LEAK ?? 1 },
    { label: "INTAKE / BOOST LEAK", key: "INTAKE_BOOST_LEAK", prob: liveFault?.probabilities?.INTAKE_BOOST_LEAK ?? 0.5 },
    { label: "OIL DEGRADATION", key: "OIL_DEGRADATION", prob: liveFault?.probabilities?.OIL_DEGRADATION ?? 1 },
    { label: "COOLING FAULT / OVERHEAT", key: "COOLING_FAULT", prob: liveFault?.probabilities?.COOLING_FAULT ?? 0.5 },
    { label: "BEARING WEAR", key: "BEARING_WEAR", prob: liveFault?.probabilities?.BEARING_WEAR ?? 0.3 },
    { label: "SENSOR FAULT", key: "SENSOR_FAULT", prob: liveFault?.probabilities?.SENSOR_FAULT ?? 0.2 },
  ];

  const predictedClass = liveFault?.predicted_class ?? data.predictedFault ?? "NOMINAL";

  return (
    <div className="fault-analysis-full-view space-y-6" aria-labelledby="fa-title">
      <div className="fa-header flex justify-between items-center">
        <div className="fa-title-group flex items-center gap-3">
          <AlertOctagon className="fa-icon text-cyan-400" />
          <div>
            <h3 id="fa-title">AI 9-TAXONOMY FAULT ANALYSIS & RULE MATRIX</h3>
            <small className="text-cyan-400/80">
              Module 13 & 20 Deep Learning Classifier, Rule-Based Evidence & Statistical Residuals
            </small>
          </div>
        </div>
        <div className="fa-status-pill">
          <span>Predicted Class:</span>
          <strong
            className={
              predictedClass === "NOMINAL" || predictedClass === "None"
                ? "badge-normal"
                : "bg-red-950 text-red-300 border border-red-500/50 px-2 py-0.5 rounded"
            }
          >
            {predictedClass} ({data.anomalyScore.toFixed(2)})
          </strong>
        </div>
      </div>

      <div className="fa-grid-layout grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 9-Class Probability Density */}
        <div className="fa-card">
          <h4>9-CLASS FAULT PROBABILITY DENSITY</h4>
          <div className="fa-prob-display mt-3">
            <div className="fa-big-percent mb-4">
              <strong>{data.faultProbability}%</strong>
              <small>Composite Fault Risk</small>
            </div>
            <div className="fa-prob-bar-list space-y-2">
              {faultTaxonomy.map((item) => (
                <div key={item.key} className="fa-prob-item">
                  <div className="fa-prob-label text-xs flex justify-between mb-0.5">
                    <span className={item.key === predictedClass ? "text-cyan-300 font-bold" : "text-gray-300"}>
                      {item.label}
                    </span>
                    <strong className="text-cyan-400">{(item.prob * (item.prob <= 1 ? 100 : 1)).toFixed(1)}%</strong>
                  </div>
                  <div className="fa-bar-track h-2 bg-gray-900 rounded overflow-hidden">
                    <div
                      className={`fa-bar-fill h-full transition-all duration-300 ${
                        item.key === "NOMINAL"
                          ? "bg-emerald-500"
                          : item.key === predictedClass
                          ? "bg-red-500"
                          : "bg-cyan-500/70"
                      }`}
                      style={{ width: `${Math.min(100, item.prob * (item.prob <= 1 ? 100 : 1))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Contributing Parameter Matrix & Condition Flags */}
        <div className="fa-card space-y-4">
          <h4>CONTRIBUTING PARAMETERS & CONDITION FLAGS</h4>
          <div className="fa-matrix-table-wrap overflow-x-auto">
            <table className="fa-matrix-table w-full text-xs">
              <thead>
                <tr>
                  <th>Signal / Symptom</th>
                  <th>Observed</th>
                  <th>Nominal</th>
                  <th>Z-Score</th>
                  <th>Contribution</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>CHT Cylinder 2</td>
                  <td>197 °C</td>
                  <td>172 °C</td>
                  <td>+1.42</td>
                  <td className="value-caution">46%</td>
                </tr>
                <tr>
                  <td>EGT Thermal Spread</td>
                  <td>46 °C</td>
                  <td>28 °C</td>
                  <td>+1.18</td>
                  <td className="value-caution">32%</td>
                </tr>
                <tr>
                  <td>Crankcase Vibration RMS</td>
                  <td>2.1 mm/s</td>
                  <td>1.8 mm/s</td>
                  <td>+0.62</td>
                  <td>14%</td>
                </tr>
                <tr>
                  <td>Oil Pressure Residual</td>
                  <td>4.8 bar</td>
                  <td>4.9 bar</td>
                  <td>-0.18</td>
                  <td>8%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-black/40 p-3 rounded border border-cyan-500/20 text-xs space-y-1">
            <span className="font-semibold text-cyan-300 block mb-1">Rule-Based Symptom Evidence:</span>
            <div className="grid grid-cols-2 gap-2 text-gray-300">
              <div>• Exhaust Spread: <span className="text-amber-400 font-semibold">Elevated (46°C)</span></div>
              <div>• Misfire Counter: <span className="text-emerald-400 font-semibold">0 (Clear)</span></div>
              <div>• Oil Degradation Index: <span className="text-emerald-400 font-semibold">Nominal</span></div>
              <div>• Cooling Residual: <span className="text-emerald-400 font-semibold">Normal</span></div>
            </div>
          </div>
        </div>

        {/* Root Cause & Prescribed Actions */}
        <div className="fa-card lg:col-span-2">
          <h4>ROOT CAUSE DIAGNOSIS & MAINTENANCE DISPATCH</h4>
          <div className="fa-action-grid grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
            <div className="fa-action-box bg-black/40 p-3.5 rounded border border-cyan-500/20">
              <div className="act-head flex items-center gap-2 mb-2">
                <Cpu className="act-icon text-cyan-400 w-4 h-4" />
                <span className="font-semibold text-cyan-300 text-xs">AI Root Cause Hypothesis</span>
              </div>
              <p className="text-xs text-gray-300 leading-relaxed">
                Thermal gradient between Cylinder 1 (178°C) and Cylinder 2 (197°C) indicates early-stage exhaust valve seat carbon deposit or ring sealing variance on starboard bank. No structural mechanical damage detected.
              </p>
            </div>
            <div className="fa-action-box bg-black/40 p-3.5 rounded border border-emerald-500/20">
              <div className="act-head flex items-center gap-2 mb-2">
                <Wrench className="act-icon text-emerald-400 w-4 h-4" />
                <span className="font-semibold text-emerald-300 text-xs">Prescribed Action</span>
              </div>
              <p className="text-xs text-gray-300 leading-relaxed">
                Continue mission under nominal flight envelope. Perform borescope inspection of Cylinder 2 exhaust valve at scheduled 50-hour service interval.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
