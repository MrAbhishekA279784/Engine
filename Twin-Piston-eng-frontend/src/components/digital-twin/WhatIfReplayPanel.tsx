import { useState, useEffect } from "react";
import { apiClient } from "@/lib/api-client";
import type { ReplayStateResponse, ScenarioCompareResponse, WhatIfResponse } from "@/types/backend-api";
import { Play, Pause, Square, FastForward, GitCompare, RotateCcw, Sliders, Activity } from "lucide-react";

export function WhatIfReplayPanel() {
  const [replayState, setReplayState] = useState<ReplayStateResponse | null>(null);
  const [rpmModification, setRpmModification] = useState<number>(2400);
  const [faultScenario, setFaultScenario] = useState<string>("NONE");
  const [duration, setDuration] = useState<number>(30);
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResponse | null>(null);
  const [comparison, setComparison] = useState<ScenarioCompareResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchReplayState = async () => {
    try {
      const state = await apiClient.getReplayState();
      setReplayState(state);
    } catch {
      // Offline fallback state
      setReplayState({
        scenario_id: "baseline_01",
        replay_position: 0,
        total_records: 1000,
        status: "STOPPED",
        current_timestamp: new Date().toISOString(),
        sequence_number: 0,
        speed_multiplier: 1.0,
        provenance: "SIMULATED",
      });
    }
  };

  useEffect(() => {
    fetchReplayState();
    const interval = setInterval(fetchReplayState, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    await apiClient.startReplay();
    fetchReplayState();
  };

  const handlePause = async () => {
    await apiClient.pauseReplay();
    fetchReplayState();
  };

  const handleResume = async () => {
    await apiClient.resumeReplay();
    fetchReplayState();
  };

  const handleStop = async () => {
    await apiClient.stopReplay();
    fetchReplayState();
  };

  const handleRunWhatIf = async () => {
    setLoading(true);
    try {
      const res = await apiClient.executeWhatIf({
        baseline_scenario_id: "baseline_01",
        modifications: {
          target_rpm: rpmModification,
          fault_type: faultScenario !== "NONE" ? faultScenario : undefined,
        },
        duration_s: duration,
      });
      setWhatIfResult(res);

      // Trigger compare
      const comp = await apiClient.compareScenarios("baseline_01", res.what_if_id);
      setComparison(comp);
    } catch (err: any) {
      alert("What-If simulation error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dedicated-subview-panel space-y-6">
      <div className="subview-header flex justify-between items-center">
        <div className="flex items-center gap-2">
          <GitCompare className="subview-icon text-cyan-400" />
          <div>
            <h3>SCENARIO REPLAY & WHAT-IF DIGITAL TWIN SIMULATION</h3>
            <small className="text-cyan-400/80">
              Module 17 & 20 Telemetry Replay Control and Predictive Counterfactual Execution
            </small>
          </div>
        </div>
      </div>

      {/* Replay Control Bar */}
      <div className="fa-card">
        <h4 className="flex items-center justify-between text-cyan-300">
          <span className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" /> TELEMETRY REPLAY CONTROL BAR
          </span>
          <span className="text-xs px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/40">
            STATUS: {replayState?.status ?? "STOPPED"}
          </span>
        </h4>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-4 bg-black/50 p-3 rounded border border-cyan-500/20">
          <div className="flex items-center gap-2">
            {replayState?.status === "RUNNING" ? (
              <button
                onClick={handlePause}
                className="px-3 py-1.5 bg-amber-600/80 hover:bg-amber-500 text-black font-bold text-xs rounded flex items-center gap-1"
              >
                <Pause className="w-3.5 h-3.5" /> Pause
              </button>
            ) : replayState?.status === "PAUSED" ? (
              <button
                onClick={handleResume}
                className="px-3 py-1.5 bg-emerald-600/80 hover:bg-emerald-500 text-black font-bold text-xs rounded flex items-center gap-1"
              >
                <Play className="w-3.5 h-3.5" /> Resume
              </button>
            ) : (
              <button
                onClick={handleStart}
                className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-black font-bold text-xs rounded flex items-center gap-1"
              >
                <Play className="w-3.5 h-3.5" /> Start Replay
              </button>
            )}
            <button
              onClick={handleStop}
              className="px-3 py-1.5 bg-red-950 hover:bg-red-900 border border-red-500/40 text-red-300 text-xs rounded flex items-center gap-1"
            >
              <Square className="w-3.5 h-3.5" /> Stop
            </button>
          </div>

          <div className="flex-1 max-w-md flex items-center gap-3 text-xs">
            <span className="text-gray-400">Position: {replayState?.replay_position ?? 0}</span>
            <input
              type="range"
              min="0"
              max={replayState?.total_records || 1000}
              value={replayState?.replay_position || 0}
              onChange={(e) => apiClient.seekReplay(parseInt(e.target.value))}
              className="flex-1 h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <span className="text-gray-400">/ {replayState?.total_records ?? 1000}</span>
          </div>
        </div>
      </div>

      {/* What-If Scenario Executor & Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Card */}
        <div className="fa-card">
          <h4 className="flex items-center gap-2 text-cyan-300">
            <Sliders className="w-4 h-4 text-cyan-400" /> WHAT-IF PARAMETERS
          </h4>
          <div className="space-y-4 mt-3 text-xs">
            <div>
              <label className="block text-gray-300 mb-1">Target Engine Speed (RPM)</label>
              <input
                type="number"
                min="800"
                max="3200"
                value={rpmModification}
                onChange={(e) => setRpmModification(parseInt(e.target.value) || 2400)}
                className="w-full bg-black/60 border border-cyan-500/30 rounded px-2.5 py-1.5 text-cyan-100"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-1">Injected Fault Condition</label>
              <select
                value={faultScenario}
                onChange={(e) => setFaultScenario(e.target.value)}
                className="w-full bg-black/60 border border-cyan-500/30 rounded px-2.5 py-1.5 text-cyan-100"
              >
                <option value="NONE">NOMINAL (Baseline)</option>
                <option value="MISFIRE">CYLINDER MISFIRE</option>
                <option value="EXHAUST_VALVE_LEAK">EXHAUST VALVE LEAK</option>
                <option value="OIL_DEGRADATION">OIL THERMAL DEGRADATION</option>
                <option value="COOLING_FAULT">COOLING CIRCUIT BLOCKAGE</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-300 mb-1">Simulation Duration (Seconds)</label>
              <input
                type="number"
                min="5"
                max="300"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value) || 30)}
                className="w-full bg-black/60 border border-cyan-500/30 rounded px-2.5 py-1.5 text-cyan-100"
              />
            </div>
            <button
              onClick={handleRunWhatIf}
              disabled={loading}
              className="w-full py-2 bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-black font-bold rounded text-xs transition"
            >
              {loading ? "Simulating Twin..." : "Execute What-If Simulation"}
            </button>
          </div>
        </div>

        {/* Results & Delta Comparison View */}
        <div className="fa-card lg:col-span-2">
          <h4>BASELINE VS WHAT-IF METRIC DELTA COMPARISON</h4>
          {!comparison ? (
            <div className="py-12 text-center text-gray-400 text-xs">
              Execute a What-If scenario on the left to compute counterfactual digital twin metric deltas.
            </div>
          ) : (
            <div className="mt-3 space-y-4 text-xs">
              <div className="flex justify-between items-center bg-cyan-950/40 p-2.5 rounded border border-cyan-500/30">
                <span>Baseline ID: <strong>{comparison.baseline_id}</strong></span>
                <span>What-If ID: <strong>{comparison.what_if_id}</strong></span>
              </div>
              <div className="overflow-x-auto">
                <table className="fa-matrix-table w-full">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Baseline Value</th>
                      <th>What-If Value</th>
                      <th>Delta Variance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparison.metric_deltas || {}).map(([key, val]: any) => (
                      <tr key={key}>
                        <td className="font-semibold text-cyan-300">{key}</td>
                        <td>{val.baseline ?? "Nominal"}</td>
                        <td>{val.what_if ?? "Altered"}</td>
                        <td className={val.delta > 0 ? "text-amber-400" : "text-cyan-400"}>
                          {val.delta !== undefined ? `${val.delta > 0 ? "+" : ""}${val.delta}` : "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
