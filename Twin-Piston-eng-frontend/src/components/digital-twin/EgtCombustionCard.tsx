import { Flame, Activity } from "lucide-react";
import type { DiagnosticSummaryResponse } from "@/types/backend-api";

export function EgtCombustionCard({ data }: { data: DiagnosticSummaryResponse | null }) {
  const egtMeanC = data ? (data.egt_mean_k - 273.15).toFixed(0) : "710";
  const egtSpreadC = data ? (data.egt_spread_k).toFixed(0) : "32";
  const misfires = data?.misfire_detected ?? [false, false, false, false];

  const cylinders = [
    { name: "Cylinder 1", egt: 712, cht: 182, misfire: misfires[0] },
    { name: "Cylinder 2", egt: 728, cht: 195, misfire: misfires[1] },
    { name: "Cylinder 3", egt: 704, cht: 178, misfire: misfires[2] },
    { name: "Cylinder 4", egt: 698, cht: 174, misfire: misfires[3] },
  ];

  return (
    <div className="telem-section bg-black/40 border border-cyan-500/20 p-4 rounded-md space-y-3">
      <div className="telem-section-header flex justify-between items-center">
        <h2 className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
          <Flame className="w-4 h-4 text-orange-400" /> 4-CYLINDER EGT THERMAL SPREAD & MISFIRE MONITOR
        </h2>
        <div className="flex gap-3 text-xs text-cyan-400">
          <span>Mean EGT: <strong className="text-cyan-200">{egtMeanC} °C</strong></span>
          <span>Max Spread: <strong className="text-amber-300">{egtSpreadC} °C</strong></span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        {cylinders.map((cyl, idx) => (
          <div
            key={cyl.name}
            className={`p-3 rounded border ${
              cyl.misfire
                ? "bg-red-950/40 border-red-500/60"
                : "bg-black/60 border-cyan-500/10"
            }`}
          >
            <div className="flex justify-between items-center mb-1">
              <span className="font-semibold text-cyan-300">{cyl.name}</span>
              <span
                className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
                  cyl.misfire ? "bg-red-600 text-white" : "bg-emerald-950 text-emerald-400"
                }`}
              >
                {cyl.misfire ? "MISFIRE" : "OK"}
              </span>
            </div>
            <div className="flex justify-between text-gray-400 text-[11px] pt-1">
              <span>EGT: <strong className="text-cyan-100">{cyl.egt} °C</strong></span>
              <span>CHT: <strong className="text-cyan-100">{cyl.cht} °C</strong></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
