import { Zap, Battery, Activity } from "lucide-react";
import type { EngineHealthResponse } from "@/types/backend-api";

export function ElectricalDiagnosticsCard({ data }: { data: EngineHealthResponse | null }) {
  const busV = data?.bus_voltage_v ?? 28.2;
  const altI = data?.alternator_current_a ?? 14.5;
  const battI = data?.battery_current_a ?? 1.2;
  const battRes = data?.battery_resistance_mohm ?? 18.4;
  const ripple = data?.voltage_ripple_pct ?? 0.8;
  const elecStatus = data?.electrical_status ?? "HEALTHY";

  return (
    <div className="telem-section bg-black/40 border border-cyan-500/20 p-4 rounded-md space-y-3">
      <div className="telem-section-header flex justify-between items-center">
        <h2 className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
          <Zap className="w-4 h-4 text-amber-400" /> FADEC ELECTRICAL BUS & BATTERY DIAGNOSTICS
        </h2>
        <span
          className={`px-2 py-0.5 text-[10px] font-bold rounded ${
            elecStatus === "HEALTHY" || elecStatus === "VALID"
              ? "bg-emerald-950 text-emerald-400 border border-emerald-500/40"
              : "bg-amber-950 text-amber-300 border border-amber-500/40"
          }`}
        >
          {elecStatus}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
        <div className="bg-black/60 p-2.5 rounded border border-cyan-500/10">
          <span className="text-gray-400 block text-[11px]">Bus Voltage</span>
          <strong className="text-cyan-200 text-sm font-mono">{busV.toFixed(1)} V</strong>
        </div>
        <div className="bg-black/60 p-2.5 rounded border border-cyan-500/10">
          <span className="text-gray-400 block text-[11px]">Alternator Current</span>
          <strong className="text-cyan-200 text-sm font-mono">{altI.toFixed(1)} A</strong>
        </div>
        <div className="bg-black/60 p-2.5 rounded border border-cyan-500/10">
          <span className="text-gray-400 block text-[11px]">Battery Current</span>
          <strong className="text-cyan-200 text-sm font-mono">{battI.toFixed(1)} A</strong>
        </div>
        <div className="bg-black/60 p-2.5 rounded border border-cyan-500/10">
          <span className="text-gray-400 block text-[11px]">Internal Resistance</span>
          <strong className="text-amber-300 text-sm font-mono">{battRes.toFixed(1)} mΩ</strong>
        </div>
        <div className="bg-black/60 p-2.5 rounded border border-cyan-500/10">
          <span className="text-gray-400 block text-[11px]">Voltage Ripple</span>
          <strong className="text-cyan-200 text-sm font-mono">{ripple.toFixed(2)} %</strong>
        </div>
      </div>
    </div>
  );
}
