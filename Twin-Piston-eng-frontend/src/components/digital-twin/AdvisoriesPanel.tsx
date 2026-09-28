import { useState, useEffect } from "react";
import { apiClient } from "@/lib/api-client";
import type { AdvisoryResponse, ExplanationResponse, DiagnosticQueryResponse } from "@/types/backend-api";
import { AlertTriangle, HelpCircle, Search, ShieldAlert, Sparkles, CheckCircle2 } from "lucide-react";

export function AdvisoriesPanel() {
  const [advisories, setAdvisories] = useState<AdvisoryResponse[]>([]);
  const [explanations, setExplanations] = useState<ExplanationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [queryInput, setQueryInput] = useState("");
  const [queryResult, setQueryResult] = useState<DiagnosticQueryResponse | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  useEffect(() => {
    async function loadData() {
      try {
        const [advs, expls] = await Promise.all([
          apiClient.getAdvisories().catch(() => []),
          apiClient.getExplanations().catch(() => []),
        ]);
        setAdvisories(advs);
        setExplanations(expls);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim()) return;
    setIsQuerying(true);
    try {
      const res = await apiClient.queryDiagnostics({ question_type: queryInput });
      setQueryResult(res);
    } catch (err: any) {
      setQueryResult({
        question_type: queryInput,
        answer: "Query error: Could not reach diagnostic reasoning engine.",
        evidence: [],
        limitations: "System offline or backend unavailable.",
        quality: 0,
        provenance: "DERIVED",
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="dedicated-subview-panel space-y-6">
      <div className="subview-header flex justify-between items-center">
        <div className="flex items-center gap-2">
          <ShieldAlert className="subview-icon text-amber-400" />
          <div>
            <h3>DECISION-SUPPORT ADVISORIES & OPERATOR DIAGNOSTIC Q&A</h3>
            <small className="text-cyan-400/80">
              Module 20 & 16 Advisory Guidance, Explanatory Evidence, and Interactive Query Interface
            </small>
          </div>
        </div>
      </div>

      {/* Top Search / Operator Query Bar */}
      <div className="fa-card">
        <h4 className="flex items-center gap-2 text-cyan-300">
          <Search className="w-4 h-4 text-cyan-400" /> OPERATOR DIAGNOSTIC QUERY SEARCH
        </h4>
        <form onSubmit={handleQuerySubmit} className="mt-3 flex gap-2">
          <input
            type="text"
            className="flex-1 bg-black/60 border border-cyan-500/30 rounded px-3 py-2 text-sm text-cyan-100 placeholder:text-gray-500 focus:outline-none focus:border-cyan-400"
            placeholder="Ask engine twin (e.g., 'HEALTH_STATUS', 'WHAT_CHANGED', 'EGT_SPREAD', 'OIL_DEGRADATION')..."
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
          />
          <button
            type="submit"
            disabled={isQuerying}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-black font-semibold rounded text-sm transition flex items-center gap-2 disabled:opacity-50"
          >
            {isQuerying ? "Analyzing..." : "Query Twin"}
          </button>
        </form>

        {queryResult && (
          <div className="mt-4 p-4 bg-cyan-950/40 border border-cyan-500/30 rounded text-sm space-y-2">
            <div className="flex items-center justify-between text-xs text-cyan-400">
              <span className="font-semibold">Query Category: {queryResult.question_type}</span>
              <span>Provenance: {queryResult.provenance}</span>
            </div>
            <p className="text-cyan-100 leading-relaxed">{queryResult.answer}</p>
            {queryResult.limitations && (
              <small className="block text-gray-400 italic">Limitations: {queryResult.limitations}</small>
            )}
          </div>
        )}
      </div>

      {/* Main Grid: Advisories & Explanations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Advisories */}
        <div className="fa-card">
          <h4>ACTIVE DECISION-SUPPORT ADVISORIES</h4>
          {loading ? (
            <p className="text-sm text-gray-400 py-4">Loading advisories from backend...</p>
          ) : advisories.length === 0 ? (
            <div className="py-6 text-center text-emerald-400/80 text-sm flex flex-col items-center gap-2">
              <CheckCircle2 className="w-8 h-8 text-emerald-400" />
              <span>No active warnings or advisories. Engine operating within nominal parameters.</span>
            </div>
          ) : (
            <div className="space-y-3 mt-3">
              {advisories.map((adv) => (
                <div
                  key={adv.advisory_id}
                  className="p-3 bg-black/40 border border-amber-500/30 rounded-md space-y-1 hover:border-amber-400/60 transition"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-amber-300 text-sm">{adv.title}</span>
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded ${
                        adv.priority === "CRITICAL" || adv.priority === "HIGH"
                          ? "bg-red-950 text-red-400 border border-red-500/40"
                          : "bg-amber-950 text-amber-300 border border-amber-500/40"
                      }`}
                    >
                      {adv.priority}
                    </span>
                  </div>
                  <p className="text-xs text-gray-300">{adv.message}</p>
                  <div className="flex items-center justify-between text-[11px] text-cyan-400/70 pt-1">
                    <span>Subsystem: {adv.subsystem}</span>
                    <span>Confidence: {(adv.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Human Explanations */}
        <div className="fa-card">
          <h4>EVIDENCE-BACKED DIAGNOSTIC EXPLANATIONS</h4>
          {loading ? (
            <p className="text-sm text-gray-400 py-4">Loading explanations...</p>
          ) : explanations.length === 0 ? (
            <div className="py-6 text-center text-gray-400 text-sm">
              No explanations available in current window.
            </div>
          ) : (
            <div className="space-y-3 mt-3">
              {explanations.map((exp) => (
                <div key={exp.explanation_id} className="p-3 bg-black/40 border border-cyan-500/20 rounded-md space-y-2">
                  <div className="flex items-center gap-2 text-cyan-300 font-semibold text-sm">
                    <Sparkles className="w-4 h-4 text-cyan-400" />
                    <span>{exp.finding}</span>
                  </div>
                  <div className="text-xs text-gray-300 space-y-1">
                    <p>
                      <strong className="text-cyan-400">Observation:</strong> {exp.observation}
                    </p>
                    <p>
                      <strong className="text-cyan-400">Interpretation:</strong> {exp.interpretation}
                    </p>
                  </div>
                  {exp.limitation && (
                    <small className="block text-[11px] text-amber-400/80 italic">Note: {exp.limitation}</small>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
