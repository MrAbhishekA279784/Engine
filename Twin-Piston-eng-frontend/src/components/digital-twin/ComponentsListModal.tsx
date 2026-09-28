import type { EngineComponent } from "@/types/digital-twin";
import {
  Cog,
  CircleDot,
  Cpu,
  Zap,
  Flame,
  Droplets,
  Activity,
  Shield,
  Layers,
  Search,
} from "lucide-react";
import { useState, useMemo } from "react";

export function ComponentsListModal({
  components,
  selectedId,
  onSelect,
  onClose,
}: {
  components: EngineComponent[];
  selectedId: string;
  onSelect: (id: string) => void;
  onClose?: () => void;
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    return components
      .filter((c) => c.id !== "assembly")
      .filter((c) => !q || c.name.toLowerCase().includes(q) || c.category?.toLowerCase().includes(q));
  }, [components, query]);

  const getIcon = (id: string) => {
    if (id.includes("plug") || id.includes("ecu")) return Zap;
    if (id.includes("injector") || id.includes("fuel")) return Droplets;
    if (id.includes("exhaust")) return Flame;
    if (id.includes("sensor")) return Activity;
    if (id.includes("case") || id.includes("head") || id.includes("cyl")) return CircleDot;
    if (id.includes("prop")) return Cog;
    return Layers;
  };

  return (
    <div className="components-registry-view" aria-labelledby="comp-reg-title">
      <div className="comp-reg-header">
        <div className="reg-title-wrap">
          <Shield className="reg-icon" />
          <div>
            <h3 id="comp-reg-title">ENGINE COMPONENTS REGISTRY</h3>
            <small>28 Monitored Subsystems & Line Replaceable Units (LRU)</small>
          </div>
        </div>
        <div className="reg-search-wrap">
          <Search className="reg-search-icon" />
          <input
            type="text"
            placeholder="Filter components..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      </div>

      <div className="comp-reg-grid">
        {filtered.map((comp) => {
          const Icon = getIcon(comp.id);
          const isSelected = selectedId === comp.id;
          const isCaution = comp.status === "caution";
          const isCritical = comp.status === "critical";

          return (
            <button
              type="button"
              key={comp.id}
              className={`comp-reg-card ${isSelected ? "reg-card-active" : ""}`}
              onClick={() => {
                onSelect(comp.id);
                if (onClose) onClose();
              }}
            >
              <div className="reg-card-num">
                {String(comp.number ?? 1).padStart(2, "0")}.
              </div>
              <Icon className="reg-card-icon" />
              <div className="reg-card-content">
                <strong>{comp.name}</strong>
                <span>{comp.category ?? "Subsystem"}</span>
              </div>
              <div className="reg-card-health">
                <span className={isCaution ? "value-caution" : isCritical ? "value-critical" : "value-normal"}>
                  {comp.health}%
                </span>
                <i className={`status-dot ${isCaution ? "status-caution" : isCritical ? "status-critical" : "status-normal"}`} />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
