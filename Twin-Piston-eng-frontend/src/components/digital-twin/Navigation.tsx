import {
  Box,
  Layers,
  Activity,
  Boxes,
  AlertOctagon,
  Clock,
  Compass,
  Settings,
  ShieldAlert,
  GitCompare,
} from "lucide-react";

export type NavItem =
  | "3D View"
  | "Exploded View"
  | "Telemetry"
  | "Components"
  | "Fault Analysis"
  | "RUL"
  | "Mission"
  | "Advisories & Q&A"
  | "Replay & What-If"
  | "Settings";

const NAV_ITEMS: { id: NavItem; label: string; icon: typeof Box }[] = [
  { id: "3D View", label: "3D VIEW", icon: Box },
  { id: "Exploded View", label: "EXPLODED VIEW", icon: Layers },
  { id: "Telemetry", label: "TELEMETRY", icon: Activity },
  { id: "Components", label: "COMPONENTS", icon: Boxes },
  { id: "Fault Analysis", label: "FAULT ANALYSIS", icon: AlertOctagon },
  { id: "RUL", label: "RUL", icon: Clock },
  { id: "Mission", label: "MISSION RELIABILITY", icon: Compass },
  { id: "Advisories & Q&A", label: "ADVISORIES & Q&A", icon: ShieldAlert },
  { id: "Replay & What-If", label: "REPLAY & WHAT-IF", icon: GitCompare },
  { id: "Settings", label: "SETTINGS", icon: Settings },
];

export function Navigation({
  active,
  onChange,
}: {
  active: string;
  onChange: (item: NavItem) => void;
}) {
  return (
    <aside className="chassis-nav-rail" aria-label="Main aerospace console navigation">
      {/* Corner screws on chassis rail */}
      <div className="screw screw-tl" />
      <div className="screw screw-tr" />
      <div className="screw screw-bl" />
      <div className="screw screw-br" />

      <nav className="nav-buttons-list">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const isActive = active === id || (active === "3D View" && id === "3D View");
          return (
            <button
              type="button"
              key={id}
              className={`chassis-nav-btn ${isActive ? "nav-btn-active" : ""}`}
              onClick={() => onChange(id)}
              title={label}
            >
              <div className="nav-btn-indicator" />
              <Icon className="nav-btn-icon" />
              <span className="nav-btn-label">{label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
