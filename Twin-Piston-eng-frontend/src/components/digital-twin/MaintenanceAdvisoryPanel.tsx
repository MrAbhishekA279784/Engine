import type { MaintenanceStatus } from "@/types/digital-twin";

type Props = {
  status: MaintenanceStatus;
  note: string;
  nextInspectionHours: number;
  reason?: string;
};

const STATUS_LABEL: Record<MaintenanceStatus, string> = {
  MONITOR: "MONITOR",
  SCHEDULE: "SCHEDULE",
  GROUND: "GROUND",
};

const STATUS_CLASS: Record<MaintenanceStatus, string> = {
  MONITOR: "maint-monitor",
  SCHEDULE: "maint-schedule",
  GROUND: "maint-ground",
};

export function MaintenanceAdvisoryPanel({ status, note, nextInspectionHours, reason }: Props) {
  return (
    <section className="ml-panel" aria-labelledby="maint-title">
      <div className="ml-panel-header">
        <h3 id="maint-title">MAINTENANCE ADVISORY</h3>
      </div>
      <div className="maint-status-row">
        <span className="maint-label">Status:</span>
        <span className={`maint-badge ${STATUS_CLASS[status]}`}>{STATUS_LABEL[status]}</span>
      </div>
      <div className="maint-detail">
        <div className="maint-block">
          <span className="maint-field-label">Recommendation</span>
          <p className="maint-text">{note}</p>
        </div>
        <div className="maint-block">
          <span className="maint-field-label">Next Inspection</span>
          <p className="maint-text">{nextInspectionHours} operating hours</p>
        </div>
        {reason && (
          <div className="maint-block">
            <span className="maint-field-label">Reason</span>
            <p className="maint-text maint-reason">{reason}</p>
          </div>
        )}
      </div>
    </section>
  );
}
