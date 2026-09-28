import { cn } from "@/lib/utils";
import type { EngineStatus } from "@/types/digital-twin";

export function StatusIndicator({ status, label }: { status: EngineStatus; label?: string }) {
  return <span className={cn("status-indicator", `status-${status}`)}><span className="status-dot" />{label ?? status}</span>;
}
