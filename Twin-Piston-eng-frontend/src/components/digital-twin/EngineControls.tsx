import { Expand, Focus, Hand, Move, Pause, Play, Rotate3D, ScanLine, Search, View } from "lucide-react";
import { Button } from "@/components/ui/button";

export type CameraView = "reset" | "front" | "side" | "top";
export function EngineControls({ autoRotate, onAutoRotate, onView, onFullscreen }: { autoRotate: boolean; onAutoRotate: () => void; onView: (view: CameraView) => void; onFullscreen: () => void; assembledOnly: boolean }) {
  const tools = [
    { label: "Orbit", icon: Rotate3D, action: () => onView("reset") },
    { label: "Zoom", icon: Search, action: () => onView("reset") },
    { label: "Pan", icon: Hand, action: () => onView("side") },
    { label: "Reset", icon: Focus, action: () => onView("reset") },
  ];
  return <div className="viewer-controls" aria-label="Engine viewer controls">
    <div className="control-cluster">{tools.map(({ label, icon: Icon, action }) => <Button key={label} variant="instrument" size="sm" onClick={action} title={label}><Icon /><span>{label}</span></Button>)}</div>
    <div className="auto-control"><span>Auto Rotate</span><Button variant="instrument" size="compactIcon" onClick={onAutoRotate} title={autoRotate ? "Pause auto rotation" : "Start auto rotation"} aria-label={autoRotate ? "Pause auto rotation" : "Start auto rotation"}>{autoRotate ? <Pause /> : <Play />}</Button></div>
    <div className="camera-chips">
      <Button variant="instrument" size="sm" onClick={() => onView("front")}><View />Front</Button>
      <Button variant="instrument" size="sm" onClick={() => onView("side")}><Move />Side</Button>
      <Button variant="instrument" size="sm" onClick={() => onView("top")}><ScanLine />Top</Button>
      <Button variant="instrument" size="compactIcon" onClick={onFullscreen} title="Fullscreen" aria-label="Fullscreen"><Expand /></Button>
    </div>
  </div>;
}
