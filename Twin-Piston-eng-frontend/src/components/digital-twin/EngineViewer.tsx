import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { AccumulativeShadows, ContactShadows, Environment, Lightformer, OrbitControls, RandomizedLight } from "@react-three/drei";
import { GLTFLoader, type OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import {
  splitEngineParts,
  createProceduralEngineParts,
  CALLOUT_PINS,
  ZONE_OFFSETS,
  type EnginePart,
  type CalloutPin,
} from "@/lib/engine-parts";
import {
  Rotate3d,
  ZoomIn,
  Move,
  RotateCcw,
  Eye,
  Sliders,
  CheckCircle2,
  EyeOff,
  Crosshair,
  Layers,
} from "lucide-react";

export type CameraView = "reset" | "front" | "side" | "top";

const VIEW_POSITIONS: Record<CameraView, THREE.Vector3> = {
  reset: new THREE.Vector3(4.8, 2.8, 5.4),
  front: new THREE.Vector3(0, 0.15, 6.8),
  side: new THREE.Vector3(6.8, 0.15, 0),
  top: new THREE.Vector3(0, 7.2, 0.01),
};

const COMPONENT_FOCUS: Record<string, THREE.Vector3> = {
  assembly: new THREE.Vector3(0, 0, 0),
  crankcase: new THREE.Vector3(0, -0.1, 0.05),
  "cylinder-1": new THREE.Vector3(-1.1, 0.05, 0.1),
  "cylinder-2": new THREE.Vector3(1.1, 0.05, 0.1),
  "cylinder-head-1": new THREE.Vector3(-1.45, 0.1, 0.2),
  "cylinder-head-2": new THREE.Vector3(1.45, 0.1, 0.2),
  "spark-plug-1": new THREE.Vector3(-1.25, 0.35, 0.15),
  "spark-plug-2": new THREE.Vector3(1.25, 0.35, 0.15),
  "fuel-injector-1": new THREE.Vector3(-0.75, 0.4, 0.05),
  "fuel-injector-2": new THREE.Vector3(0.75, 0.4, 0.05),
  "throttle-body": new THREE.Vector3(-0.1, 0.75, -0.1),
  "intake-manifold": new THREE.Vector3(0.1, 0.65, 0.05),
  "exhaust-manifold": new THREE.Vector3(-0.85, -0.45, 0.35),
  "oil-pump": new THREE.Vector3(0.1, -0.65, 0.1),
  "oil-filter": new THREE.Vector3(-0.25, -0.6, 0.25),
  "ecu-fadec": new THREE.Vector3(0.7, 0.6, 0.25),
  "propeller-drive": new THREE.Vector3(0.5, -0.2, -0.65),
};

function CameraRig({
  position,
  lookAt,
  transitionKey,
  controlsRef,
}: {
  position: THREE.Vector3;
  lookAt: THREE.Vector3;
  transitionKey: number;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
}) {
  const { camera } = useThree();
  const moving = useRef(true);
  useEffect(() => {
    moving.current = true;
  }, [transitionKey]);

  useFrame((_, rawDelta) => {
    if (!moving.current) return;
    const delta = Math.min(rawDelta, 0.05);
    const damping = 1 - Math.exp(-5.5 * delta);
    camera.position.lerp(position, damping);
    controlsRef.current?.target.lerp(lookAt, damping);
    controlsRef.current?.update();
    if (
      camera.position.distanceTo(position) < 0.02 &&
      (controlsRef.current?.target.distanceTo(lookAt) ?? 0) < 0.02
    ) {
      moving.current = false;
    }
  });
  return null;
}

// ── Screen Position Projector for Callout Pins ──────────────────────────────
function PinProjector({
  pins,
  explode,
  pinRefs,
}: {
  pins: CalloutPin[];
  explode: number;
  pinRefs: React.RefObject<Record<string, HTMLDivElement | null>>;
}) {
  const { camera, size } = useThree();
  const tempVec = useMemo(() => new THREE.Vector3(), []);
  const zeroVec = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const halfW = size.width / 2;
    const halfH = size.height / 2;
    const refs = pinRefs.current;
    if (!refs) return;

    for (let i = 0; i < pins.length; i++) {
      const pin = pins[i];
      if (!pin) continue;
      const el = refs[pin.id];
      if (!el) continue;

      tempVec.set(...pin.anchor);
      const zoneOffset = ZONE_OFFSETS[pin.zone] ?? zeroVec;
      if (explode > 0) {
        tempVec.addScaledVector(zoneOffset, explode);
      }

      tempVec.project(camera);
      const isBehind = tempVec.z > 1;
      const x = tempVec.x * halfW + halfW;
      const y = -tempVec.y * halfH + halfH;
      const isVisible =
        !isBehind && x >= -50 && x <= size.width + 50 && y >= -50 && y <= size.height + 50;

      if (isVisible) {
        el.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        if (el.style.display !== "flex") el.style.display = "flex";
      } else {
        if (el.style.display !== "none") el.style.display = "none";
      }
    }
  });

  return null;
}

let cachedEngineParts: {
  parts: EnginePart[];
  scale: number;
  offset: THREE.Vector3;
} | null = null;

// ── Engine Model ────────────────────────────────────────────────────────────
function EngineModel({
  selectedId,
  hoveredZone,
  hiddenZones,
  isolatedZone,
  onHover,
  onSelect,
  onReady,
  onError,
  explode,
}: {
  selectedId: string;
  hoveredZone: string | null;
  hiddenZones: Set<string>;
  isolatedZone: string | null;
  onHover: (zone: string | null) => void;
  onSelect: (id: string) => void;
  onReady: () => void;
  onError: () => void;
  explode: number;
}) {
  const [parts, setParts] = useState<EnginePart[]>(() => cachedEngineParts?.parts ?? []);
  const [scale, setScale] = useState(() => cachedEngineParts?.scale ?? 1);
  const [offset, setOffset] = useState(() => cachedEngineParts?.offset ?? new THREE.Vector3());
  const groupRef = useRef<THREE.Group>(null);
  const explodeRef = useRef(0);

  useEffect(() => {
    if (cachedEngineParts) {
      setParts(cachedEngineParts.parts);
      setScale(cachedEngineParts.scale);
      setOffset(cachedEngineParts.offset);
      onReady();
      return;
    }

    let active = true;
    const loader = new GLTFLoader();

    loader.load(
      "/twin-piston-engine.glb",
      (gltf) => {
        if (!active) return;

        try {
          // ── Collect ALL mesh nodes from the GLB scene ──────────────────────
          const allPositions: Float32Array[] = [];
          const allNormals: Float32Array[] = [];
          const allUVs: Float32Array[] = [];
          let totalVerts = 0;

          gltf.scene.updateMatrixWorld(true);

          gltf.scene.traverse((child) => {
            if (!(child instanceof THREE.Mesh)) return;
            const geo = child.geometry as THREE.BufferGeometry;
            const cloned = geo.clone();
            cloned.applyMatrix4(child.matrixWorld);

            const nonIdx = cloned.index ? cloned.toNonIndexed() : cloned;
            const posAttr = nonIdx.getAttribute("position") as THREE.BufferAttribute | undefined;
            if (!posAttr) return;

            totalVerts += posAttr.count;
            allPositions.push(new Float32Array(posAttr.array));

            const normAttr = nonIdx.getAttribute("normal") as THREE.BufferAttribute | undefined;
            if (normAttr) allNormals.push(new Float32Array(normAttr.array));

            const uvAttr = nonIdx.getAttribute("uv") as THREE.BufferAttribute | undefined;
            if (uvAttr) allUVs.push(new Float32Array(uvAttr.array));
          });

          if (totalVerts > 0) {
            const merged = new THREE.BufferGeometry();
            const mergedPos = new Float32Array(allPositions.reduce((s, a) => s + a.length, 0));
            let off = 0;
            for (const arr of allPositions) {
              mergedPos.set(arr, off);
              off += arr.length;
            }
            merged.setAttribute("position", new THREE.BufferAttribute(mergedPos, 3));

            if (allNormals.length === allPositions.length) {
              const mergedNorm = new Float32Array(allNormals.reduce((s, a) => s + a.length, 0));
              let noff = 0;
              for (const arr of allNormals) {
                mergedNorm.set(arr, noff);
                noff += arr.length;
              }
              merged.setAttribute("normal", new THREE.BufferAttribute(mergedNorm, 3));
            }

            if (allUVs.length === allPositions.length) {
              const mergedUV = new Float32Array(allUVs.reduce((s, a) => s + a.length, 0));
              let uoff = 0;
              for (const arr of allUVs) {
                mergedUV.set(arr, uoff);
                uoff += arr.length;
              }
              merged.setAttribute("uv", new THREE.BufferAttribute(mergedUV, 2));
            }

            const split = splitEngineParts(merged);
            const bounds = new THREE.Box3();
            split.forEach((p) => {
              p.geometry.computeBoundingBox();
              if (p.geometry.boundingBox) bounds.union(p.geometry.boundingBox);
            });
            const size = bounds.getSize(new THREE.Vector3());
            const computedScale = 4.9 / Math.max(size.x, size.y, size.z, 0.001);
            const computedOffset = bounds.getCenter(new THREE.Vector3()).negate();

            cachedEngineParts = {
              parts: split,
              scale: computedScale,
              offset: computedOffset,
            };

            setParts(split);
            setScale(computedScale);
            setOffset(computedOffset);
            onReady();
          } else {
            fallbackProcedural();
          }
        } catch (err) {
          console.error("GLB mesh parsing error:", err);
          fallbackProcedural();
        }
      },
      undefined,
      (err) => {
        if (!active) return;
        console.error("GLB fetch failed:", err);
        onError();
        fallbackProcedural();
      }
    );

    const fallbackProcedural = () => {
      const procParts = createProceduralEngineParts();
      const bounds = new THREE.Box3();
      procParts.forEach((p) => {
        p.geometry.computeBoundingBox();
        if (p.geometry.boundingBox) bounds.union(p.geometry.boundingBox);
      });
      const size = bounds.getSize(new THREE.Vector3());
      const computedScale = 4.2 / Math.max(size.x, size.y, size.z, 0.001);
      const computedOffset = bounds.getCenter(new THREE.Vector3()).negate();

      cachedEngineParts = {
        parts: procParts,
        scale: computedScale,
        offset: computedOffset,
      };

      setParts(procParts);
      setScale(computedScale);
      setOffset(computedOffset);
      onReady();
    };

    return () => {
      active = false;
    };
  }, [onReady, onError]);

  // ── Photorealistic PBR Materials — Aerospace Lab Reference ───────────────
  // All materials use envMapIntensity for realistic IBL reflections.
  // Values carefully tuned to match real aerospace engine photography.
  const materials = useMemo(
    () => ({
      // Main crankcase / housing — target base medium machined aluminium
      crankcase: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#737C82"),
        metalness: 0.92,
        roughness: 0.36,
        envMapIntensity: 1.0,
      }),
      // Cylinder barrels / cooling fins — recessed machined aluminium in shadow range
      cylinder: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#596166"),
        metalness: 0.90,
        roughness: 0.42,
        envMapIntensity: 0.90,
      }),
      // Cylinder heads & valve covers — exposed machined aluminium in highlight range
      cylinderHead: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#9AA3A8"),
        metalness: 0.95,
        roughness: 0.32,
        envMapIntensity: 1.05,
      }),
      // Exhaust manifolds — machined aluminium / stainless alloy
      exhaust: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#737C82"),
        metalness: 0.94,
        roughness: 0.30,
        envMapIntensity: 1.0,
      }),
      // Intake manifold & hoses — neutral matte machined alloy
      intake: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#656E74"),
        metalness: 0.88,
        roughness: 0.40,
        envMapIntensity: 0.90,
      }),
      // Throttle body — billet machined aluminium
      throttle: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#889298"),
        metalness: 0.94,
        roughness: 0.32,
        envMapIntensity: 1.05,
      }),
      // ECU / FADEC — medium anodized aluminium casing
      ecu: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#596166"),
        metalness: 0.88,
        roughness: 0.38,
        envMapIntensity: 0.90,
      }),
      // Propeller drive / flywheel — turned machined steel / aluminium
      propHub: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#737C82"),
        metalness: 0.96,
        roughness: 0.30,
        envMapIntensity: 1.05,
      }),
      // Oil filter — spun aluminium canister
      oilFilter: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#737C82"),
        metalness: 0.90,
        roughness: 0.35,
        envMapIntensity: 0.95,
      }),
      // Oil pump — cast aluminium housing
      oilPump: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#656E74"),
        metalness: 0.90,
        roughness: 0.38,
        envMapIntensity: 0.95,
      }),
      // Spark plug collars — machined aluminium / steel
      sparkPlug: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#808A90"),
        metalness: 0.92,
        roughness: 0.32,
        envMapIntensity: 1.0,
      }),
      // Fuel injectors — precision machined aluminium alloy
      fuelInjector: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#7A848A"),
        metalness: 0.92,
        roughness: 0.32,
        envMapIntensity: 1.0,
      }),
      // Hover overlay — restrained cyan digital twin highlight
      hover: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#40a8c2"),
        emissive: new THREE.Color("#007898"),
        emissiveIntensity: 0.25,
        metalness: 0.86,
        roughness: 0.24,
        envMapIntensity: 1.15,
      }),
      // Selected overlay — active selection indicator
      selected: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#50b8d4"),
        emissive: new THREE.Color("#0098b8"),
        emissiveIntensity: 0.32,
        metalness: 0.88,
        roughness: 0.20,
        envMapIntensity: 1.25,
      }),
      caution: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#c89040"),
        emissive: new THREE.Color("#d07800"),
        emissiveIntensity: 0.40,
        metalness: 0.68,
        roughness: 0.30,
        envMapIntensity: 1.0,
      }),
      critical: new THREE.MeshStandardMaterial({
        color: new THREE.Color("#c83428"),
        emissive: new THREE.Color("#a02418"),
        emissiveIntensity: 0.48,
        metalness: 0.64,
        roughness: 0.32,
        envMapIntensity: 1.0,
      }),
    }),
    []
  );

  // Map zone to base material
  function getZoneMaterial(zone: string) {
    switch (zone) {
      case "cylinder-head-1":
      case "cylinder-head-2":
        return materials.cylinderHead;
      case "cylinder-1":
      case "cylinder-2":
      case "cooling-fins":
        return materials.cylinder;
      case "exhaust-manifold":
        return materials.exhaust;
      case "throttle-body":
        return materials.throttle;
      case "intake-manifold":
        return materials.intake;
      case "ecu-fadec":
        return materials.ecu;
      case "propeller-drive":
        return materials.propHub;
      case "oil-filter":
        return materials.oilFilter;
      case "oil-pump":
        return materials.oilPump;
      case "spark-plug-1":
      case "spark-plug-2":
        return materials.sparkPlug;
      case "fuel-injector-1":
      case "fuel-injector-2":
        return materials.fuelInjector;
      default:
        return materials.crankcase;
    }
  }

  useFrame((_, rawDelta) => {
    const delta = Math.min(rawDelta, 0.05);
    explodeRef.current = THREE.MathUtils.lerp(
      explodeRef.current,
      explode,
      1 - Math.exp(-6 * delta)
    );
    const group = groupRef.current;
    if (!group) return;
    const amount = explodeRef.current;
    group.children.forEach((child) => {
      const offset = child.userData["explodeOffset"] as THREE.Vector3 | undefined;
      if (offset) {
        child.position.copy(offset).multiplyScalar(amount);
      }
    });
  });

  return (
    <group scale={scale}>
      <group ref={groupRef} position={[offset.x, offset.y, offset.z]}>
        {parts.map((part) => {
          if (hiddenZones.has(part.zone)) return null;
          if (isolatedZone && isolatedZone !== "assembly" && part.zone !== isolatedZone) return null;

          const isSelected = selectedId === part.zone;
          const isHovered = hoveredZone === part.zone;

          let mat = getZoneMaterial(part.zone);

          if (isSelected) {
            mat = materials.selected;
          } else if (isHovered) {
            mat = materials.hover;
          }

          return (
            <mesh
              key={part.id}
              userData={{ partId: part.id, explodeOffset: part.explodeOffset }}
              geometry={part.geometry}
              material={mat}
              castShadow
              receiveShadow
              onPointerMove={(e) => {
                e.stopPropagation();
                document.body.style.cursor = "pointer";
                onHover(part.zone);
              }}
              onPointerOut={() => {
                document.body.style.cursor = "default";
                onHover(null);
              }}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(part.zone);
              }}
            />
          );
        })}
      </group>
    </group>
  );
}

// ── Main EngineViewer Component ─────────────────────────────────────────────
export function EngineViewer({
  selectedId,
  onSelectComponent,
}: {
  selectedId: string;
  onSelectComponent: (id: string) => void;
}) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoRotate, setAutoRotate] = useState(false);
  const [explode, setExplode] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const handleModelReady = useCallback(() => setModelReady(true), []);
  const [loadError, setLoadError] = useState(false);
  const handleModelError = useCallback(() => setLoadError(true), []);

  const [hoveredZone, setHoveredZone] = useState<string | null>(null);
  const [hiddenZones, setHiddenZones] = useState<Set<string>>(new Set());
  const [isolatedZone, setIsolatedZone] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<"select" | "hide" | "isolate">("select");

  const pinRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const [cameraRequest, setCameraRequest] = useState(() => ({
    position: VIEW_POSITIONS.reset.clone(),
    lookAt: new THREE.Vector3(),
    key: 0,
  }));

  const requestCamera = (position: THREE.Vector3, lookAt = new THREE.Vector3()) => {
    setCameraRequest((curr) => ({
      position: position.clone(),
      lookAt: lookAt.clone(),
      key: curr.key + 1,
    }));
  };

  useEffect(() => {
    const focus = COMPONENT_FOCUS[selectedId] ?? COMPONENT_FOCUS["assembly"] ?? new THREE.Vector3();
    const distance = selectedId === "assembly" ? 1 : 0.72;
    requestCamera(VIEW_POSITIONS.reset.clone().multiplyScalar(distance).add(focus), focus);
  }, [selectedId]);

  const changeView = (next: CameraView) => {
    setAutoRotate(false);
    if (next === "reset") {
      onSelectComponent("assembly");
      setIsolatedZone(null);
      setHiddenZones(new Set());
    }
    requestCamera(VIEW_POSITIONS[next]);
  };

  const handleToolAction = (tool: "select" | "hide" | "isolate") => {
    setActiveTool(tool);
    if (tool === "hide" && selectedId && selectedId !== "assembly") {
      setHiddenZones((prev) => new Set([...prev, selectedId]));
      onSelectComponent("assembly");
    } else if (tool === "isolate" && selectedId && selectedId !== "assembly") {
      setIsolatedZone(isolatedZone === selectedId ? null : selectedId);
    }
  };

  const handlePinClick = (zone: string) => {
    onSelectComponent(zone);
  };

  return (
    <section
      className="viewer-shell"
      ref={containerRef}
      aria-label="Realistic 3D Aircraft Twin-Piston Digital Twin"
    >
      {/* ── Top Viewport Navigation Toolbar ──────────────────────────────── */}
      <div className="viewer-topbar-hud">
        <div className="hud-btn-cluster">
          <button
            type="button"
            className={`hud-btn ${explode === 0 ? "hud-btn-active" : ""}`}
            onClick={() => setExplode(0)}
          >
            <Layers className="hud-icon" />
            <span>ASSEMBLED</span>
          </button>
          <button
            type="button"
            className={`hud-btn ${explode > 0 ? "hud-btn-active" : ""}`}
            onClick={() => setExplode(100)}
          >
            <Sliders className="hud-icon" />
            <span>EXPLODED ASSEMBLY</span>
          </button>
          <span className="hud-divider" />
          <button
            type="button"
            className={`hud-btn ${autoRotate ? "hud-btn-active" : ""}`}
            onClick={() => setAutoRotate((v) => !v)}
            title="Toggle Orbit Auto-Rotation"
          >
            <Rotate3d className="hud-icon" />
            <span>ORBIT</span>
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => requestCamera(VIEW_POSITIONS.reset.clone().multiplyScalar(0.75))}
            title="Zoom In"
          >
            <ZoomIn className="hud-icon" />
            <span>ZOOM</span>
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => changeView("side")}
            title="Pan Side View"
          >
            <Move className="hud-icon" />
            <span>PAN</span>
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => changeView("reset")}
            title="Reset Camera"
          >
            <RotateCcw className="hud-icon" />
            <span>RESET</span>
          </button>
        </div>

        <div className="hud-btn-cluster">
          <button
            type="button"
            className="hud-btn"
            onClick={() => changeView("front")}
          >
            <Eye className="hud-icon" />
            <span>FRONT</span>
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => changeView("side")}
          >
            <Eye className="hud-icon" />
            <span>SIDE</span>
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => changeView("top")}
          >
            <Eye className="hud-icon" />
            <span>TOP</span>
          </button>
        </div>
      </div>

      {/* ── 3D WebGL Canvas ──────────────────────────────────────────────── */}
      <div className="canvas-wrap">
        <Canvas
          dpr={[1, 2]}
          camera={{ position: [4.8, 2.8, 5.4], fov: 36 }}
          gl={{
            antialias: true,
            alpha: true,
            powerPreference: "high-performance",
          }}
          onPointerMissed={() => {
            setHoveredZone(null);
            if (activeTool === "select") onSelectComponent("assembly");
          }}
        >
          {/* ── Balanced Photorealistic Aerospace Lab Lighting ────────── */}
          {/* Ambient — high-contrast fill for aerospace dark theme */}
          <ambientLight intensity={1.2} color="#d8e4ee" />

          {/* Key light — crisp overhead light tuned for medium machined aluminium */}
          <directionalLight
            position={[4, 9, 4]}
            intensity={2.8}
            color="#f4f8fc"
            castShadow
            shadow-mapSize={[2048, 2048]}
            shadow-camera-near={0.5}
            shadow-camera-far={20}
            shadow-camera-left={-5}
            shadow-camera-right={5}
            shadow-camera-top={5}
            shadow-camera-bottom={-5}
            shadow-bias={-0.0004}
          />

          {/* Fill — soft cool neutral fill */}
          <directionalLight position={[-7, 4, -2]} intensity={1.6} color="#a0b4c4" />

          {/* Rim / back-kick — defines machined metal silhouettes */}
          <directionalLight position={[1, -2, -8]} intensity={1.8} color="#c8d8e6" />

          {/* Under-bounce — ground reflection */}
          <directionalLight position={[0, -8, 2]} intensity={0.8} color="#8a9ca8" />

          {/* Side rim — catches edge chamfers */}
          <directionalLight position={[9, 2, 0]} intensity={1.4} color="#d8e2ec" />

          {/* IBL Procedural Environment — studio reflections with 0 external network dependencies */}
          <Environment resolution={256}>
            <Lightformer intensity={3.5} position={[0, 6, 2]} scale={[10, 10, 1]} />
            <Lightformer intensity={2.0} position={[-6, 1, -1]} rotation-y={Math.PI / 2} scale={[10, 4, 1]} color="#a9d4dd" />
            <Lightformer intensity={2.0} position={[6, 1, -1]} rotation-y={-Math.PI / 2} scale={[10, 4, 1]} color="#d0e0ec" />
            <Lightformer intensity={1.5} position={[0, -4, 0]} scale={[10, 10, 1]} />
          </Environment>

          <Suspense fallback={null}>
            <EngineModel
              selectedId={selectedId}
              hoveredZone={hoveredZone}
              hiddenZones={hiddenZones}
              isolatedZone={isolatedZone}
              onHover={setHoveredZone}
              onSelect={onSelectComponent}
              onReady={handleModelReady}
              onError={handleModelError}
              explode={explode / 100}
            />

            <PinProjector
              pins={CALLOUT_PINS}
              explode={explode / 100}
              pinRefs={pinRefs}
            />
          </Suspense>

          <ContactShadows position={[0, -1.6, 0]} opacity={0.45} scale={10} blur={2.2} far={5} color="#101820" />

          <OrbitControls
            ref={controlsRef}
            makeDefault
            enableDamping
            dampingFactor={0.06}
            rotateSpeed={0.65}
            zoomSpeed={0.75}
            panSpeed={0.6}
            minDistance={3.2}
            maxDistance={14}
            autoRotate={autoRotate}
            autoRotateSpeed={0.65}
          />

          <CameraRig
            position={cameraRequest.position}
            lookAt={cameraRequest.lookAt}
            transitionKey={cameraRequest.key}
            controlsRef={controlsRef}
          />
        </Canvas>

        {/* ── 3D Component Callout Leader Pins (Direct DOM Updates via pinRefs) ── */}
        <div className="callout-overlay" aria-hidden="true">
          {CALLOUT_PINS.map((pin) => {
            const isSelected = selectedId === pin.zone;
            const isHovered = hoveredZone === pin.zone;

            return (
              <div
                key={pin.id}
                ref={(el) => {
                  pinRefs.current[pin.id] = el;
                }}
                className={`callout-pin pin-${pin.placement} ${isSelected ? "pin-selected" : ""} ${isHovered ? "pin-hovered" : ""}`}
                style={{ display: "none" }}
                onClick={() => handlePinClick(pin.zone)}
                onMouseEnter={() => setHoveredZone(pin.zone)}
                onMouseLeave={() => setHoveredZone(null)}
              >
                <div className="pin-dot" />
                <div className="pin-line" />
                <div className="pin-tag">
                  <span>{pin.label}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Loading Overlay ────────────────────────────────────────────── */}
        {!modelReady && !loadError && (
          <div className="engine-loader">
            <div className="loader-rings">
              <span />
              <span />
              <span />
            </div>
            <strong>LOADING ENGINE MODEL...</strong>
            <p>Initializing twin-piston aerospace digital twin telemetry mesh...</p>
          </div>
        )}

        {/* ── Error Overlay ──────────────────────────────────────────────── */}
        {loadError && (
          <div className="engine-loader engine-error">
            <strong className="text-amber-400">ENGINE MODEL FAILED TO LOAD</strong>
            <p>Procedural engineering telemetry geometry active.</p>
          </div>
        )}

        {/* ── Corner Crosshairs / Reticles ───────────────────────────────── */}
        <div className="reticle reticle-tl" />
        <div className="reticle reticle-tr" />
        <div className="reticle reticle-bl" />
        <div className="reticle reticle-br" />
      </div>

      {/* ── Bottom Exploded & Inspection Toolbar ─────────────────────────── */}
      <div className="viewer-bottombar-hud">
        <div className="tool-cluster">
          <button
            type="button"
            className={`tool-action-btn ${activeTool === "select" ? "tool-btn-active" : ""}`}
            onClick={() => handleToolAction("select")}
          >
            <CheckCircle2 className="tool-icon" />
            <span>SELECT</span>
          </button>
          <button
            type="button"
            className={`tool-action-btn ${activeTool === "hide" ? "tool-btn-active" : ""}`}
            onClick={() => handleToolAction("hide")}
          >
            <EyeOff className="tool-icon" />
            <span>HIDE</span>
          </button>
          <button
            type="button"
            className={`tool-action-btn ${isolatedZone ? "tool-btn-active" : ""}`}
            onClick={() => handleToolAction("isolate")}
          >
            <Crosshair className="tool-icon" />
            <span>ISOLATE</span>
          </button>
        </div>

        <div className="slider-cluster">
          <div className="view-mode-toggle">
            <button
              type="button"
              className={explode === 0 ? "vmode-active" : ""}
              onClick={() => setExplode(0)}
            >
              Assembled View
            </button>
            <button
              type="button"
              className={explode > 0 ? "vmode-active" : ""}
              onClick={() => setExplode(100)}
            >
              Exploded Assembly
            </button>
          </div>

          <div className="slider-field">
            <span className="slider-label">EXPLOSION: 0% — 100%</span>
            <input
              type="range"
              min="0"
              max="100"
              value={explode}
              onChange={(e) => setExplode(Number(e.target.value))}
              aria-label="Explosion separation percentage"
            />
            <strong className="slider-val">{explode}%</strong>
          </div>
        </div>
      </div>
    </section>
  );
}
