import * as THREE from "three";

export type EngineZone =
  | "assembly"
  | "crankcase"
  | "cylinder-1"
  | "cylinder-2"
  | "cylinder-head-1"
  | "cylinder-head-2"
  | "spark-plug-1"
  | "spark-plug-2"
  | "fuel-injector-1"
  | "fuel-injector-2"
  | "throttle-body"
  | "intake-manifold"
  | "exhaust-manifold"
  | "oil-pump"
  | "oil-filter"
  | "ecu-fadec"
  | "propeller-drive"
  | "cooling-fins"
  | "sensors";

export type EnginePart = {
  id: string;
  geometry: THREE.BufferGeometry;
  center: THREE.Vector3;
  explodeOffset: THREE.Vector3;
  zone: string;
  triangles: number;
};

export const ZONE_OFFSETS: Record<string, THREE.Vector3> = {
  assembly: new THREE.Vector3(0, 0, 0),
  crankcase: new THREE.Vector3(0, 0, 0), // Reference stationary frame
  "cylinder-1": new THREE.Vector3(-0.95, 0, 0), // Displace along port cylinder axis
  "cylinder-head-1": new THREE.Vector3(-1.85, 0, 0), // Displace further along cylinder axis
  "spark-plug-1": new THREE.Vector3(-2.30, 0.20, 0.05), // Installation direction from head
  "fuel-injector-1": new THREE.Vector3(-1.40, 0.40, 0), // Mounting direction into port
  "cylinder-2": new THREE.Vector3(0.95, 0, 0), // Displace along starboard cylinder axis
  "cylinder-head-2": new THREE.Vector3(1.85, 0, 0), // Displace further along cylinder axis
  "spark-plug-2": new THREE.Vector3(2.30, 0.20, 0.05), // Installation direction from head
  "fuel-injector-2": new THREE.Vector3(1.40, 0.40, 0), // Mounting direction into port
  "throttle-body": new THREE.Vector3(0, 1.55, -0.30), // Mounts onto intake manifold
  "intake-manifold": new THREE.Vector3(0, 0.85, -0.20), // Moves up away from crankcase top
  "exhaust-manifold": new THREE.Vector3(0, -0.90, -0.25), // Moves down away from cylinder exhausts
  "oil-pump": new THREE.Vector3(0, -0.80, 0.15), // Moves down along accessory drive axis
  "oil-filter": new THREE.Vector3(-0.45, -1.25, 0.30), // Moves down and outward from filter boss
  "ecu-fadec": new THREE.Vector3(0.35, 1.10, -0.35), // Moves up away from avionics mount
  "propeller-drive": new THREE.Vector3(0, 0, 0.95), // Moves forward along crankshaft nose axis
  "cooling-fins": new THREE.Vector3(0, 0, 0),
  sensors: new THREE.Vector3(0, 0.50, 0.10),
};

export type CalloutPin = {
  id: string;
  label: string;
  zone: string;
  anchor: [number, number, number];
  placement: "left" | "right" | "top" | "bottom";
};

export const CALLOUT_PINS: CalloutPin[] = [
  { id: "pin-throttle", label: "Throttle Body", zone: "throttle-body", anchor: [-0.04, 0.42, -0.38], placement: "top" },
  { id: "pin-intake", label: "Intake Manifold", zone: "intake-manifold", anchor: [-0.02, 0.26, -0.28], placement: "top" },
  { id: "pin-ecu", label: "ECU / FADEC", zone: "ecu-fadec", anchor: [0.28, 0.38, -0.44], placement: "top" },
  { id: "pin-head-1", label: "Cylinder Head (01)", zone: "cylinder-head-1", anchor: [-0.86, 0.16, 0.12], placement: "left" },
  { id: "pin-head-2", label: "Cylinder Head (02)", zone: "cylinder-head-2", anchor: [0.86, -0.32, 0.26], placement: "right" },
  { id: "pin-cyl-1", label: "Cylinder (01)", zone: "cylinder-1", anchor: [-0.63, 0.08, 0.03], placement: "left" },
  { id: "pin-cyl-2", label: "Cylinder (02)", zone: "cylinder-2", anchor: [0.55, -0.20, -0.10], placement: "right" },
  { id: "pin-case", label: "Crankcase", zone: "crankcase", anchor: [-0.08, -0.05, -0.35], placement: "bottom" },
  { id: "pin-pump", label: "Oil Pump", zone: "oil-pump", anchor: [0.00, -0.38, 0.35], placement: "bottom" },
  { id: "pin-filter", label: "Oil Filter", zone: "oil-filter", anchor: [-0.55, -0.36, 0.58], placement: "bottom" },
  { id: "pin-exhaust", label: "Exhaust Manifold", zone: "exhaust-manifold", anchor: [0.38, -0.34, -0.28], placement: "right" },
];

/**
 * Split the raw GLB geometry into per-zone merged meshes using Connected Component Analysis.
 * Discards the backdrop plate and merges solid components cleanly into engine zones.
 */
export function splitEngineParts(source: THREE.BufferGeometry): EnginePart[] {
  const position = source.getAttribute("position") as THREE.BufferAttribute;
  const normal = source.getAttribute("normal") as THREE.BufferAttribute | undefined;
  const uv = source.getAttribute("uv") as THREE.BufferAttribute | undefined;
  const index = source.getIndex();
  const triangleCount = index ? index.count / 3 : position.count / 3;
  const vertexOf = (t: number, corner: number) =>
    index ? index.getX(t * 3 + corner) : t * 3 + corner;

  const vCount = position.count;
  const parent = new Int32Array(vCount);
  for (let i = 0; i < vCount; i++) parent[i] = i;

  function find(i: number): number {
    let root = i;
    while (root !== parent[root]!) root = parent[root]!;
    let curr = i;
    while (curr !== root) {
      const next = parent[curr]!;
      parent[curr] = root;
      curr = next;
    }
    return root;
  }

  function union(i: number, j: number) {
    const rootI = find(i);
    const rootJ = find(j);
    if (rootI !== rootJ) parent[rootI] = rootJ;
  }

  // 1. Spatial hashing to weld coincident vertices (mesh shell connectivity)
  const grid = new Map<string, number>();
  for (let i = 0; i < vCount; i++) {
    const qx = Math.round(position.getX(i) * 2000);
    const qy = Math.round(position.getY(i) * 2000);
    const qz = Math.round(position.getZ(i) * 2000);
    const key = `${qx}_${qy}_${qz}`;
    const existing = grid.get(key);
    if (existing !== undefined) {
      union(i, existing);
    } else {
      grid.set(key, i);
    }
  }

  // 2. Union vertices connected by triangles
  for (let t = 0; t < triangleCount; t++) {
    const a = vertexOf(t, 0);
    const b = vertexOf(t, 1);
    const c = vertexOf(t, 2);
    union(a, b);
    union(b, c);
  }

  // 3. Group triangles by connected component root and compute bounding boxes
  const compTris = new Map<number, number[]>();
  const compBounds = new Map<
    number,
    { minX: number; maxX: number; minY: number; maxY: number; minZ: number; maxZ: number }
  >();

  for (let t = 0; t < triangleCount; t++) {
    const a = vertexOf(t, 0);
    const root = find(a);
    let arr = compTris.get(root);
    if (!arr) {
      arr = [];
      compTris.set(root, arr);
      compBounds.set(root, {
        minX: Infinity,
        maxX: -Infinity,
        minY: Infinity,
        maxY: -Infinity,
        minZ: Infinity,
        maxZ: -Infinity,
      });
    }
    arr.push(t);

    const b = compBounds.get(root)!;
    for (let c = 0; c < 3; c++) {
      const v = vertexOf(t, c);
      const x = position.getX(v);
      const y = position.getY(v);
      const z = position.getZ(v);
      if (x < b.minX) b.minX = x;
      if (x > b.maxX) b.maxX = x;
      if (y < b.minY) b.minY = y;
      if (y > b.maxY) b.maxY = y;
      if (z < b.minZ) b.minZ = z;
      if (z > b.maxZ) b.maxZ = z;
    }
  }

  // 4. Classify each connected component into an engine zone
  function classifyComponent(b: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    minZ: number;
    maxZ: number;
  }): string | null {
    const dx = b.maxX - b.minX;
    const dy = b.maxY - b.minY;
    const cx = (b.minX + b.maxX) / 2;
    const cy = (b.minY + b.maxY) / 2;
    const cz = (b.minZ + b.maxZ) / 2;

    // ── Backdrop / mount plate removal ─────────────────────────────────────
    // (a) Spans >90% of model width AND sits in the back half
    if (dx > 0.9 && cz < -0.2) return null;
    // (b) Flat planar sheet: very wide but almost no Y thickness
    if (dx > 0.7 && dy < dx * 0.05) return null;

    // Top parts (ECU, throttle, intake)
    if (cy > 0.22) {
      if (cx > 0.15 && cz < -0.2) return "ecu-fadec";
      if (Math.abs(cx) < 0.25) return "throttle-body";
      if (cx < -0.3) return "spark-plug-1";
      if (cx > 0.3) return "spark-plug-2";
    }

    if (cy > 0.08 && Math.abs(cx) < 0.35 && cz < -0.1) return "intake-manifold";

    // Bottom parts (Oil filter, oil pump, exhaust)
    if (cy < -0.22) {
      if (cx < -0.15 && cz > 0.15) return "oil-filter";
      if (cx >= -0.15 && cx <= 0.25 && cz > 0.1) return "oil-pump";
      if (cz < -0.1) return "exhaust-manifold";
    }

    // Front nose / propeller hub
    if (cz > 0.35 && Math.abs(cx) < 0.35) return "propeller-drive";

    // Left cylinder & head
    if (cx < -0.25) {
      if (cx < -0.75 || (b.minX < -0.85 && dx < 0.35)) return "cylinder-head-1";
      return "cylinder-1";
    }

    // Right cylinder & head
    if (cx > 0.25) {
      if (cx > 0.75 || (b.maxX > 0.85 && dx < 0.35)) return "cylinder-head-2";
      return "cylinder-2";
    }

    return "crankcase";
  }

  // 5. Accumulate triangles per zone
  const zoneTris = new Map<string, number[]>();
  compBounds.forEach((b, root) => {
    const zone = classifyComponent(b);
    if (!zone) return; // Discard backdrop plate
    const tris = compTris.get(root);
    if (!tris || tris.length === 0) return;
    let list = zoneTris.get(zone);
    if (!list) {
      list = [];
      zoneTris.set(zone, list);
    }
    list.push(...tris);
  });

  // 6. Build BufferGeometry for each zone
  const parts: EnginePart[] = [];
  let n = 0;

  zoneTris.forEach((triangles, zone) => {
    if (triangles.length === 0) return;

    const positions = new Float32Array(triangles.length * 9);
    const normals = normal ? new Float32Array(triangles.length * 9) : null;
    const uvs = uv ? new Float32Array(triangles.length * 6) : null;
    const box = new THREE.Box3();
    const point = new THREE.Vector3();

    triangles.forEach((t, ti) => {
      for (let corner = 0; corner < 3; corner++) {
        const v = vertexOf(t, corner);
        const off = ti * 9 + corner * 3;
        const px = position.getX(v);
        const py = position.getY(v);
        const pz = position.getZ(v);
        positions[off] = px;
        positions[off + 1] = py;
        positions[off + 2] = pz;
        point.set(px, py, pz);
        box.expandByPoint(point);
        if (normals && normal) {
          normals[off] = normal.getX(v);
          normals[off + 1] = normal.getY(v);
          normals[off + 2] = normal.getZ(v);
        }
        if (uvs && uv) {
          const uvOff = ti * 6 + corner * 2;
          uvs[uvOff] = uv.getX(v);
          uvs[uvOff + 1] = uv.getY(v);
        }
      }
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (normals) {
      geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    } else {
      geo.computeVertexNormals();
    }
    if (uvs) geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

    geo.computeBoundingBox();
    geo.computeBoundingSphere();

    const center = box.getCenter(new THREE.Vector3());
    n += 1;
    parts.push({
      id: `zone-${zone}-${n}`,
      geometry: geo,
      center,
      explodeOffset: (ZONE_OFFSETS[zone] ?? ZONE_OFFSETS["crankcase"] ?? new THREE.Vector3()).clone(),
      zone,
      triangles: triangles.length,
    });
  });

  return parts.sort((a, b) => b.triangles - a.triangles);
}

export function createProceduralEngineParts(): EnginePart[] {
  const parts: EnginePart[] = [];
  let partIndex = 0;

  const addPart = (geo: THREE.BufferGeometry, zone: string) => {
    geo.computeVertexNormals();
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    const center = geo.boundingBox ? geo.boundingBox.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const index = geo.getIndex();
    const pos = geo.getAttribute("position") as THREE.BufferAttribute;
    const triangles = index ? index.count / 3 : pos.count / 3;
    partIndex += 1;
    parts.push({
      id: `proc-part-${partIndex}`,
      geometry: geo,
      center,
      explodeOffset: (ZONE_OFFSETS[zone] ?? new THREE.Vector3()).clone(),
      zone,
      triangles,
    });
  };

  // Crankcase
  const block = new THREE.BoxGeometry(1.2, 0.95, 1.35);
  addPart(block, "crankcase");

  const nose = new THREE.CylinderGeometry(0.28, 0.44, 0.75, 24);
  nose.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  nose.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0, -0.85));
  addPart(nose, "crankcase");

  const hub = new THREE.CylinderGeometry(0.12, 0.12, 0.45, 16);
  hub.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  hub.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0, -1.35));
  addPart(hub, "propeller-drive");

  const sump = new THREE.BoxGeometry(0.85, 0.35, 1.1);
  sump.applyMatrix4(new THREE.Matrix4().makeTranslation(0, -0.62, 0.05));
  addPart(sump, "crankcase");

  // Cylinders & Heads
  const cyl1Barrel = new THREE.CylinderGeometry(0.44, 0.44, 0.95, 24);
  cyl1Barrel.applyMatrix4(new THREE.Matrix4().makeRotationZ(Math.PI / 2));
  cyl1Barrel.applyMatrix4(new THREE.Matrix4().makeTranslation(-1.05, 0.05, 0));
  addPart(cyl1Barrel, "cylinder-1");

  for (let i = 0; i < 6; i++) {
    const fin = new THREE.CylinderGeometry(0.56, 0.56, 0.04, 24);
    fin.applyMatrix4(new THREE.Matrix4().makeRotationZ(Math.PI / 2));
    fin.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.72 - i * 0.13, 0.05, 0));
    addPart(fin, "cylinder-1");
  }

  const cyl1Head = new THREE.BoxGeometry(0.45, 0.92, 0.92);
  cyl1Head.applyMatrix4(new THREE.Matrix4().makeTranslation(-1.68, 0.05, 0));
  addPart(cyl1Head, "cylinder-head-1");

  const cyl2Barrel = new THREE.CylinderGeometry(0.44, 0.44, 0.95, 24);
  cyl2Barrel.applyMatrix4(new THREE.Matrix4().makeRotationZ(Math.PI / 2));
  cyl2Barrel.applyMatrix4(new THREE.Matrix4().makeTranslation(1.05, 0.05, 0));
  addPart(cyl2Barrel, "cylinder-2");

  for (let i = 0; i < 6; i++) {
    const fin = new THREE.CylinderGeometry(0.56, 0.56, 0.04, 24);
    fin.applyMatrix4(new THREE.Matrix4().makeRotationZ(Math.PI / 2));
    fin.applyMatrix4(new THREE.Matrix4().makeTranslation(0.72 + i * 0.13, 0.05, 0));
    addPart(fin, "cylinder-2");
  }

  const cyl2Head = new THREE.BoxGeometry(0.45, 0.92, 0.92);
  cyl2Head.applyMatrix4(new THREE.Matrix4().makeTranslation(1.68, 0.05, 0));
  addPart(cyl2Head, "cylinder-head-2");

  // Intake & Throttle
  const plenum = new THREE.BoxGeometry(0.65, 0.32, 0.48);
  plenum.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 0.78, -0.05));
  addPart(plenum, "intake-manifold");

  const throttle = new THREE.CylinderGeometry(0.24, 0.24, 0.4, 20);
  throttle.applyMatrix4(new THREE.Matrix4().makeTranslation(0, 1.08, -0.05));
  addPart(throttle, "throttle-body");

  // ECU
  const ecu = new THREE.BoxGeometry(0.55, 0.35, 0.4);
  ecu.applyMatrix4(new THREE.Matrix4().makeTranslation(0.75, 0.85, 0.25));
  addPart(ecu, "ecu-fadec");

  // Exhaust
  const muffler = new THREE.CylinderGeometry(0.22, 0.22, 1.45, 20);
  muffler.applyMatrix4(new THREE.Matrix4().makeRotationX(Math.PI / 2));
  muffler.applyMatrix4(new THREE.Matrix4().makeTranslation(0, -0.68, 0.65));
  addPart(muffler, "exhaust-manifold");

  // Oil filter & pump
  const filter = new THREE.CylinderGeometry(0.18, 0.18, 0.45, 20);
  filter.applyMatrix4(new THREE.Matrix4().makeTranslation(-0.25, -0.72, 0.28));
  addPart(filter, "oil-filter");

  const pump = new THREE.BoxGeometry(0.3, 0.3, 0.35);
  pump.applyMatrix4(new THREE.Matrix4().makeTranslation(0.12, -0.72, 0.15));
  addPart(pump, "oil-pump");

  return parts;
}
