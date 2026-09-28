const fs = require('fs');
const buf = fs.readFileSync('public/twin-piston-engine.glb');
const jsonChunkLen = buf.readUInt32LE(12);
const jsonStr = buf.toString('utf8', 20, 20 + jsonChunkLen);
const gltf = JSON.parse(jsonStr);

const binChunkOffset = 20 + jsonChunkLen;
const binChunkLen = buf.readUInt32LE(binChunkOffset);
const binData = buf.subarray(binChunkOffset + 8, binChunkOffset + 8 + binChunkLen);

const posView = gltf.bufferViews[gltf.accessors[0].bufferView];
const posOffset = (posView.byteOffset || 0) + (gltf.accessors[0].byteOffset || 0);
const pos = new Float32Array(binData.buffer, binData.byteOffset + posOffset, gltf.accessors[0].count * 3);

const idxView = gltf.bufferViews[gltf.accessors[1].bufferView];
const idxOffset = (idxView.byteOffset || 0) + (gltf.accessors[1].byteOffset || 0);
const idx = new Uint32Array(binData.buffer, binData.byteOffset + idxOffset, gltf.accessors[1].count);

const grid = new Map();
function hashPos(x, y, z) {
  const qx = Math.round(x * 2000);
  const qy = Math.round(y * 2000);
  const qz = Math.round(z * 2000);
  return qx + '_' + qy + '_' + qz;
}

const vCount = pos.length / 3;
const parent = new Int32Array(vCount);
for (let i = 0; i < vCount; i++) parent[i] = i;

function find(i) {
  let root = i;
  while (root !== parent[root]) root = parent[root];
  let curr = i;
  while (curr !== root) {
    let next = parent[curr];
    parent[curr] = root;
    curr = next;
  }
  return root;
}
function union(i, j) {
  const rootI = find(i);
  const rootJ = find(j);
  if (rootI !== rootJ) parent[rootI] = rootJ;
}

for (let i = 0; i < vCount; i++) {
  const key = hashPos(pos[i*3], pos[i*3+1], pos[i*3+2]);
  if (grid.has(key)) {
    union(i, grid.get(key));
  } else {
    grid.set(key, i);
  }
}

const tCount = idx.length / 3;
for (let t = 0; t < tCount; t++) {
  const a = idx[t*3], b = idx[t*3+1], c = idx[t*3+2];
  union(a, b);
  union(b, c);
}

const compTris = new Map();
const compBounds = new Map();

for (let t = 0; t < tCount; t++) {
  const a = idx[t*3];
  const root = find(a);
  if (!compTris.has(root)) {
    compTris.set(root, []);
    compBounds.set(root, { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity, count: 0 });
  }
  compTris.get(root).push(t);
  const b = compBounds.get(root);
  for (let c = 0; c < 3; c++) {
    const v = idx[t*3+c];
    const x = pos[v*3], y = pos[v*3+1], z = pos[v*3+2];
    if (x < b.minX) b.minX = x; if (x > b.maxX) b.maxX = x;
    if (y < b.minY) b.minY = y; if (y > b.maxY) b.maxY = y;
    if (z < b.minZ) b.minZ = z; if (z > b.maxZ) b.maxZ = z;
  }
  b.count++;
}

// Classify connected components by their bounding box and center
function classifyComponent(b) {
  const widthX = b.maxX - b.minX;
  const heightY = b.maxY - b.minY;
  const depthZ = b.maxZ - b.minZ;
  const cx = (b.minX + b.maxX) / 2;
  const cy = (b.minY + b.maxY) / 2;
  const cz = (b.minZ + b.maxZ) / 2;

  // Backdrop plate: very wide in X and sits at the back
  if (widthX > 1.2 && cz < -0.3) return "BACKDROP_PLATE_REMOVE";

  // ECU/FADEC: top-right electronics box
  if (cy > 0.25 && cx > 0.15 && cz < -0.2) return "ecu-fadec";

  // Throttle body & Intake manifold: top center
  if (cy > 0.25 && Math.abs(cx) < 0.25 && cz < -0.15) return "throttle-body";
  if (cy > 0.10 && Math.abs(cx) < 0.35 && cz < -0.15) return "intake-manifold";

  // Oil filter: bottom left canister
  if (cy < -0.25 && cx < 0 && cz > 0.2) return "oil-filter";

  // Oil pump: bottom center
  if (cy < -0.25 && cx >= 0 && cz > 0.1) return "oil-pump";

  // Exhaust manifold: lower piping
  if (cy < -0.25 && cz < -0.1) return "exhaust-manifold";

  // Propeller drive / flywheel: front nose / drive hub
  if (cz > 0.35 && Math.abs(cx) < 0.35) return "propeller-drive";

  // Left cylinder & head (X < -0.3)
  if (cx < -0.3) {
    if (cx < -0.7) return "cylinder-head-1";
    if (cy > 0.2) return "spark-plug-1";
    return "cylinder-1";
  }

  // Right cylinder & head (X > 0.3)
  if (cx > 0.3) {
    if (cx > 0.7) return "cylinder-head-2";
    if (cy > 0.2) return "spark-plug-2";
    return "cylinder-2";
  }

  // Default central block
  return "crankcase";
}

const classified = new Map();
for (const [r, b] of compBounds.entries()) {
  const zone = classifyComponent(b);
  if (!classified.has(zone)) classified.set(zone, []);
  classified.get(zone).push({ root: r, tris: b.count, bounds: b });
}

for (const [zone, items] of classified.entries()) {
  const totalTris = items.reduce((s, it) => s + it.tris, 0);
  console.log(`Zone [${zone}]: ${items.length} components, total tris=${totalTris}`);
}
