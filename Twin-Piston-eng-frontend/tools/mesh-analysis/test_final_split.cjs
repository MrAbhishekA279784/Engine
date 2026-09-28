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

function testAlgorithm() {
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
    if (grid.has(key)) union(i, grid.get(key));
    else grid.set(key, i);
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

  // Classify each component into zones
  const zoneTris = new Map();

  function classify(b) {
    const dx = b.maxX - b.minX;
    const dy = b.maxY - b.minY;
    const dz = b.maxZ - b.minZ;
    const cx = (b.minX + b.maxX) / 2;
    const cy = (b.minY + b.maxY) / 2;
    const cz = (b.minZ + b.maxZ) / 2;

    // Filter out backdrop plate: wide X and at the back
    if (dx > 1.2 && cz < -0.3) return null;

    // Top parts
    if (cy > 0.22) {
      if (cx > 0.15 && cz < -0.2) return "ecu-fadec";
      if (Math.abs(cx) < 0.25) return "throttle-body";
      if (cx < -0.3) return "spark-plug-1";
      if (cx > 0.3) return "spark-plug-2";
    }

    // Upper center
    if (cy > 0.08 && Math.abs(cx) < 0.35 && cz < -0.1) return "intake-manifold";

    // Bottom parts
    if (cy < -0.22) {
      if (cx < -0.15 && cz > 0.15) return "oil-filter";
      if (cx >= -0.15 && cx <= 0.25 && cz > 0.1) return "oil-pump";
      if (cz < -0.1) return "exhaust-manifold";
    }

    // Front nose / hub
    if (cz > 0.35 && Math.abs(cx) < 0.35) return "propeller-drive";

    // Cylinders and heads
    if (cx < -0.25) {
      if (cx < -0.75 || (b.minX < -0.85 && dx < 0.35)) return "cylinder-head-1";
      return "cylinder-1";
    }

    if (cx > 0.25) {
      if (cx > 0.75 || (b.maxX > 0.85 && dx < 0.35)) return "cylinder-head-2";
      return "cylinder-2";
    }

    return "crankcase";
  }

  for (const [r, b] of compBounds.entries()) {
    const zone = classify(b);
    if (!zone) continue; // Plate removed!
    const tris = compTris.get(r);
    if (!zoneTris.has(zone)) zoneTris.set(zone, []);
    zoneTris.get(zone).push(...tris);
  }

  console.log('--- CLASSIFICATION RESULTS ---');
  let sum = 0;
  for (const [zone, tris] of zoneTris.entries()) {
    console.log(`Zone: ${zone.padEnd(18)} -> ${tris.length} triangles`);
    sum += tris.length;
  }
  console.log('Total triangles rendered:', sum);
}

testAlgorithm();
