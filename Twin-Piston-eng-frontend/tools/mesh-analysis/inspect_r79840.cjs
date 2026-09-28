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

// Check root 79840 triangles:
// How many triangles have flat normals along Z or what is root 79840?
let nxSum = 0, nySum = 0, nzSum = 0;
let r79840Count = 0;
for (let t = 0; t < tCount; t++) {
  const i0 = idx[t*3], i1 = idx[t*3+1], i2 = idx[t*3+2];
  if (find(i0) === 79840) {
    r79840Count++;
    const x0 = pos[i0*3], y0 = pos[i0*3+1], z0 = pos[i0*3+2];
    const x1 = pos[i1*3], y1 = pos[i1*3+1], z1 = pos[i1*3+2];
    const x2 = pos[i2*3], y2 = pos[i2*3+1], z2 = pos[i2*3+2];
    const e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
    const e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz) || 1;
    nxSum += Math.abs(nx / len);
    nySum += Math.abs(ny / len);
    nzSum += Math.abs(nz / len);
  }
}
console.log('Root 79840 normal alignment:', {
  avgAbsNx: nxSum / r79840Count,
  avgAbsNy: nySum / r79840Count,
  avgAbsNz: nzSum / r79840Count,
});
