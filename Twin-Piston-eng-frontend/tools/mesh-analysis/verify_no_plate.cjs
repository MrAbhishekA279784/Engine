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
for (let t = 0; t < tCount; t++) {
  const r = find(idx[t*3]);
  if (!compTris.has(r)) compTris.set(r, []);
  compTris.get(r).push(t);
}

console.log('--- ALL COMPONENTS WITHOUT ROOT 79840 ---');
let totalEngineTris = 0;
for (const [r, tris] of compTris.entries()) {
  if (r === 79840) continue;
  totalEngineTris += tris.length;
}
console.log('Total triangles in original GLB:', tCount);
console.log('Triangles in backdrop plate Root 79840:', compTris.get(79840)?.length);
console.log('Total triangles in remaining engine parts:', totalEngineTris);
