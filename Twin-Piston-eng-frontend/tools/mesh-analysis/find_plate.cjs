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

// Let's inspect the triangles in the GLB to see what forms the plate in the background.
// In 3D space, which triangles have Z close to the back or form a large flat quad / box behind the engine?
// Let's examine triangles near min/max X, Y, Z.
// Let's check the distribution of Z coordinates:
const zBuckets = new Array(20).fill(0);
let minZ = -0.771, maxZ = 0.782;
for (let i = 0; i < pos.length; i += 3) {
  const z = pos[i+2];
  const b = Math.min(19, Math.max(0, Math.floor(((z - minZ) / (maxZ - minZ)) * 20)));
  zBuckets[b]++;
}
console.log('Z buckets (from minZ to maxZ):', zBuckets);

// Let's find triangles that have nearly constant Z or are a flat rectangular plane
// Look at the screenshot: in media_1789489392055.png, the plate is at the back.
// Let's check which component from analyze.cjs corresponds to it!
