const fs = require('fs');
const THREE = require('three');

const data = fs.readFileSync('public/twin-piston-engine.glb');
const jsonChunkLen = data.readUInt32LE(12);
const jsonString = data.toString('utf8', 20, 20 + jsonChunkLen);
const gltf = JSON.parse(jsonString);

const binOffset = 20 + jsonChunkLen;
const binData = data.subarray(binOffset + 8);

const posAccessor = gltf.accessors[0];
const posBufView = gltf.bufferViews[posAccessor.bufferView];
const posByteOffset = (posBufView.byteOffset || 0) + (posAccessor.byteOffset || 0);
const posByteLength = posAccessor.count * 3 * 4;
const posSlice = binData.subarray(posByteOffset, posByteOffset + posByteLength);
const posFloats = new Float32Array(posSlice.buffer, posSlice.byteOffset, posAccessor.count * 3);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(posFloats, 3));

const position = geometry.getAttribute("position");
const index = geometry.getIndex();
const triangleCount = index ? index.count / 3 : position.count / 3;
const vertexOf = (t, corner) => (index ? index.getX(t * 3 + corner) : t * 3 + corner);

const weld = new Map();
const parent = new Int32Array(position.count);
for (let i = 0; i < position.count; i += 1) {
  const key = `${Math.round(position.getX(i) * 1e4)},${Math.round(position.getY(i) * 1e4)},${Math.round(position.getZ(i) * 1e4)}`;
  const existing = weld.get(key);
  if (existing === undefined) {
    weld.set(key, i);
    parent[i] = i;
  } else {
    parent[i] = existing;
  }
}
const find = (x) => {
  let root = x;
  let up = parent[root];
  while (up !== root) {
    const grand = parent[up];
    parent[root] = grand;
    root = up;
    up = parent[root];
  }
  return root;
};
const union = (a, b) => {
  const ra = find(a);
  const rb = find(b);
  if (ra !== rb) parent[ra] = rb;
};
for (let t = 0; t < triangleCount; t += 1) {
  const a = vertexOf(t, 0);
  const b = vertexOf(t, 1);
  const c = vertexOf(t, 2);
  union(a, b);
  union(b, c);
}

const buckets = new Map();
for (let t = 0; t < triangleCount; t += 1) {
  const root = find(vertexOf(t, 0));
  const bucket = buckets.get(root);
  if (bucket) bucket.push(t);
  else buckets.set(root, [t]);
}

console.log('Buckets count:', buckets.size);

const parts = [];
buckets.forEach((triangles, root) => {
  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  triangles.forEach((t) => {
    for (let corner = 0; corner < 3; corner += 1) {
      const v = vertexOf(t, corner);
      point.set(position.getX(v), position.getY(v), position.getZ(v));
      box.expandByPoint(point);
    }
  });

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  // Backdrop plate check
  const isPlate = size.x > 1.2 || (size.x > 0.9 && size.y > 0.55 && size.z < 0.45) || (size.z > 1.3 && size.y > 0.5) || (size.x > 0.8 && size.y > 0.5 && size.z < 0.2);
  parts.push({
    root,
    triCount: triangles.length,
    size: [Number(size.x.toFixed(3)), Number(size.y.toFixed(3)), Number(size.z.toFixed(3))],
    center: [Number(center.x.toFixed(3)), Number(center.y.toFixed(3)), Number(center.z.toFixed(3))],
    isPlate
  });
});

parts.sort((a, b) => b.triCount - a.triCount);
console.log('Top 15 parts:');
parts.slice(0, 15).forEach((p, i) => console.log(i + 1, p));
console.log('Total parts kept:', parts.filter(p => !p.isPlate).length);
console.log('Total plates dropped:', parts.filter(p => p.isPlate).length);
