import fs from 'fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three-stdlib';

const loader = new GLTFLoader();
const buffer = fs.readFileSync('public/twin-piston-engine.glb');
const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);

loader.parse(
  arrayBuffer,
  '',
  (gltf) => {
    console.log('Scene children:', gltf.scene.children.length);
    let meshCount = 0;
    gltf.scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        meshCount++;
        child.geometry.computeBoundingBox();
        const bbox = child.geometry.boundingBox;
        const size = bbox.getSize(new THREE.Vector3());
        const center = bbox.getCenter(new THREE.Vector3());
        console.log(`Mesh ${meshCount}: name="${child.name}" tri=${child.geometry.attributes.position.count / 3}`);
        console.log(`   size: [${size.x.toFixed(3)}, ${size.y.toFixed(3)}, ${size.z.toFixed(3)}]`);
        console.log(`   center: [${center.x.toFixed(3)}, ${center.y.toFixed(3)}, ${center.z.toFixed(3)}]`);
      }
    });
  },
  (err) => {
    console.error('Parse error:', err);
  }
);
