import * as THREE from "three";
import { Sky } from "three/addons/objects/Sky.js";

/** Late-afternoon sun over the Pacific, sky, haze and the sea. */
export function buildWorld(scene: THREE.Scene, renderer: THREE.WebGLRenderer) {
  // Sun low in the west-southwest, as at sunset behind the Golden Gate.
  const sunElevation = 7;
  const sunAzimuth = 248;
  const sunDir = new THREE.Vector3().setFromSphericalCoords(
    1,
    THREE.MathUtils.degToRad(90 - sunElevation),
    THREE.MathUtils.degToRad(sunAzimuth),
  );
  // setFromSphericalCoords measures theta from +z toward +x; our north is -z, so azimuth
  // 0 would point south. Flip z so the bearing is a compass bearing.
  sunDir.z = -sunDir.z;

  const sky = new Sky();
  sky.scale.setScalar(60000);
  const u = sky.material.uniforms;
  u.turbidity.value = 7;
  u.rayleigh.value = 2.2;
  u.mieCoefficient.value = 0.006;
  u.mieDirectionalG.value = 0.85;
  u.sunPosition.value.copy(sunDir);
  scene.add(sky);

  // Image-based lighting and reflections from the sky alone.
  const pmrem = new THREE.PMREMGenerator(renderer);
  const skyScene = new THREE.Scene();
  const skyCopy = new Sky();
  skyCopy.scale.setScalar(60000);
  Object.assign(skyCopy.material.uniforms, THREE.UniformsUtils.clone(u));
  skyCopy.material.uniforms.sunPosition.value.copy(sunDir);
  skyScene.add(skyCopy);
  scene.environment = pmrem.fromScene(skyScene).texture;

  const sun = new THREE.DirectionalLight("#ffd7a8", 2.4);
  sun.position.copy(sunDir).multiplyScalar(5000);
  scene.add(sun);
  scene.add(new THREE.HemisphereLight("#bcd3ea", "#5a4a3a", 0.5));

  // Warm marine haze; distant hills fade into it.
  scene.fog = new THREE.FogExp2("#cfb59a", 0.000075);

  const water = buildWater();
  scene.add(water.mesh);
  return { sunDir, update: water.update };
}

function buildWater() {
  const normalMap = waveNormalMap();
  normalMap.wrapS = normalMap.wrapT = THREE.RepeatWrapping;
  normalMap.repeat.set(220, 220);
  const mat = new THREE.MeshPhysicalMaterial({
    color: "#123848",
    roughness: 0.12,
    metalness: 0,
    normalMap,
    normalScale: new THREE.Vector2(0.55, 0.55),
    envMapIntensity: 1.1,
    clearcoat: 0.6,
    clearcoatRoughness: 0.2,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(80000, 80000), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0;
  return {
    mesh,
    update(t: number) {
      normalMap.offset.set(t * 0.004, t * 0.0025);
    },
  };
}

/** Tileable wave normals from a few crossing sine swells, drawn once. */
function waveNormalMap() {
  const n = 256;
  const h = new Float32Array(n * n);
  const waves = [
    [3, 1, 0.0], [2, -3, 1.3], [5, 2, 2.1], [-4, 5, 0.7], [7, -2, 3.0], [1, 6, 4.2],
  ];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let v = 0;
      for (const [kx, ky, ph] of waves) v += Math.sin(((kx * x + ky * y) / n) * Math.PI * 2 + ph) / Math.hypot(kx, ky);
      h[y * n + x] = v;
    }
  }
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(n, n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const dx = h[y * n + ((x + 1) % n)] - h[y * n + ((x - 1 + n) % n)];
      const dy = h[((y + 1) % n) * n + x] - h[((y - 1 + n) % n) * n + x];
      const len = Math.hypot(dx, dy, 1);
      const i = (y * n + x) * 4;
      img.data[i] = ((-dx / len) * 0.5 + 0.5) * 255;
      img.data[i + 1] = ((-dy / len) * 0.5 + 0.5) * 255;
      img.data[i + 2] = ((1 / len) * 0.5 + 0.5) * 255;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return new THREE.CanvasTexture(c);
}
