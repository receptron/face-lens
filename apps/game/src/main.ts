import "./style.css";
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { Bridge } from "./bridge";
import { FaceControl, KeyboardControl, merge, type Input } from "./control";
import { Swarm } from "./swarm";
import { Explosions, Targets } from "./targets";
import { Terrain } from "./terrain";
import { buildWorld } from "./world";

const BASE = import.meta.env.BASE_URL;
const ROUND_SECONDS = 150;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// --- Renderer and scene ---------------------------------------------------

const canvas = $<HTMLCanvasElement>("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.6;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(62, 1, 1, 45000);
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
// Threshold above 1: only things drawn brighter than white (orbs, lights) glow, not the sunlit bridge.
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.9, 0.5, 1.0);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

const world = buildWorld(scene, renderer);
const bridge = new Bridge();
scene.add(bridge.group);
const explosions = new Explosions();
scene.add(explosions.points);

// --- Game state -----------------------------------------------------------

// West of the bridge, on a line down the middle of the strait (all water to the main span).
const START = new THREE.Vector3(-3400, 140, 400);
let terrain: Terrain;
let swarm: Swarm;
let targets: Targets;
let yaw = 90; // compass degrees: start heading east, toward the bridge
let pitch = 0;
let score = 0;
let combo = 0;
let comboTimer = 0;
let timeLeft = ROUND_SECONDS;
let running = false;
let wasUnder = false;
let bridgeHitCount = 0;
let hitCount = 0;
let face: FaceControl | null = null;
const keys = new KeyboardControl();
const lockRing = new THREE.Mesh(
  new THREE.TorusGeometry(26, 1.2, 8, 48),
  new THREE.MeshBasicMaterial({ color: "#ffffff", toneMapped: false, transparent: true, opacity: 0.85 }),
);
scene.add(lockRing);

function collides(p: THREE.Vector3) {
  return p.y < Math.max(0, terrain.heightAt(p.x, p.z)) + 0.5 || bridge.hits(p);
}

/**
 * Random spot for an orb, 60–260 m up over water or low hills: a third of them along the
 * strait and around the bridge (where every run starts), the rest within ~7 km of the bay.
 */
function placeOrb() {
  const b = terrain.bounds();
  for (let tries = 0; tries < 50; tries++) {
    const nearBridge = Math.random() < 0.35;
    const r = 7000 * Math.sqrt(Math.random());
    const a = Math.random() * Math.PI * 2;
    const x = nearBridge
      ? THREE.MathUtils.lerp(-3600, 1800, Math.random())
      : THREE.MathUtils.clamp(Math.cos(a) * r + 2500, b.minX + 1500, b.maxX - 1500);
    const z = nearBridge
      ? THREE.MathUtils.lerp(-500, 1300, Math.random())
      : THREE.MathUtils.clamp(Math.sin(a) * r * 0.7 - 1000, b.minZ + 1500, b.maxZ - 1500);
    const ground = Math.max(0, terrain.heightAt(x, z));
    if (ground < 120) return new THREE.Vector3(x, ground + 60 + Math.random() * 200, z);
  }
  return new THREE.Vector3(0, 150, -2000);
}

// --- HUD ------------------------------------------------------------------

let bannerTimer = 0;
function banner(text: string, seconds = 1.6) {
  $("banner").textContent = text;
  $("banner").classList.add("show");
  bannerTimer = seconds;
}

function hud() {
  $("score").textContent = score.toLocaleString();
  $("drones").textContent = String(swarm.count);
  $("time").textContent = `${Math.floor(timeLeft / 60)}:${String(Math.floor(timeLeft % 60)).padStart(2, "0")}`;
  $("combo").textContent = combo > 1 ? `×${combo}` : "";
  if (import.meta.env.DEV) {
    document.body.dataset.state = JSON.stringify({
      score, drones: swarm.count, under: wasUnder, bridgeHits: bridgeHitCount, hits: hitCount, strikeCrashes: swarm.strikeCrashes, ahead: Math.round(swarm.maxAhead()), lateral: Math.round(swarm.offset().lateral), behind: Math.round(swarm.offset().behind), x: Math.round(swarm.leader.x), y: Math.round(swarm.leader.y), z: Math.round(swarm.leader.z), yaw: Math.round(yaw),
    });
  }
}

// --- Loop -----------------------------------------------------------------

const camPos = new THREE.Vector3();
const camLook = new THREE.Vector3();
const centroid = new THREE.Vector3();
let last = performance.now();

function frame(now: number) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  const t = now / 1000;
  world.update(t);

  if (running) step(dt, t, now);
  explosions.update(dt);
  targets?.update(dt, t);

  // Chase camera behind and above the leader, nudged toward the flock so it stays in view.
  swarm.centroid(centroid);
  const focus = swarm.leader.clone().lerp(centroid, 0.35);
  const want = focus.clone().addScaledVector(swarm.heading, -95).add(new THREE.Vector3(0, 30, 0));
  const k = 1 - Math.exp(-dt * 3);
  camPos.lerp(want, k);
  camLook.lerp(focus.clone().addScaledVector(swarm.heading, 90), k);
  camera.position.copy(camPos);
  camera.lookAt(camLook);
  lockRing.lookAt(camera.position);

  if (bannerTimer > 0 && (bannerTimer -= dt) <= 0) $("banner").classList.remove("show");
  composer.render();
  requestAnimationFrame(frame);
}

function step(dt: number, t: number, now: number) {
  const input: Input = face ? merge(face.read(now), keys.read()) : keys.read();
  $("face-warning").hidden = !face || input.faceVisible;

  // Steering: head turn sets the turn rate, nod sets the climb angle.
  yaw = (yaw + input.turn * 58 * dt + 360) % 360;
  pitch += (input.climb * 28 - pitch) * Math.min(1, dt * 3);
  // Keep clear of the ground and the water; stay inside the map.
  const ground = Math.max(0, terrain.heightAt(swarm.leader.x, swarm.leader.z));
  const ahead = swarm.leader.clone().addScaledVector(swarm.heading, 250);
  const groundAhead = Math.max(0, terrain.heightAt(ahead.x, ahead.z));
  const floor = Math.max(ground, groundAhead) + 35;
  if (swarm.leader.y < floor) pitch = Math.max(pitch, Math.min(35, (floor - swarm.leader.y) * 0.8));
  if (swarm.leader.y > 750) pitch = Math.min(pitch, -5);
  const b = terrain.bounds();
  const m = 1200;
  if (swarm.leader.x < b.minX + m || swarm.leader.x > b.maxX - m || swarm.leader.z < b.minZ + m || swarm.leader.z > b.maxZ - m) {
    // Compass bearing from the leader back to the map centre (the bridge): turn toward it.
    const back = ((Math.atan2(-swarm.leader.x, swarm.leader.z) * 180) / Math.PI + 360) % 360;
    yaw += THREE.MathUtils.clamp(((back - yaw + 540) % 360) - 180, -60 * dt, 60 * dt);
  }
  swarm.setHeading(yaw, pitch);
  const speed = input.boost ? 140 : 80;

  // Strikes: left wink → target on the left, right wink → on the right.
  const lockTarget = targets.pick(swarm.leader, swarm.heading, 950, 0);
  lockRing.visible = !!lockTarget;
  if (lockTarget) lockRing.position.copy(lockTarget.mesh.position);
  for (const [fired, side] of [[input.winkLeft, -1], [input.winkRight, 1]] as const) {
    if (!fired) continue;
    const orb = targets.pick(swarm.leader, swarm.heading, 950, side) ?? lockTarget;
    if (!orb) {
      banner("No target in range");
      continue;
    }
    const color = (orb.mesh.material as THREE.MeshBasicMaterial).color.clone();
    // One drone per wink, and it does not come back.
    const sent = swarm.strike(orb.mesh.position.clone(), () => {
      targets.destroy(orb);
      hitCount++;
      explosions.burst(orb.mesh.position, color, 260, 90);
      // Quick successive hits build a combo, capped so steady firing does not run away.
      combo = comboTimer > 0 ? Math.min(5, combo + 1) : 1;
      comboTimer = 4;
      score += 100 * combo;
      banner(combo > 1 ? `Hit! combo ×${combo}` : "Hit!");
    });
    if (sent) targets.claim(orb);
    else banner("No drones left to send");
  }

  const deaths = swarm.update(dt, t, speed, collides, (p) => Math.max(0, terrain.heightAt(p.x, p.z)));
  for (const p of deaths) explosions.burst(p, new THREE.Color("#ffb347"), 40, 35);

  // Bonus for flying under the main span.
  swarm.centroid(centroid);
  const under = bridge.underMainSpan(centroid);
  if (under && !wasUnder) {
    bridgeHitCount++;
    score += 500;
    banner("Under the Golden Gate! +500", 2.2);
  }
  wasUnder = under;

  if ((comboTimer -= dt) <= 0) combo = 0;
  timeLeft -= dt;
  hud();
  if (timeLeft <= 0 || swarm.wiped) end();
}

function end() {
  running = false;
  $("final-score").textContent = score.toLocaleString();
  $("end").hidden = false;
}

async function start(useFace: boolean) {
  $("intro").hidden = true;
  $("loading").hidden = false;
  try {
    if (useFace && !face) face = await FaceControl.create($<HTMLVideoElement>("cam"));
  } catch (e) {
    console.error(e);
    banner("Camera unavailable — keyboard mode", 3);
    face = null;
  }
  $("loading").hidden = true;
  $("cam-wrap").hidden = !face;
  score = 0;
  combo = 0;
  timeLeft = ROUND_SECONDS;
  yaw = 90;
  pitch = 0;
  swarm.setHeading(yaw, pitch);
  swarm.reset(START);
  running = true;
  if (face) setTimeout(() => face?.calibrate(), 600);
  banner(face ? "Turn your head to steer · wink to strike" : "Arrows to steer · Q / E to strike", 3);
}

async function init() {
  terrain = await Terrain.load(BASE);
  scene.add(terrain.mesh);
  $("attribution").textContent = terrain.attribution;
  swarm = new Swarm(START, 90);
  scene.add(swarm.mesh, swarm.lights);
  targets = new Targets(60, placeOrb);
  scene.add(targets.group);
  swarm.centroid(centroid);
  camPos.copy(centroid).add(new THREE.Vector3(-95, 30, 0));
  camLook.copy(centroid);
  hud();
  $("loading").hidden = true;
  $("intro").hidden = false;
  requestAnimationFrame(frame);
}

$("play-face").addEventListener("click", () => start(true));
$("play-keys").addEventListener("click", () => start(false));
$("again").addEventListener("click", () => {
  $("end").hidden = true;
  void start(!!face);
});
addEventListener("keydown", (e) => {
  if (e.code === "KeyC") face?.calibrate();
});

init().catch((e) => {
  console.error(e);
  $("loading").textContent = `Error: ${e.message ?? e}`;
});
