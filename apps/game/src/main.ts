import "./style.css";
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { Bridge } from "./bridge";
import { FaceControl, KeyboardControl, type Input } from "./control";
import { ALTITUDE, Course } from "./course";
import { chart, GOOD, grade, rank, type Note, type Song } from "./rhythm";
import { Swarm } from "./swarm";
import { Explosions } from "./targets";
import { Terrain } from "./terrain";
import { buildWorld } from "./world";

const BASE = import.meta.env.BASE_URL;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const START_DRONES = 24;
const COMBO_FOR_DRONE = 8; // every 8 hits in a row brings back one drone
const MISS_COST = 2; // drones lost when a missed orb hits the flock
const COUNTDOWN = 3; // seconds before the song starts
/** Camera + hand tracking reach us late; finger changes are dated this much earlier. */
const CAMERA_LAG = 0.12;
const COUNT_COLORS = ["", "#5ae0ff", "#ffd23f", "#ff5ad1", "#6dff8b", "#ff8a3d"];

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
// High threshold: only things drawn far brighter than white (orbs, explosions) glow.
composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 0.9, 0.5, 1.6));
composer.addPass(new OutputPass());

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

const world = buildWorld(scene, renderer);
const bridge = new Bridge();
scene.add(bridge.group);
const course = new Course();
scene.add(course.laneMarkers());
const explosions = new Explosions();
scene.add(explosions.points);

// --- Note orbs ------------------------------------------------------------

/** Big white digits with a dark outline, one texture per count. */
const LABELS: THREE.Texture[] = [];
for (let n = 0; n <= 5; n++) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  g.font = "bold 104px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.lineWidth = 12;
  g.strokeStyle = "rgba(0,0,0,0.75)";
  g.strokeText(String(n), 64, 70);
  g.fillStyle = "#fff";
  g.fillText(String(n), 64, 70);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  LABELS.push(t);
}

/** A glowing orb with its finger count on a sprite, reused from a pool. */
class NoteOrb {
  readonly group = new THREE.Group();
  private orb: THREE.Mesh;
  private label: THREE.Sprite;

  constructor() {
    this.orb = new THREE.Mesh(new THREE.IcosahedronGeometry(7, 2), new THREE.MeshBasicMaterial({ toneMapped: false }));
    this.label = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, transparent: true }));
    this.label.scale.setScalar(16);
    this.label.position.y = 15;
    this.group.add(this.orb, this.label);
    this.group.visible = false;
  }

  show(count: number, pos: THREE.Vector3, pulse: number) {
    (this.orb.material as THREE.MeshBasicMaterial).color.set(COUNT_COLORS[count]).multiplyScalar(4);
    (this.label.material as THREE.SpriteMaterial).map = LABELS[count];
    this.group.position.copy(pos);
    this.orb.scale.setScalar(1 + 0.18 * pulse);
    this.group.visible = true;
  }
}

const orbs = Array.from({ length: 48 }, () => new NoteOrb());
for (const o of orbs) scene.add(o.group);

// --- Game state -----------------------------------------------------------

let terrain: Terrain;
let swarm: Swarm;
let songs: Song[] = [];
let song: Song | null = null;
let notes: Note[] = [];
let speed = 120;
let audio: AudioContext | null = null;
let source: AudioBufferSourceNode | null = null;
let startAt = 0;
let state: "menu" | "playing" | "done" = "menu";
let face: FaceControl | null = null;
let useCamera = true;
const keys = new KeyboardControl();

let lanePos = 0;
let keyLane = 0;
let shown = 0;
let changedAt = -99;
let score = 0;
let combo = 0;
let maxCombo = 0;
let tally = { perfect: 0, good: 0, miss: 0 };
let shake = 0;
const buffers = new Map<string, AudioBuffer>();

function songTime() {
  return audio ? audio.currentTime - startAt : -COUNTDOWN;
}

function collides(p: THREE.Vector3) {
  return p.y < Math.max(0, terrain.heightAt(p.x, p.z)) + 0.5 || bridge.hits(p);
}

// --- HUD ------------------------------------------------------------------

function flash(id: string, text: string, cls = "") {
  const el = $(id);
  el.textContent = text;
  el.className = `pop ${cls}`;
  void el.offsetWidth; // restart the CSS animation
  el.classList.add("show");
}

function hud(t: number) {
  $("score").textContent = score.toLocaleString();
  $("drones").textContent = String(swarm.count);
  $("combo").textContent = combo >= 2 ? `${combo} combo` : "";
  $("progress").style.width = song ? `${Math.max(0, Math.min(100, (t / song.duration) * 100))}%` : "0";
  for (const [i, el] of [...document.querySelectorAll<HTMLElement>(".lane")].entries()) {
    el.classList.toggle("on", Math.round(lanePos) === i - 1);
  }
  $("shown").textContent = shown ? String(shown) : "–";
  $("shown").style.color = COUNT_COLORS[shown] || "";
  if (import.meta.env.DEV) {
    document.body.dataset.state = JSON.stringify({
      state, t: Math.round(t * 10) / 10, score, combo, drones: swarm.count, ...tally, lane: Math.round(lanePos * 10) / 10, shown,
      next: notes.filter((n) => !n.judged).slice(0, 3).map((n) => ({ t: Math.round(n.time * 100) / 100, lane: n.lane, count: n.count })),
    });
  }
}

// --- Loop -----------------------------------------------------------------

const camPos = new THREE.Vector3();
const camLook = new THREE.Vector3();
let last = performance.now();

function frame(now: number) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  world.update(now / 1000);
  const t = state === "playing" ? songTime() : -COUNTDOWN;
  if (state === "playing") step(dt, t, now);
  explosions.update(dt);

  // Camera behind the flock on the centre line, so the lanes stay put and the flock moves.
  const s = speed * t;
  const k = 1 - Math.exp(-dt * 4);
  camPos.lerp(course.at(s - 115, 0).add(new THREE.Vector3(0, 34, 0)), k);
  camLook.lerp(course.at(s + 140, 0).setY(ALTITUDE - 6), k);
  camera.position.copy(camPos);
  if (shake > 0) {
    camera.position.add(new THREE.Vector3().randomDirection().multiplyScalar(shake * 6));
    shake = Math.max(0, shake - dt * 2);
  }
  camera.lookAt(camLook);
  composer.render();
  requestAnimationFrame(frame);
}

function step(dt: number, t: number, now: number) {
  const input: Input = face ? face.read(now) : keys.read();
  const kb = face ? keys.read() : input;
  if (kb.laneStep) keyLane = Math.max(-1, Math.min(1, keyLane + kb.laneStep));
  const target = input.lane ?? keyLane;
  lanePos += (target - lanePos) * Math.min(1, dt * 8);
  $("face-warning").hidden = !face || input.faceVisible;
  const h = input.hands;
  $("fingers").textContent = h.left === null && h.right === null ? "" : `L ${h.left ?? "–"} · R ${h.right ?? "–"}`;

  // Finger count: remember when it last changed (dated back by the camera's delay).
  const nowShown = face ? input.fingers || kb.fingers : input.fingers;
  if (nowShown !== shown) {
    shown = nowShown;
    changedAt = t - (face ? CAMERA_LAG : 0);
  }

  // Fly the rail: the leader rides its lane; the flock follows as boids.
  const s = speed * t;
  const pos = course.at(s, lanePos);
  const f = course.frame(s);
  swarm.drive(pos, f.tangent.clone().multiplyScalar(speed));
  const deaths = swarm.update(dt, now / 1000, speed, collides, (p) => Math.max(0, terrain.heightAt(p.x, p.z)));
  for (const p of deaths) explosions.burst(p, new THREE.Color("#ffb347"), 30, 30);

  // Notes: show the ones ahead; judge the ones we are passing.
  const beat = song ? 60 / song.bpm : 0.5;
  const phase = ((((t - (song?.offset ?? 0)) % beat) + beat) % beat) / beat;
  const pulse = Math.max(0, 1 - phase / 0.35);
  let o = 0;
  for (const n of notes) {
    if (n.time > t + 4.5) break;
    if (n.judged) continue;
    if (o < orbs.length) orbs[o++].show(n.count, course.at(speed * n.time, n.lane).setY(ALTITUDE + 2), pulse);
    judgeNote(n, t);
  }
  for (; o < orbs.length; o++) orbs[o].group.visible = false;

  hud(t);
  if (song && t > song.duration) finish(false);
  else if (swarm.wiped || swarm.count === 0) finish(true);
}

function judgeNote(n: Note, t: number) {
  if (t < n.time - GOOD) return;
  const laneOk = Math.abs(lanePos - n.lane) < 0.5;
  const right = laneOk && shown === n.count;
  const at = course.at(speed * n.time, n.lane).setY(ALTITUDE + 2);
  if (right && (changedAt >= n.time - GOOD || t >= n.time)) {
    // Changed onto the right count within the window (Perfect/Good), or held it into the beat (Good).
    const j = changedAt >= n.time - GOOD ? grade(n, changedAt) : "good";
    n.judged = true;
    n.result = j;
    tally[j]++;
    combo++;
    maxCombo = Math.max(maxCombo, combo);
    const mult = 1 + Math.min(3, Math.floor(combo / 10) * 0.5);
    score += Math.round((j === "perfect" ? 300 : 120) * mult);
    explosions.burst(at, new THREE.Color(COUNT_COLORS[n.count]).multiplyScalar(3), 160, 70);
    flash("judgement", j === "perfect" ? "PERFECT" : "GOOD", j);
    if (combo % COMBO_FOR_DRONE === 0 && swarm.revive()) flash("bonus", "+1 drone");
  } else if (t > n.time + GOOD) {
    n.judged = true;
    n.result = "miss";
    tally.miss++;
    combo = 0;
    for (const p of swarm.lose(MISS_COST, at)) explosions.burst(p, new THREE.Color("#ff6a3d").multiplyScalar(2), 60, 45);
    shake = 0.6;
    flash("judgement", "MISS", "miss");
  }
}

// --- Flow -----------------------------------------------------------------

async function loadBuffer(s: Song) {
  if (!buffers.has(s.id)) {
    const data = await (await fetch(`${BASE}music/${s.file}`)).arrayBuffer();
    buffers.set(s.id, await audio!.decodeAudioData(data));
  }
  return buffers.get(s.id)!;
}

async function play(s: Song) {
  $("menu").hidden = true;
  $("results").hidden = true;
  $("loading").hidden = false;
  $("loading").textContent = "Loading…";
  audio ??= new AudioContext();
  await audio.resume();
  try {
    if (useCamera && !face) face = await FaceControl.create($<HTMLVideoElement>("cam"));
  } catch (e) {
    console.error(e);
    useCamera = false;
    face = null;
  }
  if (!useCamera) face = null;
  const buffer = await loadBuffer(s);
  $("loading").hidden = true;
  $("cam-wrap").hidden = !face;
  song = s;
  notes = chart(s);
  speed = course.length / s.duration;
  score = combo = maxCombo = 0;
  tally = { perfect: 0, good: 0, miss: 0 };
  lanePos = keyLane = 0;
  shown = 0;
  changedAt = -99;
  const start = course.at(-COUNTDOWN * speed, 0);
  swarm.drive(start, course.frame(-COUNTDOWN * speed).tangent.multiplyScalar(speed));
  swarm.reset(start, START_DRONES);
  $("credit").textContent = `♪ ${s.title} — ${s.creator} (${s.license})`;
  face?.calibrate();
  source?.stop();
  source = audio.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.destination);
  startAt = audio.currentTime + COUNTDOWN;
  source.start(startAt);
  state = "playing";
  for (let i = COUNTDOWN; i > 0; i--) setTimeout(() => flash("judgement", String(i), "count"), (COUNTDOWN - i) * 1000);
  setTimeout(() => flash("judgement", "GO!", "count"), COUNTDOWN * 1000);
}

function finish(failed: boolean) {
  state = "done";
  source?.stop();
  source = null;
  for (const o of orbs) o.group.visible = false;
  const total = notes.length || 1;
  const accuracy = (tally.perfect + tally.good * 0.6) / total;
  const key = `swarm-strike:best:${song!.id}`;
  let best = 0;
  try {
    best = Number(localStorage.getItem(key) ?? 0);
    if (!failed && score > best) localStorage.setItem(key, String(score));
  } catch {
    /* storage unavailable: no best score */
  }
  $("rank").textContent = failed ? "✕" : rank(accuracy);
  $("result-title").textContent = failed ? "Swarm lost" : "Run complete";
  $("final-score").textContent = score.toLocaleString();
  $("result-detail").textContent =
    `${tally.perfect} perfect · ${tally.good} good · ${tally.miss} miss · max combo ${maxCombo} · accuracy ${Math.round(accuracy * 100)}%`;
  $("result-best").textContent = !failed && score > best ? "New best!" : `Best ${best.toLocaleString()}`;
  $("results").hidden = false;
  renderMenu();
}

function renderMenu() {
  const list = $("songs");
  list.innerHTML = "";
  for (const s of songs) {
    let best = 0;
    try {
      best = Number(localStorage.getItem(`swarm-strike:best:${s.id}`) ?? 0);
    } catch {
      /* ignore */
    }
    const b = document.createElement("button");
    b.type = "button";
    b.className = "song";
    const len = `${Math.floor(s.duration / 60)}:${String(Math.round(s.duration % 60)).padStart(2, "0")}`;
    b.innerHTML = `<b>${s.title}</b><span>${s.level} · ${Math.round(s.bpm)} BPM · ${len}${best ? ` · best ${best.toLocaleString()}` : ""}</span>`;
    b.addEventListener("click", () => void play(s));
    list.append(b);
  }
}

async function init() {
  terrain = await Terrain.load(BASE);
  scene.add(terrain.mesh);
  songs = await (await fetch(`${BASE}music/songs.json`)).json();
  $("attribution").textContent = terrain.attribution;
  const start = course.at(-COUNTDOWN * 120, 0);
  swarm = new Swarm(start, 90, START_DRONES);
  swarm.drive(start, course.frame(0).tangent.multiplyScalar(120));
  scene.add(swarm.mesh, swarm.lights);
  camPos.copy(course.at(-COUNTDOWN * 120 - 115, 0)).add(new THREE.Vector3(0, 34, 0));
  camLook.copy(course.at(0, 0));
  renderMenu();
  $("loading").hidden = true;
  $("menu").hidden = false;
  requestAnimationFrame(frame);
}

for (const el of document.querySelectorAll<HTMLInputElement>("input[name=mode]")) {
  el.addEventListener("change", () => {
    useCamera = el.value === "camera";
  });
}
$("again").addEventListener("click", () => song && void play(song));
$("to-menu").addEventListener("click", () => {
  $("results").hidden = true;
  $("menu").hidden = false;
});
addEventListener("keydown", (e) => {
  if (e.code === "KeyC") face?.calibrate();
});

init().catch((e) => {
  console.error(e);
  $("loading").textContent = `Error: ${e.message ?? e}`;
});
