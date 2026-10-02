import "./style.css";
import labels from "../labels.json";
import {
  createLandmarker,
  cropBox,
  DirectionClassifier,
  drawCrop,
  expressionScores,
  headPose,
  type Box,
  type Direction,
  type Pose,
  type Scores,
} from "./face";
import { Student, type Probs } from "./student";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const statusEl = $("status");

const MIRRORED = true;
const DIRECTION_GLYPH: Record<Direction, string> = {
  "up-left": "↖", up: "↑", "up-right": "↗",
  left: "←", center: "●", right: "→",
  "down-left": "↙", down: "↓", "down-right": "↘",
};
const GRID: Direction[] = ["up-left", "up", "up-right", "left", "center", "right", "down-left", "down", "down-right"];

// --- Static UI -------------------------------------------------------------

const dirCells = new Map<Direction, HTMLElement>();
for (const d of GRID) {
  const cell = document.createElement("div");
  cell.className = "cell";
  cell.textContent = DIRECTION_GLYPH[d];
  cell.title = d;
  $("direction").append(cell);
  dirCells.set(d, cell);
}

const chips = new Map<string, HTMLElement>();
for (const e of labels.rules.expressions.classes) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = e;
  $("expressions").append(chip);
  chips.set(e, chip);
}
const tongueChip = document.createElement("span");
tongueChip.className = "chip learned-chip";
tongueChip.textContent = "tongue-out";
tongueChip.title = "From the trained model";
$("expressions").append(tongueChip);

const LEARNED = Object.entries(labels.learned).filter(([key]) => key !== "tongue");
const bars = new Map<string, HTMLElement>();
for (const [key, spec] of LEARNED) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h2>${spec.title}</h2><div class="bars"></div>`;
  $("learned").append(card);
  bars.set(key, card.querySelector(".bars")!);
}

function renderBars(probs: Probs | null) {
  for (const [key, spec] of LEARNED) {
    const el = bars.get(key)!;
    const p = probs?.[key];
    if (!p) {
      el.innerHTML = `<p class="muted">${student ? "No face" : "Model not trained yet"}</p>`;
      continue;
    }
    const top = spec.classes.map((name, i) => ({ name, p: p[i] })).sort((a, b) => b.p - a.p).slice(0, 3);
    el.innerHTML = top
      .map(
        ({ name, p }, i) =>
          `<div class="bar${i === 0 ? " top" : ""}"><span class="name">${name}</span>` +
          `<span class="track"><span class="fill" style="width:${(p * 100).toFixed(0)}%"></span></span>` +
          `<span class="pct">${(p * 100).toFixed(0)}%</span></div>`,
      )
      .join("");
  }
}

// --- Smoothing -------------------------------------------------------------

function emaRecord<T extends Record<string, number>>(prev: T | null, next: T, a: number): T {
  if (!prev) return next;
  const out = { ...next };
  for (const k in next) (out as Record<string, number>)[k] = prev[k] + a * (next[k] - prev[k]);
  return out;
}

function emaProbs(prev: Probs | null, next: Probs, a: number): Probs {
  if (!prev) return next;
  const out: Probs = {};
  for (const k in next) out[k] = next[k].map((x, i) => prev[k][i] + a * (x - prev[k][i]));
  return out;
}

// --- Main loop -------------------------------------------------------------

let student: Student | null = null;
let smoothScores: Scores | null = null;
let smoothProbs: Probs | null = null;
let lastPose: Pose | null = null;
let lastBox: Box | null = null;
const direction = new DirectionClassifier();

async function start() {
  $("start-button").setAttribute("disabled", "");
  statusEl.textContent = "Loading face tracker…";
  const [{ landmarker, delegate }, stream] = await Promise.all([
    createLandmarker(),
    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false }),
  ]);
  video.srcObject = stream;
  await video.play();
  $("start").hidden = true;

  statusEl.textContent = "Loading attribute model…";
  student = await Student.load().catch((e) => {
    console.error(e);
    return null;
  });
  renderBars(null);

  let frames = 0;
  let fps = 0;
  let fpsStart = performance.now();
  let lastTs = -1;

  const tick = () => {
    const now = performance.now();
    if (video.readyState >= 2 && now !== lastTs) {
      lastTs = now;
      const result = landmarker.detectForVideo(video, now);
      const face = result.faceLandmarks[0];
      if (face) {
        lastBox = cropBox(face, video.videoWidth, video.videoHeight);
        lastPose = headPose(result.facialTransformationMatrixes[0], MIRRORED);
        smoothScores = emaRecord(smoothScores, expressionScores(result.faceBlendshapes[0].categories), 0.5);
        renderRules(direction.update(lastPose), lastPose, smoothScores);
        student?.run(video, lastBox).then((p) => {
          if (!p || !lastBox) return;
          smoothProbs = emaProbs(smoothProbs, p, 0.35);
          renderBars(smoothProbs);
          renderTongue(smoothProbs.tongue?.[1] ?? 0);
        });
      } else {
        lastBox = null;
        lastPose = null;
        smoothScores = null;
        smoothProbs = null;
        renderRules(null, null, null);
        renderBars(null);
        renderTongue(0);
      }
      drawOverlay();
      frames++;
      if (now - fpsStart > 1000) {
        fps = (frames * 1000) / (now - fpsStart);
        frames = 0;
        fpsStart = now;
        const model = student ? `attributes ${student.backend}` : "attributes: not trained";
        statusEl.textContent = `${fps.toFixed(0)} fps · tracker ${delegate} · ${model}`;
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderRules(dir: Direction | null, pose: Pose | null, scores: Scores | null) {
  for (const [d, cell] of dirCells) cell.classList.toggle("on", d === dir);
  $("pose").textContent = pose
    ? `yaw ${(pose.yaw - direction.center.yaw).toFixed(0)}° · pitch ${(pose.pitch - direction.center.pitch).toFixed(0)}°`
    : "No face";
  for (const [name, chip] of chips) {
    const s = scores?.[name as keyof Scores] ?? 0;
    chip.classList.toggle("on", s > 0.5);
    chip.style.setProperty("--level", s.toFixed(2));
  }
}

function renderTongue(p: number) {
  tongueChip.classList.toggle("on", p > 0.5);
  tongueChip.style.setProperty("--level", p.toFixed(2));
}

function drawOverlay() {
  const rect = video.getBoundingClientRect();
  const dpr = devicePixelRatio;
  if (overlay.width !== Math.round(rect.width * dpr)) {
    overlay.width = Math.round(rect.width * dpr);
    overlay.height = Math.round(rect.height * dpr);
  }
  const ctx = overlay.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!lastBox || !lastPose) return;
  // object-fit: cover — map video pixels to the displayed element.
  const s = Math.max(overlay.width / video.videoWidth, overlay.height / video.videoHeight);
  const ox = (overlay.width - video.videoWidth * s) / 2;
  const oy = (overlay.height - video.videoHeight * s) / 2;
  const x = ox + lastBox.x * s;
  const sx = MIRRORED ? overlay.width - x - lastBox.size * s : x;
  const y = oy + lastBox.y * s;
  const size = lastBox.size * s;
  ctx.strokeStyle = "rgba(120, 220, 255, 0.9)";
  ctx.lineWidth = 2 * dpr;
  ctx.strokeRect(sx, y, size, size);
  // Arrow from the crop center along the head direction.
  const cx = sx + size / 2;
  const cy = y + size / 2;
  const len = size * 0.5;
  const yaw = ((lastPose.yaw - direction.center.yaw) * Math.PI) / 180;
  const pitch = ((lastPose.pitch - direction.center.pitch) * Math.PI) / 180;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.sin(yaw) * len, cy - Math.sin(pitch) * len);
  ctx.strokeStyle = "rgba(255, 200, 80, 0.95)";
  ctx.lineWidth = 4 * dpr;
  ctx.stroke();
}

$("start-button").addEventListener("click", () =>
  start().catch((e) => {
    console.error(e);
    statusEl.textContent = `Error: ${e.message ?? e}`;
    $("start-button").removeAttribute("disabled");
  }),
);

$("calibrate").addEventListener("click", () => {
  if (lastPose) direction.center = { ...lastPose };
});

// --- Capture (dev server only) --------------------------------------------

const CAPTURE_TAGS = [
  "free",
  "tongue=yes",
  "tongue=no",
  ...labels.learned.emotion.classes.map((c) => `emotion=${c}`),
];
const tagSelect = $<HTMLSelectElement>("capture-tag");
for (const t of CAPTURE_TAGS) tagSelect.append(new Option(t === "free" ? "free (anything — teacher labels it)" : t, t));

$("toggle-capture").addEventListener("click", () => {
  $("capture").hidden = !$("capture").hidden;
});
if (!import.meta.env.DEV) $("toggle-capture").hidden = true;

let saved = 0;
let captureTimer: number | undefined;
const captureCanvas = new OffscreenCanvas(1, 1);

async function captureOne() {
  if (!lastBox) return;
  drawCrop(video, lastBox, captureCanvas, labels.crop.saveSize);
  const blob = await captureCanvas.convertToBlob({ type: "image/jpeg", quality: 0.92 });
  const image = await new Promise<string>((resolve) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.readAsDataURL(blob);
  });
  const res = await fetch("/api/capture", { method: "POST", body: JSON.stringify({ tag: tagSelect.value, image }) });
  if (res.ok) $("capture-count").textContent = `${++saved} saved`;
}

const hold = $("capture-hold");
const stopCapture = () => {
  clearInterval(captureTimer);
  hold.classList.remove("recording");
};
hold.addEventListener("pointerdown", () => {
  hold.classList.add("recording");
  captureTimer = window.setInterval(captureOne, 160);
});
hold.addEventListener("pointerup", stopCapture);
hold.addEventListener("pointerleave", stopCapture);
