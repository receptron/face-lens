import "./style.css";
import { DIRECTIONS, EXPRESSIONS, FaceLens, drawCrop, type Direction, type FaceLensResult } from "@receptron/face-lens";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const statusEl = $("status");

// Everything is served next to the page (see scripts/assets.mjs); VITE_MODEL_URL can point the
// attribute model elsewhere, e.g. the Hugging Face repo.
const local = (p: string) => new URL(p, location.href).href;
const ASSETS = {
  modelUrl: import.meta.env.VITE_MODEL_URL ?? local("./models/"),
  mediapipeWasm: local("./mediapipe/wasm"),
  faceLandmarker: local("./models/face_landmarker.task"),
  ortWasm: { wasm: local("./ort/ort-wasm-simd-threaded.asyncify.wasm") },
};

const TITLES: Record<string, string> = {
  gender: "Gender",
  age: "Age group",
  emotion: "Emotion",
  hair: "Hair color",
  eyes: "Eye color",
};
const GLYPH: Record<Direction, string> = {
  "up-left": "↖", up: "↑", "up-right": "↗",
  left: "←", center: "●", right: "→",
  "down-left": "↙", down: "↓", "down-right": "↘",
};
const GRID: Direction[] = ["up-left", "up", "up-right", "left", "center", "right", "down-left", "down", "down-right"];
console.assert(GRID.length === DIRECTIONS.length);

// --- Static UI -------------------------------------------------------------

const dirCells = new Map<Direction, HTMLElement>();
for (const d of GRID) {
  const cell = document.createElement("div");
  cell.className = "cell";
  cell.textContent = GLYPH[d];
  cell.title = d;
  $("direction").append(cell);
  dirCells.set(d, cell);
}

const chips = new Map<string, HTMLElement>();
for (const e of EXPRESSIONS) {
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

const bars = new Map<string, HTMLElement>();
function buildAttributeCards(heads: string[]) {
  for (const key of heads) {
    if (key === "tongue") continue;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h2>${TITLES[key] ?? key}</h2><div class="bars"></div>`;
    $("learned").append(card);
    bars.set(key, card.querySelector(".bars")!);
  }
}

function renderAttributes(r: FaceLensResult | null) {
  for (const [key, el] of bars) {
    const a = r?.attributes?.[key];
    if (!a) {
      el.innerHTML = `<p class="muted">${r ? "…" : "No face"}</p>`;
      continue;
    }
    const top = Object.entries(a.probs).sort((x, y) => y[1] - x[1]).slice(0, 3);
    el.innerHTML = top
      .map(
        ([name, p], i) =>
          `<div class="bar${i === 0 ? " top" : ""}"><span class="name">${name}</span>` +
          `<span class="track"><span class="fill" style="width:${(p * 100).toFixed(0)}%"></span></span>` +
          `<span class="pct">${(p * 100).toFixed(0)}%</span></div>`,
      )
      .join("");
  }
  const tongue = r?.attributes?.tongue?.probs.yes ?? 0;
  tongueChip.classList.toggle("on", tongue > 0.5);
  tongueChip.style.setProperty("--level", tongue.toFixed(2));
}

function renderRules(r: FaceLensResult | null) {
  for (const [d, cell] of dirCells) cell.classList.toggle("on", d === r?.direction);
  $("pose").textContent = r ? `yaw ${r.pose.yaw.toFixed(0)}° · pitch ${r.pose.pitch.toFixed(0)}°` : "No face";
  for (const [name, chip] of chips) {
    const s = r?.expressions[name as keyof FaceLensResult["expressions"]] ?? 0;
    chip.classList.toggle("on", s > 0.5);
    chip.style.setProperty("--level", s.toFixed(2));
  }
}

// --- Main loop -------------------------------------------------------------

let lens: FaceLens | null = null;
let last: FaceLensResult | null = null;

async function start() {
  $("start-button").setAttribute("disabled", "");
  statusEl.textContent = "Loading models…";
  const [created, stream] = await Promise.all([
    FaceLens.create(ASSETS),
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    }),
  ]);
  lens = created;
  buildAttributeCards(Object.keys(lens.heads));
  video.srcObject = stream;
  await video.play();
  $("start").hidden = true;

  let frames = 0;
  let fpsStart = performance.now();
  const tick = () => {
    const now = performance.now();
    if (video.readyState >= 2) {
      last = lens!.detect(video, now);
      renderRules(last);
      renderAttributes(last);
      drawOverlay();
      frames++;
      if (now - fpsStart > 1000) {
        const fps = (frames * 1000) / (now - fpsStart);
        frames = 0;
        fpsStart = now;
        statusEl.textContent = `${fps.toFixed(0)} fps · tracker ${lens!.trackerDelegate} · attributes ${lens!.backend ?? "off"}`;
      }
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function drawOverlay() {
  const rect = video.getBoundingClientRect();
  const dpr = devicePixelRatio;
  if (overlay.width !== Math.round(rect.width * dpr)) {
    overlay.width = Math.round(rect.width * dpr);
    overlay.height = Math.round(rect.height * dpr);
  }
  const ctx = overlay.getContext("2d")!;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!last) return;
  // object-fit: cover — map video pixels to the displayed (mirrored) element.
  const s = Math.max(overlay.width / video.videoWidth, overlay.height / video.videoHeight);
  const ox = (overlay.width - video.videoWidth * s) / 2;
  const oy = (overlay.height - video.videoHeight * s) / 2;
  const size = last.box.size * s;
  const x = overlay.width - (ox + last.box.x * s) - size;
  const y = oy + last.box.y * s;
  ctx.strokeStyle = "rgba(120, 220, 255, 0.9)";
  ctx.lineWidth = 2 * dpr;
  ctx.strokeRect(x, y, size, size);
  // Arrow from the crop center along the head direction.
  const cx = x + size / 2;
  const cy = y + size / 2;
  const len = size * 0.5;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.sin((last.pose.yaw * Math.PI) / 180) * len, cy - Math.sin((last.pose.pitch * Math.PI) / 180) * len);
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
$("calibrate").addEventListener("click", () => lens?.calibrate());

// --- Capture (dev server only) --------------------------------------------

const EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"];
const CAPTURE_TAGS = ["free", "tongue=yes", "tongue=no", ...EMOTIONS.map((c) => `emotion=${c}`)];
const SAVE_SIZE = 384; // labels.json crop.saveSize
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
  if (!last) return;
  drawCrop(video, last.box, captureCanvas, SAVE_SIZE);
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
