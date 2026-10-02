import {
  FaceLandmarker,
  FilesetResolver,
  type Category,
  type Matrix,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import labels from "../labels.json";

const BASE = import.meta.env.BASE_URL;

export type Direction = (typeof labels.rules.direction.classes)[number];
export type Expression = (typeof labels.rules.expressions.classes)[number];

export async function createLandmarker(): Promise<{ landmarker: FaceLandmarker; delegate: "GPU" | "CPU" }> {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}mediapipe/wasm`);
  for (const delegate of ["GPU", "CPU"] as const) {
    try {
      const landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
      return { landmarker, delegate };
    } catch (e) {
      if (delegate === "CPU") throw e;
      console.warn("MediaPipe GPU delegate failed, falling back to CPU", e);
    }
  }
  throw new Error("unreachable");
}

// --- Head pose -------------------------------------------------------------

export interface Pose {
  /** Degrees. Positive = the face points toward the right of the (mirrored) screen. */
  yaw: number;
  /** Degrees. Positive = the face points up. */
  pitch: number;
}

/**
 * The canonical face looks down +z. The transformation matrix is column-major,
 * so its third column is where that forward vector ends up in camera space
 * (x = image right, y = up). The video is shown mirrored, so yaw flips sign to
 * match what the user sees.
 */
export function headPose(m: Matrix, mirrored: boolean): Pose {
  const [fx, fy, fz] = [m.data[8], m.data[9], m.data[10]];
  const yaw = (Math.atan2(fx, fz) * 180) / Math.PI;
  const pitch = (Math.atan2(fy, Math.hypot(fx, fz)) * 180) / Math.PI;
  return { yaw: mirrored ? -yaw : yaw, pitch };
}

const YAW_ON = 16;
const YAW_OFF = 11;
const PITCH_ON = 13;
const PITCH_OFF = 9;

/** 1 + 8 directions with hysteresis so the label does not flicker at the edges. */
export class DirectionClassifier {
  private h: -1 | 0 | 1 = 0;
  private v: -1 | 0 | 1 = 0;
  center: Pose = { yaw: 0, pitch: 0 };

  update(p: Pose): Direction {
    const yaw = p.yaw - this.center.yaw;
    const pitch = p.pitch - this.center.pitch;
    this.h = step(this.h, yaw, YAW_ON, YAW_OFF);
    this.v = step(this.v, pitch, PITCH_ON, PITCH_OFF);
    const vert = this.v > 0 ? "up" : this.v < 0 ? "down" : "";
    const horiz = this.h > 0 ? "right" : this.h < 0 ? "left" : "";
    return ((vert && horiz ? `${vert}-${horiz}` : vert || horiz) || "center") as Direction;
  }
}

function step(state: -1 | 0 | 1, x: number, on: number, off: number): -1 | 0 | 1 {
  if (state === 0) return x > on ? 1 : x < -on ? -1 : 0;
  return Math.abs(x) < off || Math.sign(x) !== state ? 0 : state;
}

// --- Expressions from blendshapes ----------------------------------------

/**
 * MediaPipe names blendshapes ARKit-style, from the subject's point of view:
 * eyeBlinkLeft is the person's own left eye. Flip this if a test shows otherwise.
 */
const SUBJECT_LEFT_IS_LEFT = true;

export type Scores = Record<Expression, number>;

export function expressionScores(categories: Category[]): Scores {
  const b: Record<string, number> = {};
  for (const c of categories) b[c.categoryName] = c.score;
  const g = (k: string) => b[k] ?? 0;
  const blinkL = SUBJECT_LEFT_IS_LEFT ? g("eyeBlinkLeft") : g("eyeBlinkRight");
  const blinkR = SUBJECT_LEFT_IS_LEFT ? g("eyeBlinkRight") : g("eyeBlinkLeft");
  // A wink is one eye shut while the other stays open; smiling squints both,
  // so the score is the gap between the eyes rather than one eye alone.
  const wink = (shut: number, open: number) => clamp01((shut - open - 0.25) / 0.35);
  return {
    "wink-left": wink(blinkL, blinkR),
    "wink-right": wink(blinkR, blinkL),
    "eyes-closed": clamp01((Math.min(blinkL, blinkR) - 0.45) / 0.3),
    smile: clamp01(((g("mouthSmileLeft") + g("mouthSmileRight")) / 2 - 0.3) / 0.4),
    "mouth-open": clamp01((g("jawOpen") - 0.25) / 0.35),
    kiss: clamp01((g("mouthPucker") - 0.45) / 0.35),
    "cheek-puff": clamp01((g("cheekPuff") - 0.25) / 0.35),
    "brows-raised": clamp01(((g("browInnerUp") + g("browOuterUpLeft") + g("browOuterUpRight")) / 3 - 0.3) / 0.35),
    frown: clamp01(((g("browDownLeft") + g("browDownRight")) / 2 - 0.3) / 0.35),
    "mouth-sideways": clamp01((Math.max(g("mouthLeft"), g("mouthRight")) - 0.3) / 0.35),
  };
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// --- Face crop (shared spec with the teacher and training) ----------------

export interface Box {
  x: number;
  y: number;
  size: number;
}

/** Square crop in video pixels, per labels.json `crop`. */
export function cropBox(landmarks: NormalizedLandmark[], width: number, height: number): Box {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const w = (maxX - minX) * width;
  const h = (maxY - minY) * height;
  const size = Math.max(w, h) * labels.crop.scale;
  const cx = ((minX + maxX) / 2) * width;
  const cy = ((minY + maxY) / 2) * height + h * labels.crop.shiftY;
  return { x: cx - size / 2, y: cy - size / 2, size };
}

/** Draws the crop into `canvas` at `out`×`out`; area outside the frame is black. */
export function drawCrop(video: HTMLVideoElement, box: Box, canvas: OffscreenCanvas, out: number) {
  canvas.width = out;
  canvas.height = out;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, out, out);
  const k = out / box.size;
  ctx.drawImage(video, -box.x * k, -box.y * k, video.videoWidth * k, video.videoHeight * k);
  return ctx;
}
