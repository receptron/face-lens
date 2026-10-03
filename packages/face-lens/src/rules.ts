import type { Category, Matrix } from "@mediapipe/tasks-vision";

export const DIRECTIONS = [
  "center", "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right",
] as const;
export type Direction = (typeof DIRECTIONS)[number];

export const EXPRESSIONS = [
  "wink-left", "wink-right", "eyes-closed", "smile", "mouth-open",
  "kiss", "cheek-puff", "brows-raised", "frown", "mouth-sideways",
] as const;
export type Expression = (typeof EXPRESSIONS)[number];

export interface Pose {
  /** Degrees. Positive = the face points right, as seen on screen (mirrored if `mirrored`). */
  yaw: number;
  /** Degrees. Positive = the face points up. */
  pitch: number;
}

/**
 * The canonical face looks down +z. The transformation matrix is column-major, so its third
 * column is where that forward vector ends up in camera space (x = image right, y = up).
 * Signs were checked against landmark geometry on FairFace (144/144 yaw, 67/68 pitch).
 */
export function headPose(m: Matrix, mirrored: boolean): Pose {
  const [fx, fy, fz] = [m.data[8], m.data[9], m.data[10]];
  const yaw = (Math.atan2(fx, fz) * 180) / Math.PI;
  const pitch = (Math.atan2(fy, Math.hypot(fx, fz)) * 180) / Math.PI;
  return { yaw: mirrored ? -yaw : yaw, pitch };
}

export interface DirectionThresholds {
  yawOn: number;
  yawOff: number;
  pitchOn: number;
  pitchOff: number;
}

export const DEFAULT_THRESHOLDS: DirectionThresholds = { yawOn: 16, yawOff: 11, pitchOn: 13, pitchOff: 9 };

/** 1 + 8 directions with hysteresis so the label does not flicker at the edges. */
export class DirectionClassifier {
  private h: -1 | 0 | 1 = 0;
  private v: -1 | 0 | 1 = 0;
  center: Pose = { yaw: 0, pitch: 0 };

  constructor(private t: DirectionThresholds = DEFAULT_THRESHOLDS) {}

  /** Pose relative to the calibrated center. */
  relative(p: Pose): Pose {
    return { yaw: p.yaw - this.center.yaw, pitch: p.pitch - this.center.pitch };
  }

  update(p: Pose): Direction {
    const { yaw, pitch } = this.relative(p);
    this.h = step(this.h, yaw, this.t.yawOn, this.t.yawOff);
    this.v = step(this.v, pitch, this.t.pitchOn, this.t.pitchOff);
    const vert = this.v > 0 ? "up" : this.v < 0 ? "down" : "";
    const horiz = this.h > 0 ? "right" : this.h < 0 ? "left" : "";
    return ((vert && horiz ? `${vert}-${horiz}` : vert || horiz) || "center") as Direction;
  }

  reset() {
    this.h = 0;
    this.v = 0;
  }
}

function step(state: -1 | 0 | 1, x: number, on: number, off: number): -1 | 0 | 1 {
  if (state === 0) return x > on ? 1 : x < -on ? -1 : 0;
  return Math.abs(x) < off || Math.sign(x) !== state ? 0 : state;
}

/**
 * Expression scores in 0..1 from MediaPipe's 52 blendshapes. Blendshapes are named from the
 * subject's point of view (eyeBlinkLeft = the person's own left eye; verified on FairFace), so
 * "wink-left" is the user's own left eye, which is also the left one in a mirrored view.
 */
export function expressionScores(categories: Category[]): Record<Expression, number> {
  const b: Record<string, number> = {};
  for (const c of categories) b[c.categoryName] = c.score;
  const g = (k: string) => b[k] ?? 0;
  const blinkL = g("eyeBlinkLeft");
  const blinkR = g("eyeBlinkRight");
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
