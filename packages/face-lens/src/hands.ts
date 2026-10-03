import type { Landmark, NormalizedLandmark } from "@mediapipe/tasks-vision";

export const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;
export type Finger = (typeof FINGERS)[number];

export interface Hand {
  /** Fingers counted as raised, 0..5. */
  count: number;
  /** Which fingers are raised. */
  up: Record<Finger, boolean>;
  /** 21 landmarks, normalized to the frame. */
  landmarks: NormalizedLandmark[];
  /** MediaPipe's confidence that this is the hand it says it is. */
  handednessScore: number;
}

export interface Hands {
  /** The user's own left hand. */
  left: Hand | null;
  /** The user's own right hand. */
  right: Hand | null;
  /** Raised fingers over both hands, 0..10. */
  total: number;
}

// Same rule and thresholds as teacher/fingers.py, tuned on HaGRID (83% exact on held-out hands).
const STRAIGHT_DEG = 70;
const THUMB_DEG = 100;
const THUMB_REACH = 1.1;
const THUMB_REF = 13;
const PINCH = 0.75;
const JOINTS: Record<Exclude<Finger, "thumb">, [number, number, number, number]> = {
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

type V = [number, number, number];
const v = (p: Landmark): V => [p.x, p.y, p.z];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dist = (a: V, b: V) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
function angle(u: V, w: V) {
  const nu = Math.hypot(...u) || 1e-9;
  const nw = Math.hypot(...w) || 1e-9;
  const c = Math.max(-1, Math.min(1, (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (nu * nw)));
  return (Math.acos(c) * 180) / Math.PI;
}
const bend = (p: V[], a: number, b: number, c: number, d: number) =>
  angle(sub(p[b], p[a]), sub(p[c], p[b])) + angle(sub(p[c], p[b]), sub(p[d], p[c]));

/** Raised fingers from MediaPipe's 21 world landmarks (meters, hand-centred: rotation-free). */
export function fingersUp(world: Landmark[]): Record<Finger, boolean> {
  const p = world.map(v);
  const palm = dist(p[5], p[17]);
  const up = { thumb: false, index: false, middle: false, ring: false, pinky: false };
  for (const [name, [mcp, pip, dip, tip]] of Object.entries(JOINTS) as [keyof typeof JOINTS, number[]][]) {
    up[name] = bend(p, mcp, pip, dip, tip) < STRAIGHT_DEG && dist(p[tip], p[0]) > dist(p[pip], p[0]);
  }
  up.thumb = bend(p, 1, 2, 3, 4) < THUMB_DEG && dist(p[4], p[THUMB_REF]) > THUMB_REACH * palm;
  // Thumb and index tips touching ("OK", pinch): neither counts as raised.
  if (dist(p[4], p[8]) < PINCH * palm) {
    up.thumb = false;
    up.index = false;
  }
  return up;
}
