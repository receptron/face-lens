import { FaceLens } from "@receptron/face-lens";

/** What the player is doing right now, from the camera or the keyboard. */
export interface Input {
  /** Lane from the head turn: -1 (left) .. 1 (right), continuous; null when no face is seen. */
  lane: number | null;
  /** Keyboard lane change this frame: -1, 0 or 1. */
  laneStep: -1 | 0 | 1;
  /** Fingers shown (both hands added up), 0 when no hand is seen. */
  fingers: number;
  /** Per hand, for the HUD (null = hand not seen). */
  hands: { left: number | null; right: number | null };
  faceVisible: boolean;
}

/** Degrees of head turn that reach the outer lane; a little dead zone keeps the centre steady. */
const FULL_TURN = 18;
const DEAD = 4;

/** Head turn → lane, raised fingers → the number to match. */
export class FaceControl {
  private constructor(
    readonly lens: FaceLens,
    readonly video: HTMLVideoElement,
  ) {}

  static async create(video: HTMLVideoElement): Promise<FaceControl> {
    const [lens, stream] = await Promise.all([
      // Direction and fingers only; no attribute model, so nothing heavy to download.
      FaceLens.create({ attributes: false, hands: true, handsEvery: 1 }),
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false }),
    ]);
    video.srcObject = stream;
    await video.play();
    return new FaceControl(lens, video);
  }

  read(now: number): Input {
    const r = this.lens.detect(this.video, now);
    const hands = { left: r.hands?.left?.count ?? null, right: r.hands?.right?.count ?? null };
    const fingers = (hands.left ?? 0) + (hands.right ?? 0);
    const f = r.face;
    if (!f) return { lane: null, laneStep: 0, fingers, hands, faceVisible: false };
    const yaw = f.pose.yaw;
    const a = Math.max(0, Math.abs(yaw) - DEAD) / (FULL_TURN - DEAD);
    return { lane: Math.sign(yaw) * Math.min(1, a), laneStep: 0, fingers, hands, faceVisible: true };
  }

  calibrate() {
    this.lens.calibrate();
  }
}

/** ← / → (or A / D) change lane; hold 1–5 for the finger count. */
export class KeyboardControl {
  private down = new Set<string>();
  private pressed = new Set<string>();

  constructor() {
    addEventListener("keydown", (e) => {
      if (!this.down.has(e.code)) this.pressed.add(e.code);
      this.down.add(e.code);
    });
    addEventListener("keyup", (e) => this.down.delete(e.code));
  }

  read(): Input {
    const p = (...codes: string[]) => codes.some((c) => this.pressed.has(c));
    let fingers = 0;
    for (let n = 1; n <= 5; n++) if (this.down.has(`Digit${n}`) || this.down.has(`Numpad${n}`)) fingers = n;
    const laneStep = (p("ArrowRight", "KeyD") ? 1 : 0) - (p("ArrowLeft", "KeyA") ? 1 : 0);
    this.pressed.clear();
    return {
      lane: null,
      laneStep: laneStep as -1 | 0 | 1,
      fingers,
      hands: { left: null, right: null },
      faceVisible: false,
    };
  }
}
