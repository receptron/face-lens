import { FaceLens } from "@receptron/face-lens";

/** What the player is doing right now, from the face, the hands or the keyboard. */
export interface Input {
  /** -1 (left) .. 1 (right): turn rate. */
  turn: number;
  /** -1 (down) .. 1 (up): climb. */
  climb: number;
  /** Rising edges this frame: strike at a target on that side. */
  strikeLeft: boolean;
  strikeRight: boolean;
  boost: boolean;
  faceVisible: boolean;
  /** Raised fingers per hand, for the HUD (null = hand not seen). */
  fingers: { left: number | null; right: number | null };
}

const DEAD = 6; // degrees of head turn ignored around centre
/** One finger must be up this long before it counts, so a passing hand shape does not fire. */
const HOLD_MS = 120;

type Side = "left" | "right";

/**
 * Head turn → steering, nod → climb, open mouth → boost, and one raised finger → strike:
 * the left hand at a target on the left, the right hand on the right. Each showing of one
 * finger fires once; drop the hand (or change the count) to fire again.
 */
export class FaceControl {
  private oneSince: Record<Side, number> = { left: 0, right: 0 };
  private armed: Record<Side, boolean> = { left: true, right: true };

  private constructor(
    readonly lens: FaceLens,
    readonly video: HTMLVideoElement,
  ) {}

  static async create(video: HTMLVideoElement): Promise<FaceControl> {
    const [lens, stream] = await Promise.all([
      // Direction, expressions and fingers; no attribute model, so nothing heavy to download.
      FaceLens.create({ attributes: false, hands: true }),
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false }),
    ]);
    video.srcObject = stream;
    await video.play();
    return new FaceControl(lens, video);
  }

  read(now: number): Input {
    const r = this.lens.detect(this.video, now);
    const f = r.face;
    const fingers = { left: r.hands?.left?.count ?? null, right: r.hands?.right?.count ?? null };
    const strike = (side: Side) => {
      if (fingers[side] !== 1) {
        this.oneSince[side] = 0;
        this.armed[side] = true;
        return false;
      }
      if (!this.oneSince[side]) this.oneSince[side] = now;
      if (this.armed[side] && now - this.oneSince[side] >= HOLD_MS) {
        this.armed[side] = false;
        return true;
      }
      return false;
    };
    const strikeLeft = strike("left");
    const strikeRight = strike("right");
    if (!f) {
      return { turn: 0, climb: 0, strikeLeft, strikeRight, boost: false, faceVisible: false, fingers };
    }
    const shape = (deg: number, full: number) => {
      const a = Math.abs(deg);
      return a < DEAD ? 0 : Math.sign(deg) * Math.min(1, (a - DEAD) / (full - DEAD));
    };
    return {
      turn: shape(f.pose.yaw, 28),
      climb: shape(f.pose.pitch, 22),
      strikeLeft,
      strikeRight,
      boost: f.expressions["mouth-open"] > 0.5,
      faceVisible: true,
      fingers,
    };
  }

  calibrate() {
    this.lens.calibrate();
  }
}

/** Arrow keys / WASD steer, Q / E strike left / right, Shift boosts. */
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
    const k = (...codes: string[]) => codes.some((c) => this.down.has(c));
    const p = (...codes: string[]) => codes.some((c) => this.pressed.has(c));
    const out: Input = {
      turn: (k("ArrowRight", "KeyD") ? 1 : 0) - (k("ArrowLeft", "KeyA") ? 1 : 0),
      climb: (k("ArrowUp", "KeyW") ? 1 : 0) - (k("ArrowDown", "KeyS") ? 1 : 0),
      strikeLeft: p("KeyQ"),
      strikeRight: p("KeyE", "Space"),
      boost: k("ShiftLeft", "ShiftRight"),
      faceVisible: true,
      fingers: { left: null, right: null },
    };
    this.pressed.clear();
    return out;
  }
}

/** Face and keyboard together: whichever is active wins for steering, any can trigger actions. */
export function merge(a: Input, b: Input): Input {
  const pick = (x: number, y: number) => (Math.abs(x) > Math.abs(y) ? x : y);
  return {
    turn: pick(a.turn, b.turn),
    climb: pick(a.climb, b.climb),
    strikeLeft: a.strikeLeft || b.strikeLeft,
    strikeRight: a.strikeRight || b.strikeRight,
    boost: a.boost || b.boost,
    faceVisible: a.faceVisible || b.faceVisible,
    fingers: a.fingers.left !== null || a.fingers.right !== null ? a.fingers : b.fingers,
  };
}
