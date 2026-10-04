import { FaceLens } from "@receptron/face-lens";

/** What the player is doing right now, from the face or the keyboard. */
export interface Input {
  /** -1 (left) .. 1 (right): turn rate. */
  turn: number;
  /** -1 (down) .. 1 (up): climb. */
  climb: number;
  /** Rising edges this frame. */
  winkLeft: boolean;
  winkRight: boolean;
  boost: boolean;
  /** Rising edge of a smile (not bound to an action yet). */
  smile: boolean;
  tongue: boolean;
  faceVisible: boolean;
}

const DEAD = 6; // degrees of head turn ignored around centre

/** Head turn → steering, nod → climb, wink → strike, open mouth → boost. */
export class FaceControl {
  private prev = { winkL: false, winkR: false, smile: false };
  private lastWink = 0;

  private constructor(
    readonly lens: FaceLens,
    readonly video: HTMLVideoElement,
  ) {}

  static async create(video: HTMLVideoElement): Promise<FaceControl> {
    const [lens, stream] = await Promise.all([
      // Direction and expressions only: no attribute model, nothing heavy to download.
      FaceLens.create({ attributes: false }),
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false }),
    ]);
    video.srcObject = stream;
    await video.play();
    return new FaceControl(lens, video);
  }

  read(now: number): Input {
    const r = this.lens.detect(this.video, now);
    const f = r.face;
    if (!f) {
      this.prev = { winkL: false, winkR: false, smile: false };
      return { turn: 0, climb: 0, winkLeft: false, winkRight: false, boost: false, smile: false, tongue: false, faceVisible: false };
    }
    const shape = (deg: number, full: number) => {
      const a = Math.abs(deg);
      return a < DEAD ? 0 : Math.sign(deg) * Math.min(1, (a - DEAD) / (full - DEAD));
    };
    const winkL = f.expressions["wink-left"] > 0.6;
    const winkR = f.expressions["wink-right"] > 0.6;
    const smile = f.expressions.smile > 0.7;
    // A blink is not a wink: require one eye open; debounce strikes.
    const fire = (cur: boolean, was: boolean) => cur && !was && now - this.lastWink > 350;
    const out: Input = {
      turn: shape(f.pose.yaw, 28),
      climb: shape(f.pose.pitch, 22),
      winkLeft: fire(winkL, this.prev.winkL),
      winkRight: fire(winkR, this.prev.winkR),
      boost: f.expressions["mouth-open"] > 0.5,
      smile: smile && !this.prev.smile,
      tongue: false,
      faceVisible: true,
    };
    if (out.winkLeft || out.winkRight) this.lastWink = now;
    this.prev = { winkL, winkR, smile };
    return out;
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
      winkLeft: p("KeyQ"),
      winkRight: p("KeyE", "Space"),
      boost: k("ShiftLeft", "ShiftRight"),
      smile: p("KeyF"),
      tongue: false,
      faceVisible: true,
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
    winkLeft: a.winkLeft || b.winkLeft,
    winkRight: a.winkRight || b.winkRight,
    boost: a.boost || b.boost,
    smile: a.smile || b.smile,
    tongue: a.tongue || b.tongue,
    faceVisible: a.faceVisible || b.faceVisible,
  };
}
