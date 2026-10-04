import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import {
  AttributeModel,
  clothingBox,
  cropBox,
  sourceSize,
  type AttributeProbs,
  type Box,
  type FrameSource,
  type ModelMeta,
} from "./attributes.js";
import { clothingColors, type ClothingColor, type ColorName } from "./colors.js";
import { fingersUp, type Finger, type Hand, type Hands } from "./hands.js";
import {
  DEFAULT_THRESHOLDS,
  DirectionClassifier,
  expressionScores,
  headPose,
  type Direction,
  type DirectionThresholds,
  type Expression,
  type Pose,
} from "./rules.js";

export { DIRECTIONS, EXPRESSIONS } from "./rules.js";
export { FINGERS } from "./hands.js";
export { COLOR_NAMES, nameColor } from "./colors.js";
/** The exact crop the attribute model was trained on, e.g. for collecting training images. */
export { clothingBox, cropBox, drawCrop } from "./attributes.js";
export type {
  Box, ClothingColor, ColorName, Direction, DirectionThresholds, Expression, Finger, FrameSource, Hand, Hands,
  ModelMeta, Pose,
};

// Keep in sync with package.json dependencies.
const MEDIAPIPE_VERSION = "1.0.1";
const ORT_VERSION = "1.30.0";
const MP_MODELS = "https://storage.googleapis.com/mediapipe-models";

export const DEFAULTS = {
  modelUrl: "https://huggingface.co/snakajima/face-lens/resolve/v1/",
  mediapipeWasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`,
  faceLandmarker: `${MP_MODELS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
  handLandmarker: `${MP_MODELS}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  segmenter: `${MP_MODELS}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`,
  ortWasm: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
};

export interface FaceLensOptions {
  /**
   * Learned attributes to run: `true` for every head the model has, `false` for none (only
   * direction and expressions, and onnxruntime-web is never loaded), or a list of head names.
   * Default `true`.
   */
  attributes?: boolean | string[];
  /** Count raised fingers on each hand (MediaPipe Hand Landmarker, ~8 MB). Default `false`. */
  hands?: boolean;
  /**
   * Clothing: colors from MediaPipe selfie segmentation (~16 MB), plus style and pattern from a
   * small model (~5 MB) when the model URL has one. Default `false`.
   */
  clothing?: boolean;
  /** Run hand tracking every N frames (1 = every frame). Default 2. */
  handsEvery?: number;
  /** Run clothing segmentation every N frames. Default 8. */
  clothingEvery?: number;
  /** Base URL of `student.onnx` + `student.json`. Default: the Hugging Face repo, pinned. */
  modelUrl?: string;
  /** Directory of MediaPipe's wasm files. */
  mediapipeWasm?: string;
  /** URL of `face_landmarker.task`. */
  faceLandmarker?: string;
  /** URL of `hand_landmarker.task`. */
  handLandmarker?: string;
  /** URL of `selfie_multiclass_256x256.tflite`. */
  segmenter?: string;
  /** Directory (or explicit file map) for onnxruntime-web's wasm. */
  ortWasm?: string | { wasm: string; mjs?: string };
  /** "auto" tries WebGPU first, then WASM. */
  executionProvider?: "auto" | "webgpu" | "wasm";
  /** Report left/right as seen in a mirrored selfie view. Default `true`. */
  mirrored?: boolean;
  /** Exponential smoothing per frame, 0 = off, closer to 1 = smoother. */
  smoothing?: { expressions?: number; attributes?: number };
  thresholds?: Partial<DirectionThresholds>;
  /** Called whenever a new attribute estimate arrives (they lag a frame or two behind). */
  onAttributes?: (attributes: Attributes) => void;
}

export interface Attribute {
  /** Most likely class. */
  label: string;
  /** Its probability. */
  confidence: number;
  /** Every class with its probability. */
  probs: Record<string, number>;
}

export type Attributes = Record<string, Attribute>;

export interface Face {
  /** 478 landmarks, normalized to the frame. */
  landmarks: NormalizedLandmark[];
  /** The square crop the attribute model sees, in frame pixels. */
  box: Box;
  /** Head angles relative to the calibrated center. */
  pose: Pose;
  direction: Direction;
  /** 0..1 per expression; treat > 0.5 as "on". */
  expressions: Record<Expression, number>;
  /** Latest smoothed attribute estimate, or null until the first one arrives. */
  attributes: Attributes | null;
}

export interface Clothing {
  /** Up to three colors, largest share first; empty when too little clothing is visible. */
  colors: ClothingColor[];
  /** Garment type (t-shirt, hoodie, …), or null when the torso is not in frame or no model. */
  style: Attribute | null;
  /** solid, striped, checked-plaid or printed-logo; null like `style`. */
  pattern: Attribute | null;
}

export interface FaceLensResult {
  timestamp: number;
  /** null when no face is visible. */
  face: Face | null;
  /** null when `hands` is off. */
  hands: Hands | null;
  /** null when `clothing` is off. */
  clothing: Clothing | null;
}

export class FaceLens {
  private direction: DirectionClassifier;
  private smoothExpr: Record<Expression, number> | null = null;
  private smoothAttr: AttributeProbs | null = null;
  private lastPose: Pose | null = null;
  private lastTs = -1;
  private frame = 0;
  private faceGeneration = 0;
  private lastHands: Hands | null = null;
  private lastClothing: Clothing | null = null;
  private smoothCloth: AttributeProbs | null = null;
  private pixelCanvas = new OffscreenCanvas(1, 1);

  private constructor(
    private landmarker: FaceLandmarker,
    private handLandmarker: HandLandmarker | null,
    private segmenter: ImageSegmenter | null,
    private model: AttributeModel | null,
    private clothingModel: AttributeModel | null,
    private opts: FaceLensOptions,
    /** Where the attribute model runs, or null when attributes are off. */
    readonly backend: "webgpu" | "wasm" | null,
    /** Where MediaPipe runs. */
    readonly trackerDelegate: "GPU" | "CPU",
  ) {
    this.direction = new DirectionClassifier({ ...DEFAULT_THRESHOLDS, ...opts.thresholds });
    if (handLandmarker) this.lastHands = { left: null, right: null, total: 0 };
    if (segmenter) this.lastClothing = { colors: [], style: null, pattern: null };
  }

  static async create(opts: FaceLensOptions = {}): Promise<FaceLens> {
    const fileset = await FilesetResolver.forVisionTasks(opts.mediapipeWasm ?? DEFAULTS.mediapipeWasm);
    const [face, hands, segmenter, model, cloth] = await Promise.all([
      withDelegate((delegate) =>
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: opts.faceLandmarker ?? DEFAULTS.faceLandmarker, delegate },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        }),
      ),
      opts.hands
        ? withDelegate((delegate) =>
            HandLandmarker.createFromOptions(fileset, {
              baseOptions: { modelAssetPath: opts.handLandmarker ?? DEFAULTS.handLandmarker, delegate },
              runningMode: "VIDEO",
              numHands: 2,
            }),
          )
        : null,
      opts.clothing
        ? withDelegate((delegate) =>
            ImageSegmenter.createFromOptions(fileset, {
              baseOptions: { modelAssetPath: opts.segmenter ?? DEFAULTS.segmenter, delegate },
              runningMode: "VIDEO",
              outputCategoryMask: true,
              outputConfidenceMasks: false,
            }),
          )
        : null,
      opts.attributes === false ? null : loadModel(opts, "student", true),
      opts.clothing ? loadModel(opts, "clothing", false) : null,
    ]);
    return new FaceLens(
      face.task, hands?.task ?? null, segmenter?.task ?? null,
      model?.model ?? null, cloth?.model ?? null, opts, model?.backend ?? cloth?.backend ?? null, face.delegate,
    );
  }

  /** Class names per attribute head, or {} when attributes are off. */
  get heads(): Record<string, string[]> {
    return this.model ? { ...this.model.meta.heads } : {};
  }

  /**
   * Analyzes one frame. Call once per video frame with a monotonically increasing timestamp
   * (e.g. `performance.now()`).
   */
  detect(source: FrameSource, timestamp: number): FaceLensResult {
    if (timestamp <= this.lastTs) timestamp = this.lastTs + 0.001;
    this.lastTs = timestamp;
    this.frame++;
    const face = this.detectFace(source, timestamp);
    if (this.handLandmarker && this.frame % (this.opts.handsEvery ?? 2) === 0) {
      this.lastHands = this.detectHands(source, timestamp);
    }
    if (this.segmenter && this.frame % (this.opts.clothingEvery ?? 8) === 0) {
      this.lastClothing = this.detectClothing(source, timestamp, face);
    }
    return { timestamp, face, hands: this.lastHands, clothing: this.lastClothing };
  }

  /**
   * Changes how often hands and clothing run, or the smoothing, while running — e.g. to
   * lighten the load on a slow device. Hands and clothing must have been enabled at create().
   */
  configure(options: Pick<FaceLensOptions, "handsEvery" | "clothingEvery" | "smoothing">) {
    this.opts = { ...this.opts, ...options, smoothing: { ...this.opts.smoothing, ...options.smoothing } };
  }

  /** Treats the current head pose as "center" (webcams sit above or below the eyes). */
  calibrate() {
    if (this.lastPose) this.direction.center = { ...this.lastPose };
  }

  async close() {
    this.landmarker.close();
    this.handLandmarker?.close();
    this.segmenter?.close();
    await this.model?.release();
    await this.clothingModel?.release();
  }

  private detectFace(source: FrameSource, timestamp: number): Face | null {
    const result = this.landmarker.detectForVideo(source, timestamp);
    const landmarks = result.faceLandmarks[0];
    if (!landmarks) {
      this.faceGeneration++;
      this.smoothExpr = null;
      this.smoothAttr = null;
      this.direction.reset();
      return null;
    }
    const [w, h] = sourceSize(source);
    const c = this.model?.meta.crop;
    const box = cropBox(landmarks, w, h, { scale: c?.scale ?? 1.7, shiftY: c?.shiftY ?? -0.12 });
    const raw = headPose(result.facialTransformationMatrixes[0], this.opts.mirrored ?? true);
    this.lastPose = raw;
    const direction = this.direction.update(raw);
    const expr = expressionScores(result.faceBlendshapes[0].categories);
    this.smoothExpr = this.smoothExpr ? blend(this.smoothExpr, expr, this.opts.smoothing?.expressions ?? 0.5) : expr;

    if (this.model) {
      const generation = this.faceGeneration;
      void this.model.run(source, box).then((p) => {
        if (!p || generation !== this.faceGeneration) return;
        const k = this.opts.smoothing?.attributes ?? 0.65;
        this.smoothAttr = this.smoothAttr ? blendProbs(this.smoothAttr, p, k) : p;
        this.opts.onAttributes?.(summarize(this.smoothAttr));
      });
    }
    return {
      landmarks,
      box,
      pose: this.direction.relative(raw),
      direction,
      expressions: { ...this.smoothExpr },
      attributes: this.smoothAttr ? summarize(this.smoothAttr) : null,
    };
  }

  private detectHands(source: FrameSource, timestamp: number): Hands {
    const r = this.handLandmarker!.detectForVideo(source, timestamp);
    const out: Hands = { left: null, right: null, total: 0 };
    r.landmarks.forEach((landmarks, i) => {
      const cat = r.handednesses[i]?.[0];
      // MediaPipe labels the person's own hand on an unmirrored frame (checked on HaGRID: 97%).
      const side = cat?.categoryName === "Left" ? "left" : "right";
      const up = fingersUp(r.worldLandmarks[i]);
      const hand: Hand = {
        count: Object.values(up).filter(Boolean).length,
        up,
        landmarks,
        handednessScore: cat?.score ?? 0,
      };
      // Two hands with the same label: keep the more confident one.
      const prev = out[side];
      if (!prev || hand.handednessScore > prev.handednessScore) out[side] = hand;
    });
    out.total = (out.left?.count ?? 0) + (out.right?.count ?? 0);
    return out;
  }

  private detectClothing(source: FrameSource, timestamp: number, face: Face | null): Clothing {
    const prev = this.lastClothing;
    if (this.clothingModel && face) {
      const [w, h] = sourceSize(source);
      const meta = this.clothingModel.meta;
      const { box, visible } = clothingBox(face.landmarks, w, h, meta.crop.scale ?? 2.2);
      if (visible >= (meta.crop.minVisible ?? 0.5)) {
        void this.clothingModel.run(source, box).then((p) => {
          if (!p) return;
          this.smoothCloth = this.smoothCloth ? blendProbs(this.smoothCloth, p, 0.5) : p;
        });
      } else {
        this.smoothCloth = null;
      }
    } else if (!face) {
      this.smoothCloth = null;
    }
    const cloth = this.smoothCloth ? summarize(this.smoothCloth) : null;
    const r = this.segmenter!.segmentForVideo(source, timestamp);
    try {
      const mask = r.categoryMask;
      let colors = prev?.colors ?? [];
      if (mask) {
        const [w, h] = sourceSize(source);
        const pw = 160;
        const ph = Math.max(1, Math.round((160 * h) / w));
        this.pixelCanvas.width = pw;
        this.pixelCanvas.height = ph;
        const ctx = this.pixelCanvas.getContext("2d", { willReadFrequently: true })!;
        ctx.drawImage(source, 0, 0, pw, ph);
        const pixels = ctx.getImageData(0, 0, pw, ph).data;
        // Only below the chin, so hats and hair accessories do not count.
        const chin = face ? Math.max(...face.landmarks.map((p) => p.y)) : 0;
        colors = clothingColors(mask.getAsUint8Array(), mask.width, mask.height, pixels, pw, ph, chin);
      }
      return { colors, style: cloth?.style ?? null, pattern: cloth?.pattern ?? null };
    } finally {
      r.close();
    }
  }
}

/** Creates a MediaPipe task on the GPU, falling back to the CPU. */
async function withDelegate<T>(make: (delegate: "GPU" | "CPU") => Promise<T>) {
  try {
    return { task: await make("GPU"), delegate: "GPU" as const };
  } catch {
    return { task: await make("CPU"), delegate: "CPU" as const };
  }
}

/** Loads `<name>.json` + `<name>.onnx`. A missing optional model resolves to null. */
async function loadModel(opts: FaceLensOptions, name: string, required: boolean) {
  const base = withSlash(opts.modelUrl ?? DEFAULTS.modelUrl);
  const res = await fetch(`${base}${name}.json`);
  // Hosts differ in the content type of .json (Hugging Face: text/plain); reject only an HTML
  // page, which is what an SPA dev server answers for a missing file.
  if (!res.ok || res.headers.get("content-type")?.includes("text/html")) {
    if (!required) return null;
    throw new Error(`face-lens: ${base}${name}.json → HTTP ${res.status}`);
  }
  const meta = (await res.json()) as ModelMeta;
  const heads = name === "student" && Array.isArray(opts.attributes) ? opts.attributes : Object.keys(meta.heads);
  for (const h of heads) if (!(h in meta.heads)) throw new Error(`face-lens: model has no "${h}" attribute`);

  const ort = await import("onnxruntime-web/webgpu");
  ort.env.wasm.wasmPaths = opts.ortWasm ?? DEFAULTS.ortWasm;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const wanted = opts.executionProvider ?? "auto";
  const providers = wanted === "auto" ? (["webgpu", "wasm"] as const) : ([wanted] as const);
  let lastError: unknown;
  for (const ep of providers) {
    try {
      const session = await serialized(() =>
        ort.InferenceSession.create(`${base}${name}.onnx`, { executionProviders: [ep], graphOptimizationLevel: "all" }),
      );
      return { model: new AttributeModel(ort, session, meta, ep, heads), backend: ep };
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError;
}

// Creating two WebGPU sessions at once makes one of them fall back to WASM; queue them.
let sessionQueue: Promise<unknown> = Promise.resolve();
function serialized<T>(make: () => Promise<T>): Promise<T> {
  const next = sessionQueue.then(make, make);
  sessionQueue = next.catch(() => undefined);
  return next;
}

function withSlash(url: string) {
  return url.endsWith("/") ? url : `${url}/`;
}

function blend<T extends Record<string, number>>(prev: T, next: T, a: number): T {
  const out = { ...next };
  for (const k in next) (out as Record<string, number>)[k] = prev[k] + (1 - a) * (next[k] - prev[k]);
  return out;
}

function blendProbs(prev: AttributeProbs, next: AttributeProbs, a: number): AttributeProbs {
  const out: AttributeProbs = {};
  for (const k in next) out[k] = blend(prev[k] ?? next[k], next[k], a);
  return out;
}

function summarize(probs: AttributeProbs): Attributes {
  const out: Attributes = {};
  for (const [key, p] of Object.entries(probs)) {
    let label = "";
    let confidence = -1;
    for (const [c, v] of Object.entries(p)) if (v > confidence) [label, confidence] = [c, v];
    out[key] = { label, confidence, probs: { ...p } };
  }
  return out;
}
