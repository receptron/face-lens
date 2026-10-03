import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import {
  AttributeModel,
  cropBox,
  sourceSize,
  type AttributeProbs,
  type Box,
  type FrameSource,
  type ModelMeta,
} from "./attributes.js";
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
/** The exact crop the attribute model was trained on, e.g. for collecting training images. */
export { cropBox, drawCrop } from "./attributes.js";
export type { Box, Direction, DirectionThresholds, Expression, FrameSource, ModelMeta, Pose };

// Keep in sync with package.json dependencies.
const MEDIAPIPE_VERSION = "1.0.1";
const ORT_VERSION = "1.30.0";

export const DEFAULTS = {
  modelUrl: "https://huggingface.co/snakajima/face-lens/resolve/v1/",
  mediapipeWasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`,
  faceLandmarker:
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  ortWasm: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
};

export interface FaceLensOptions {
  /**
   * Learned attributes to run: `true` for every head the model has, `false` for none (only
   * direction and expressions, and onnxruntime-web is never loaded), or a list of head names.
   * Default `true`.
   */
  attributes?: boolean | string[];
  /** Base URL of `student.onnx` + `student.json`. Default: the Hugging Face repo, pinned. */
  modelUrl?: string;
  /** Directory of MediaPipe's wasm files. */
  mediapipeWasm?: string;
  /** URL of `face_landmarker.task`. */
  faceLandmarker?: string;
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

export interface FaceLensResult {
  timestamp: number;
  /** Face landmarks (478), normalized to the frame. */
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

export class FaceLens {
  private direction: DirectionClassifier;
  private smoothExpr: Record<Expression, number> | null = null;
  private smoothAttr: AttributeProbs | null = null;
  private lastPose: Pose | null = null;
  private lastTs = -1;
  private faceGeneration = 0;

  private constructor(
    private landmarker: FaceLandmarker,
    private model: AttributeModel | null,
    private opts: FaceLensOptions,
    /** Where the attribute model runs, or null when attributes are off. */
    readonly backend: "webgpu" | "wasm" | null,
    /** Where face tracking runs. */
    readonly trackerDelegate: "GPU" | "CPU",
  ) {
    this.direction = new DirectionClassifier({ ...DEFAULT_THRESHOLDS, ...opts.thresholds });
  }

  static async create(opts: FaceLensOptions = {}): Promise<FaceLens> {
    const [{ landmarker, delegate }, model] = await Promise.all([
      createLandmarker(opts),
      opts.attributes === false ? Promise.resolve(null) : loadModel(opts),
    ]);
    return new FaceLens(landmarker, model?.model ?? null, opts, model?.backend ?? null, delegate);
  }

  /** Class names per attribute head, or {} when attributes are off. */
  get heads(): Record<string, string[]> {
    return this.model ? Object.fromEntries(Object.entries(this.model.meta.heads)) : {};
  }

  /**
   * Analyzes one frame. Call once per video frame with a monotonically increasing timestamp
   * (e.g. `performance.now()`). Returns null when no face is visible.
   */
  detect(source: FrameSource, timestamp: number): FaceLensResult | null {
    if (timestamp <= this.lastTs) timestamp = this.lastTs + 0.001;
    this.lastTs = timestamp;
    const result = this.landmarker.detectForVideo(source, timestamp);
    const landmarks = result.faceLandmarks[0];
    if (!landmarks) {
      this.lost();
      return null;
    }
    const [w, h] = sourceSize(source);
    const crop = this.model?.meta.crop ?? { scale: 1.7, shiftY: -0.12 };
    const box = cropBox(landmarks, w, h, crop);
    const raw = headPose(result.facialTransformationMatrixes[0], this.opts.mirrored ?? true);
    this.lastPose = raw;
    const direction = this.direction.update(raw);
    const a = this.opts.smoothing?.expressions ?? 0.5;
    const expr = expressionScores(result.faceBlendshapes[0].categories);
    this.smoothExpr = this.smoothExpr ? blend(this.smoothExpr, expr, a) : expr;

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
      timestamp,
      landmarks,
      box,
      pose: this.direction.relative(raw),
      direction,
      expressions: { ...this.smoothExpr },
      attributes: this.smoothAttr ? summarize(this.smoothAttr) : null,
    };
  }

  /** Treats the current head pose as "center" (webcams sit above or below the eyes). */
  calibrate() {
    if (this.lastPose) this.direction.center = { ...this.lastPose };
  }

  async close() {
    this.landmarker.close();
    await this.model?.release();
  }

  private lost() {
    this.faceGeneration++;
    this.smoothExpr = null;
    this.smoothAttr = null;
    this.direction.reset();
  }
}

async function createLandmarker(opts: FaceLensOptions) {
  const fileset = await FilesetResolver.forVisionTasks(opts.mediapipeWasm ?? DEFAULTS.mediapipeWasm);
  for (const delegate of ["GPU", "CPU"] as const) {
    try {
      const landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: opts.faceLandmarker ?? DEFAULTS.faceLandmarker, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
      return { landmarker, delegate };
    } catch (e) {
      if (delegate === "CPU") throw e;
    }
  }
  throw new Error("unreachable");
}

async function loadModel(opts: FaceLensOptions) {
  const base = withSlash(opts.modelUrl ?? DEFAULTS.modelUrl);
  const res = await fetch(`${base}student.json`);
  if (!res.ok) throw new Error(`face-lens: ${base}student.json → HTTP ${res.status}`);
  const meta = (await res.json()) as ModelMeta;
  const heads = Array.isArray(opts.attributes) ? opts.attributes : Object.keys(meta.heads);
  for (const h of heads) if (!(h in meta.heads)) throw new Error(`face-lens: model has no "${h}" attribute`);

  const ort = await import("onnxruntime-web/webgpu");
  ort.env.wasm.wasmPaths = opts.ortWasm ?? DEFAULTS.ortWasm;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const wanted = opts.executionProvider ?? "auto";
  const providers = wanted === "auto" ? (["webgpu", "wasm"] as const) : ([wanted] as const);
  let lastError: unknown;
  for (const ep of providers) {
    try {
      const session = await ort.InferenceSession.create(`${base}student.onnx`, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      return { model: new AttributeModel(ort, session, meta, ep, heads), backend: ep };
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError;
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
