import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type * as Ort from "onnxruntime-web";

/** `student.json`, published next to `student.onnx`. */
export interface ModelMeta {
  inputSize: number;
  mean: [number, number, number];
  std: [number, number, number];
  /** Output name → class names, in logit order. */
  heads: Record<string, string[]>;
  /** Square crop around the landmark box; the model was trained on exactly this crop. */
  crop: { scale: number; shiftY: number };
}

export interface Box {
  x: number;
  y: number;
  size: number;
}

export type FrameSource = HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLImageElement;

export function sourceSize(src: FrameSource): [number, number] {
  if (typeof HTMLVideoElement !== "undefined" && src instanceof HTMLVideoElement) return [src.videoWidth, src.videoHeight];
  if (typeof HTMLImageElement !== "undefined" && src instanceof HTMLImageElement) return [src.naturalWidth, src.naturalHeight];
  return [src.width, src.height];
}

/** Square crop in source pixels around the landmark box. */
export function cropBox(landmarks: NormalizedLandmark[], width: number, height: number, crop: ModelMeta["crop"]): Box {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const w = (maxX - minX) * width;
  const h = (maxY - minY) * height;
  const size = Math.max(w, h) * crop.scale;
  const cx = ((minX + maxX) / 2) * width;
  const cy = ((minY + maxY) / 2) * height + h * crop.shiftY;
  return { x: cx - size / 2, y: cy - size / 2, size };
}

/** Draws the crop at `out`×`out`; area outside the frame is black (as in training). */
export function drawCrop(src: FrameSource, box: Box, canvas: OffscreenCanvas, out: number) {
  canvas.width = out;
  canvas.height = out;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, out, out);
  const [w, h] = sourceSize(src);
  const k = out / box.size;
  ctx.drawImage(src, -box.x * k, -box.y * k, w * k, h * k);
  return ctx;
}

export type AttributeProbs = Record<string, Record<string, number>>;

export class AttributeModel {
  private busy = false;
  private canvas = new OffscreenCanvas(1, 1);

  constructor(
    private ort: typeof Ort,
    private session: Ort.InferenceSession,
    readonly meta: ModelMeta,
    readonly backend: "webgpu" | "wasm",
    private heads: string[],
  ) {}

  /** Runs one crop; resolves null if the previous one is still in flight. */
  async run(src: FrameSource, box: Box): Promise<AttributeProbs | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      const n = this.meta.inputSize;
      const { data } = drawCrop(src, box, this.canvas, n).getImageData(0, 0, n, n);
      const input = new Float32Array(3 * n * n);
      const { mean, std } = this.meta;
      for (let i = 0, p = 0; i < n * n; i++, p += 4) {
        input[i] = (data[p] / 255 - mean[0]) / std[0];
        input[n * n + i] = (data[p + 1] / 255 - mean[1]) / std[1];
        input[2 * n * n + i] = (data[p + 2] / 255 - mean[2]) / std[2];
      }
      const out = await this.session.run(
        { input: new this.ort.Tensor("float32", input, [1, 3, n, n]) },
        this.heads,
      );
      const probs: AttributeProbs = {};
      for (const key of this.heads) {
        const p = softmax(out[key].data as Float32Array);
        probs[key] = Object.fromEntries(this.meta.heads[key].map((c, i) => [c, p[i]]));
      }
      return probs;
    } finally {
      this.busy = false;
    }
  }

  release() {
    return this.session.release();
  }
}

function softmax(logits: Float32Array): number[] {
  let max = -Infinity;
  for (const x of logits) max = Math.max(max, x);
  const e = Array.from(logits, (x) => Math.exp(x - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}
