import * as ort from "onnxruntime-web/webgpu";
import { drawCrop, type Box } from "./face";

const BASE = import.meta.env.BASE_URL;

// The bundled build carries its JS loader; only the .wasm binary is fetched from public/.
ort.env.wasm.wasmPaths = { wasm: `${BASE}ort/ort-wasm-simd-threaded.asyncify.wasm` };
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

/** Written by train/export.py next to student.onnx. */
export interface StudentMeta {
  inputSize: number;
  mean: [number, number, number];
  std: [number, number, number];
  /** Output name → class names, in logit order. */
  heads: Record<string, string[]>;
}

export type Probs = Record<string, number[]>;

export class Student {
  private busy = false;
  private canvas = new OffscreenCanvas(1, 1);

  private constructor(
    private session: ort.InferenceSession,
    readonly meta: StudentMeta,
    readonly backend: string,
  ) {}

  /** Returns null when no trained model has been exported yet. */
  static async load(): Promise<Student | null> {
    const res = await fetch(`${BASE}models/student.json`);
    if (!res.ok || !res.headers.get("content-type")?.includes("json")) return null;
    const meta = (await res.json()) as StudentMeta;
    const url = `${BASE}models/student.onnx`;
    for (const ep of ["webgpu", "wasm"] as const) {
      try {
        const session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
        return new Student(session, meta, ep);
      } catch (e) {
        if (ep === "wasm") throw e;
        console.warn("WebGPU execution provider failed, falling back to WASM", e);
      }
    }
    return null;
  }

  /** Runs one crop. Returns null if the previous frame is still in flight. */
  async run(video: HTMLVideoElement, box: Box): Promise<Probs | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      const n = this.meta.inputSize;
      const ctx = drawCrop(video, box, this.canvas, n);
      const { data } = ctx.getImageData(0, 0, n, n);
      const input = new Float32Array(3 * n * n);
      const { mean, std } = this.meta;
      for (let i = 0, p = 0; i < n * n; i++, p += 4) {
        input[i] = (data[p] / 255 - mean[0]) / std[0];
        input[n * n + i] = (data[p + 1] / 255 - mean[1]) / std[1];
        input[2 * n * n + i] = (data[p + 2] / 255 - mean[2]) / std[2];
      }
      const out = await this.session.run({ input: new ort.Tensor("float32", input, [1, 3, n, n]) });
      const probs: Probs = {};
      for (const key of Object.keys(this.meta.heads)) probs[key] = softmax(out[key].data as Float32Array);
      return probs;
    } finally {
      this.busy = false;
    }
  }
}

function softmax(logits: Float32Array): number[] {
  const max = Math.max(...logits);
  const e = Array.from(logits, (x) => Math.exp(x - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}
