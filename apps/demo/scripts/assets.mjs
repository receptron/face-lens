// Copies runtime files out of node_modules and fetches the MediaPipe face model into public/.
// Runs on `npm install`; these are generated, so they stay out of git.
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const modules = new URL("../../../node_modules/", import.meta.url).pathname;
const pub = (p) => join(root, "public", p);

mkdirSync(pub("mediapipe/wasm"), { recursive: true });
mkdirSync(pub("ort"), { recursive: true });
mkdirSync(pub("models"), { recursive: true });

const mpWasm = join(modules, "@mediapipe/tasks-vision/wasm");
for (const f of readdirSync(mpWasm)) copyFileSync(join(mpWasm, f), pub(`mediapipe/wasm/${f}`));
copyFileSync(
  join(modules, "onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm"),
  pub("ort/ort-wasm-simd-threaded.asyncify.wasm"),
);

const model = pub("models/face_landmarker.task");
if (!existsSync(model)) {
  const url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  writeFileSync(model, Buffer.from(await res.arrayBuffer()));
}
console.log("public/ assets ready");
