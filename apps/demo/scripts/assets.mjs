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

const MP = "https://storage.googleapis.com/mediapipe-models";
const models = {
  "face_landmarker.task": `${MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
  "hand_landmarker.task": `${MP}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  "selfie_multiclass_256x256.tflite": `${MP}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`,
};
for (const [name, url] of Object.entries(models)) {
  const dest = pub(`models/${name}`);
  if (existsSync(dest)) continue;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
}
console.log("public/ assets ready");
