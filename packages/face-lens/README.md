# @receptron/face-lens

Real-time face attributes from a webcam, **entirely in the browser**. No server, no frames leave the device.

| Output | Values |
|---|---|
| `direction` | `center`, `up`, `down`, `left`, `right`, `up-left`, `up-right`, `down-left`, `down-right` |
| `pose` | `{ yaw, pitch }` in degrees, relative to a calibratable center |
| `expressions` | 0..1 each: `wink-left`, `wink-right`, `eyes-closed`, `smile`, `mouth-open`, `kiss`, `cheek-puff`, `brows-raised`, `frown`, `mouth-sideways` |
| `attributes` | `gender`, `age` (0-9 … 60+), `emotion` (7 classes), `hair` color, `eyes` color, `tongue` (out or not) — each with label, confidence and full probabilities |

Face tracking, head pose and expressions come from [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
The attributes come from a small MobileNetV4 (≈17 MB) on [onnxruntime-web](https://onnxruntime.ai/) — WebGPU when available, WASM otherwise.

## Install

```sh
npm install @receptron/face-lens
```

## Use

```ts
import { FaceLens } from "@receptron/face-lens";

const video = document.querySelector("video")!;
video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
await video.play();

const lens = await FaceLens.create();

function frame() {
  const r = lens.detect(video, performance.now());
  if (r) {
    console.log(r.direction, r.expressions["wink-left"] > 0.5, r.attributes?.emotion.label);
  }
  requestAnimationFrame(frame);
}
frame();
```

- `detect()` is synchronous for tracking, direction and expressions. Attributes are estimated
  asynchronously and lag a frame or two; `r.attributes` is the latest smoothed estimate.
  Pass `onAttributes` to be called when a new one arrives.
- `lens.calibrate()` makes the current head pose "center" — webcams usually sit above or below
  the eyes.
- Left/right are as seen in a mirrored selfie view (`mirrored: true`, the default):
  `wink-left` is the user's own left eye.

### Options

```ts
await FaceLens.create({
  attributes: ["emotion", "tongue"], // or false: no attribute model at all (nothing extra is downloaded)
  executionProvider: "auto",         // "webgpu" | "wasm"
  mirrored: true,
  smoothing: { expressions: 0.5, attributes: 0.65 },  // 0 = off, closer to 1 = smoother
  thresholds: { yawOn: 16, yawOff: 11, pitchOn: 13, pitchOff: 9 },  // direction hysteresis, degrees
  // Self-hosting (defaults: Hugging Face, jsDelivr, Google):
  modelUrl: "/models/",              // student.onnx + student.json
  mediapipeWasm: "/mediapipe/wasm",
  faceLandmarker: "/models/face_landmarker.task",
  ortWasm: "/ort/",
});
```

Default downloads: MediaPipe (~13 MB) and, with attributes on, the attribute model (~17 MB) and
onnxruntime-web's wasm (~26 MB). The browser caches them after the first visit.
Camera access requires HTTPS (or localhost).

## The model

Trained on [FairFace](https://github.com/joojs/fairface) (gender, age; CC BY 4.0) and on soft labels
distilled from [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
(emotion, hair, eye color, tongue), plus openly licensed tongue-out photos. Model card, metrics and
data sources: [huggingface.co/snakajima/face-lens](https://huggingface.co/snakajima/face-lens).
Training code: [github.com/receptron/face-lens](https://github.com/receptron/face-lens).

## Limitations and responsible use

- Gender, age and emotion are guesses from appearance. They are often wrong and can be biased
  across groups. Do not use them to make decisions about people.
- The model deliberately does **not** estimate race or ethnicity.
- Under the EU AI Act, emotion recognition is prohibited in workplaces and education, and
  biometric categorisation of sensitive traits is prohibited. Check the rules where you deploy.
- Eye color is the weakest attribute at webcam resolution.

## License

Code: Apache-2.0. Model weights: see the model card.
