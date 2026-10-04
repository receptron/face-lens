# @receptron/face-lens

Real-time face, hand and clothing attributes from a webcam, **entirely in the browser**.
No server; no frame leaves the device.

`lens.detect(video, timestamp)` returns `{ face, hands, clothing }`:

| Field | Values |
|---|---|
| `face.direction` | `center`, `up`, `down`, `left`, `right`, `up-left`, `up-right`, `down-left`, `down-right` |
| `face.pose` | `{ yaw, pitch }` in degrees, relative to a calibratable center |
| `face.expressions` | 0..1 each: `wink-left`, `wink-right`, `eyes-closed`, `smile`, `mouth-open`, `kiss`, `cheek-puff`, `brows-raised`, `frown`, `mouth-sideways` |
| `face.attributes` | `gender`, `age` (0-9 … 60+), `emotion` (7 classes), `hair` color, `eyes` color, `tongue` (out or not) — each with label, confidence and all probabilities |
| `hands` (opt-in) | raised fingers per hand: `left.count`, `right.count` (0..5), `total` (0..10), which fingers |
| `clothing` (opt-in) | up to 3 `colors` (`yellow`, `navy`, …), `style` (`t-shirt`, `hoodie`, `suit-blazer`, …), `pattern` (`solid`, `striped`, …) |

Face and hand tracking, head pose, expressions and clothing segmentation come from
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide). Attributes and clothing style
come from two small MobileNetV4 models on [onnxruntime-web](https://onnxruntime.ai/) — WebGPU when
available, WASM otherwise.

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

const lens = await FaceLens.create({ hands: true, clothing: true });

function frame() {
  const { face, hands, clothing } = lens.detect(video, performance.now());
  if (face) {
    console.log(face.direction, face.expressions["wink-left"] > 0.5, face.attributes?.emotion.label);
  }
  if (hands) console.log("fingers", hands.left?.count, hands.right?.count, hands.total);
  if (clothing) console.log(clothing.colors[0]?.name, clothing.style?.label);
  requestAnimationFrame(frame);
}
frame();
```

- `detect()` returns synchronously. `face` is `null` when no face is visible; `hands` and
  `clothing` are `null` when their option is off.
- Attributes and clothing style are estimated asynchronously and lag a frame or two;
  `face.attributes` is the latest smoothed estimate (`null` until the first one).
  Pass `onAttributes` to be called when a new one arrives.
- Hand tracking runs every 2nd frame and clothing every 8th by default (`handsEvery`,
  `clothingEvery`); in between, the last result is returned.
- `lens.calibrate()` makes the current head pose "center" — webcams usually sit above or below
  the eyes.
- Left/right: `wink-left` is the user's own left eye and `hands.left` the user's own left hand,
  which is also the left one in a mirrored selfie view. `face.pose.yaw` is positive when the
  face turns toward the right of a mirrored view (`mirrored: false` for an unmirrored one).
- `clothing.style` needs the torso in frame (the area below the chin); colors are `[]` and style
  `null` when too little of it is visible.

### Options

```ts
await FaceLens.create({
  attributes: ["emotion", "tongue"], // true (default) = all; false = no attribute model (nothing extra downloaded)
  hands: true,                       // default false
  clothing: true,                    // default false
  handsEvery: 2,                     // run hand tracking every N frames
  clothingEvery: 8,                  // run clothing every N frames
  executionProvider: "auto",         // "webgpu" | "wasm"
  mirrored: true,
  smoothing: { expressions: 0.5, attributes: 0.65 },               // 0 = off, closer to 1 = smoother
  thresholds: { yawOn: 16, yawOff: 11, pitchOn: 13, pitchOff: 9 }, // direction hysteresis, degrees
  onAttributes: (attributes) => {},
});
```

### Self-hosting

By default the models load from [Hugging Face](https://huggingface.co/snakajima/face-lens)
(tag `v1`), MediaPipe and onnxruntime-web's wasm from jsDelivr, and MediaPipe's models from
Google. To serve everything yourself:

```ts
await FaceLens.create({
  modelUrl: "/models/",                                  // student.{onnx,json}, clothing.{onnx,json}
  mediapipeWasm: "/mediapipe/wasm",                      // @mediapipe/tasks-vision/wasm/*
  faceLandmarker: "/models/face_landmarker.task",
  handLandmarker: "/models/hand_landmarker.task",
  segmenter: "/models/selfie_multiclass_256x256.tflite",
  ortWasm: { wasm: "/ort/ort-wasm-simd-threaded.asyncify.wasm" },
});
```

Downloads (cached by the browser after the first visit):

| Part | Size | When |
|---|---|---|
| MediaPipe wasm + face model | ~13 MB | always |
| attribute model (`student.onnx`) + onnxruntime-web wasm | ~17 MB + ~26 MB | `attributes` on |
| hand model | ~8 MB | `hands: true` |
| segmentation model + clothing model | ~16 MB + ~5 MB | `clothing: true` |

Camera access requires HTTPS (or localhost). Serving the page cross-origin isolated
(COOP/COEP headers) lets the WASM fallback use threads; WebGPU does not need it.

## Accuracy

| Output | How measured | Result |
|---|---|---|
| gender | FairFace validation (10,081 faces) | 95.5% |
| age (7 groups) | FairFace validation | 60.9% |
| emotion / hair / eyes | agreement with the teacher model | 90.9% / 85.5% / 86.7% |
| raised fingers | HaGRID hands, exact count | 83% (held-out); 96% on webcam-sized hands |
| left / right hand | HaGRID | 97–100% |
| clothing style / pattern | agreement with the teacher model | 64.5% / 85.4% |

## The models

Distilled from [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
and trained on [FairFace](https://github.com/joojs/fairface) (gender, age) and openly licensed
photos. Finger counting is a rule on MediaPipe's hand landmarks, tuned on
[HaGRID](https://github.com/hukenovs/hagrid). Model card, metrics and data sources:
[huggingface.co/snakajima/face-lens](https://huggingface.co/snakajima/face-lens).
Training code: [github.com/receptron/face-lens](https://github.com/receptron/face-lens).

## Limitations and responsible use

- Gender, age and emotion are guesses from appearance. They are often wrong and can be biased
  across groups. Do not use them to make decisions about people.
- The models deliberately do **not** estimate race or ethnicity.
- Under the EU AI Act, emotion recognition is prohibited in workplaces and education, and
  biometric categorisation of sensitive traits is prohibited. Check the rules where you deploy.
- Eye color is the weakest attribute at webcam resolution; clothing style struggles with small,
  distant or layered clothing; finger counts are least reliable with the hand pointing down or
  the back of the hand toward the camera.

## License

Code: Apache-2.0. Model weights: CC BY 4.0 (see the model card).
