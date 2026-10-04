# Face Lens

Real-time face, hand and clothing attributes from the webcam, entirely in the browser.

[![npm](https://img.shields.io/npm/v/@receptron/face-lens)](https://www.npmjs.com/package/@receptron/face-lens)

**Live demo: [receptron.github.io/face-lens/lens/](https://receptron.github.io/face-lens/lens/)** —
allow the camera; everything runs in your browser.

## Use it: `@receptron/face-lens`

The library is on npm: **[@receptron/face-lens](https://www.npmjs.com/package/@receptron/face-lens)**.

```sh
npm install @receptron/face-lens
```

```ts
import { FaceLens } from "@receptron/face-lens";

const lens = await FaceLens.create({ hands: true, clothing: true });
const { face, hands, clothing } = lens.detect(video, performance.now());
```

- **API documentation:** [packages/face-lens/README.md](packages/face-lens/README.md) — every
  option, the result fields, self-hosting and accuracy.
- **Models:** [huggingface.co/snakajima/face-lens](https://huggingface.co/snakajima/face-lens) —
  model card, metrics and training-data attribution (CC BY 4.0).

## What it detects

| Output | How |
|---|---|
| Face direction (center + 8) | MediaPipe head pose, thresholds with hysteresis |
| Wink L/R, eyes closed, smile, mouth open, kiss, cheek puff, brows raised, frown, mouth sideways | Rules on MediaPipe's 52 blendshapes |
| Gender, age group | Student CNN trained on FairFace human labels |
| Emotion, hair color, eye color, tongue out | Student CNN distilled from **Ternary Bonsai 2 27B** soft labels |
| Raised fingers per hand (0–5 each) | Rule on MediaPipe hand landmarks, tuned on HaGRID |
| Clothing colors | MediaPipe selfie segmentation + color naming in CIE LCh |
| Clothing style and pattern | Small CNN distilled from Bonsai on the area below the chin |

Race is deliberately not estimated (EU AI Act). The models are MobileNetV4s (17 MB + 5 MB ONNX,
fp16 weights) on onnxruntime-web (WebGPU, WASM fallback). Bonsai never runs in the browser; it
labels the training crops offline (MLX on Apple Silicon).

## Repository layout

The site (landing page at `/`, demo at `/lens/`) deploys to GitHub Pages from `main` via
`.github/workflows/pages.yml`; the demo loads the models from Hugging Face.


| Path | What |
|---|---|
| `packages/face-lens` | **`@receptron/face-lens`**, the npm library (API only, no UI) |
| `apps/demo` | The demo page ([receptron.github.io/face-lens/lens/](https://receptron.github.io/face-lens/lens/); later swarmstrike.com/lens/) |
| `apps/game` | Swarm Strike, the drone-swarm game (coming) |
| `teacher/`, `train/` | Offline pipeline: Bonsai 27B teacher labels → student training → ONNX export |

## Run the demo

```sh
npm install        # also copies the wasm runtimes and fetches the MediaPipe face model into apps/demo/public/
npm run dev        # http://localhost:5173
```

`apps/demo/public/models/student.{onnx,json}` is produced by training and is not in git; until it
exists the page runs with direction and expressions only. Set `VITE_MODEL_URL` to load it from
elsewhere (e.g. the Hugging Face repo).

`labels.json` is the single source of truth for classes and the face-crop spec, shared by the
teacher and training; the exported `student.json` carries both to the library.

## Train / retrain

```sh
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir teacher/models/bonsai2-27b-mlx
huggingface-cli download HuggingFaceM4/FairFace --repo-type dataset --include "1.25/*" --local-dir teacher/data/fairface
./pipeline.sh      # crop → Bonsai labels (new crops only) → train → export to apps/demo/public/models/
```

- `teacher/prepare.py` — FairFace + browser captures → crops identical to the browser's (`data/manifest.jsonl`)
- `teacher/bonsai.py` — the teacher: one prefill per image (question prefix cached), answer
  distributions read from logits, no free text (~2.5 s/image on an M2 Max)
- `teacher/label.py` — resumable labeling into `data/teacher.jsonl`
- `teacher/fetch_tongue.py` — openly licensed tongue-out photos from Wikimedia Commons and Openverse,
  cropped and checked by Bonsai into `captures/tongue={yes,no}/`; sources and licenses in
  `data/web/sources.jsonl` (some are CC BY-NC: keep this non-commercial)
- `train/train.py`, `train/export.py` — student training (PyTorch MPS) and ONNX export

### Your own clips

Rare classes (tongue out, strong emotions) are scarce in FairFace. Press **Capture** in the dev
site, pick a tag such as `tongue=yes`, and hold the button while doing it — crops land in
`teacher/data/captures/<tag>/`. Then run `./pipeline.sh`.

## Notes

- MediaPipe for Python must stay at 0.10.21: 1.0.1 aborts on macOS in a Metal service check and hangs.
- Gender, age and emotion are guesses from appearance, often wrong and possibly biased across
  groups. Emotion recognition is prohibited at work and in education under the EU AI Act.

## Publishing the library

From the repository root:

```sh
npm version patch -w @receptron/face-lens   # bump the version first; a published version can't be reused
npm run publish:lib                          # builds, then publishes @receptron/face-lens (public)
```

Do not run a bare `npm publish` at the repository root: the root is a private workspace without a
version, and npm fails with "Cannot read properties of null (reading 'prerelease')".
