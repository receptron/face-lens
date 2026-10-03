# Face Lens

Real-time face attributes from the webcam, entirely in the browser.

| Output | How |
|---|---|
| Face direction (center + 8) | MediaPipe head pose, thresholds with hysteresis (`src/face.ts`) |
| Wink L/R, eyes closed, smile, mouth open, kiss, cheek puff, brows raised, frown, mouth sideways | Rules on MediaPipe's 52 blendshapes |
| Gender, age group, race | Student CNN trained on FairFace human labels |
| Emotion, hair color, eye color, tongue out | Student CNN distilled from **Ternary Bonsai 2 27B** soft labels |

The student is a MobileNetV4 (≈34 MB ONNX) running on onnxruntime-web (WebGPU, WASM fallback).
Bonsai never runs in the browser; it labels the training crops offline on this Mac (MLX).

## Run the site

```sh
npm install        # also copies the wasm runtimes and fetches the MediaPipe face model into public/
npm run dev        # http://localhost:5173
```

`public/models/student.{onnx,json}` is produced by training and is not in git; until it exists
the page runs with direction and expressions only.

`labels.json` is the single source of truth for classes and the face-crop spec,
shared by the page, the teacher and training.

## Train / retrain

```sh
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir teacher/models/bonsai2-27b-mlx
huggingface-cli download HuggingFaceM4/FairFace --repo-type dataset --include "1.25/*" --local-dir teacher/data/fairface
./pipeline.sh      # crop → Bonsai labels (new crops only) → train → export to public/models/
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
- Gender, age, race and emotion are guesses from appearance, often wrong and biased across groups.
  Emotion recognition and race categorisation are prohibited uses under the EU AI Act in many
  settings; keep this a demo.
