"""Build training crops with the same crop as the browser (labels.json `crop`).

FairFace rows are cropped through MediaPipe exactly like camera frames and keep
their human gender / age / race labels. Browser captures under
data/captures/<tag>/ are already crops; their tag (e.g. tongue=yes) becomes a
human label too. Everything lands in data/manifest.jsonl.

    python teacher/prepare.py --train 12000 --val 1500
"""

import argparse
import io
import json
import random
import zlib
from pathlib import Path

import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "teacher/data"
LABELS = json.loads((ROOT / "labels.json").read_text())
CROP = LABELS["crop"]
CLASSES = {k: v["classes"] for k, v in LABELS["learned"].items()}

FF_GENDER = ["male", "female"]
FF_AGE = ["0-9", "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+", "60+"]
FF_RACE = ["east-asian", "indian", "black", "white", "middle-eastern", "latino-hispanic", "southeast-asian"]


def make_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "teacher/models/face_landmarker.task")),
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_face(landmarker, img: Image.Image) -> Image.Image | None:
    """Mirror of cropBox() + drawCrop() in src/face.ts."""
    rgb = np.asarray(img.convert("RGB"))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    pts = result.face_landmarks[0]
    h, w = rgb.shape[:2]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    bw = (max(xs) - min(xs)) * w
    bh = (max(ys) - min(ys)) * h
    size = max(bw, bh) * CROP["scale"]
    cx = (min(xs) + max(xs)) / 2 * w
    cy = (min(ys) + max(ys)) / 2 * h + bh * CROP["shiftY"]
    x0, y0 = cx - size / 2, cy - size / 2
    out = CROP["saveSize"]
    canvas = Image.new("RGB", (out, out))
    k = out / size
    scaled = img.convert("RGB").resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BILINEAR)
    canvas.paste(scaled, (round(-x0 * k), round(-y0 * k)))
    return canvas


def fairface_rows(split: str, n: int, seed: int):
    files = sorted((DATA / "fairface/1.25").glob(f"{split}-*.parquet"))
    tables = [pq.read_table(f, columns=["age", "gender", "race"]) for f in files]
    total = sum(t.num_rows for t in tables)
    picks = sorted(random.Random(seed).sample(range(total), min(n, total)))
    offset = 0
    for f, t in zip(files, tables):
        local = [i - offset for i in picks if offset <= i < offset + t.num_rows]
        offset += t.num_rows
        if not local:
            continue
        full = pq.read_table(f).take(local).to_pylist()
        for i, row in zip(local, full):
            yield f"{f.stem.split('-')[0]}-{f.stem.split('-')[1]}-{i}", row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--val", type=int, default=1500)
    args = ap.parse_args()

    landmarker = make_landmarker()
    manifest = DATA / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["path"] for l in manifest.read_text().splitlines() if l}
    out = manifest.open("a")
    stats = {"kept": 0, "no_face": 0, "skipped": 0}

    for split, n in (("train", args.train), ("validation", args.val)):
        dest = DATA / "crops" / split
        dest.mkdir(parents=True, exist_ok=True)
        for rid, row in fairface_rows(split, n, seed=0):
            rel = f"crops/{split}/{rid}.jpg"
            if rel in done:
                stats["skipped"] += 1
                continue
            crop = crop_face(landmarker, Image.open(io.BytesIO(row["image"]["bytes"])))
            if crop is None:
                stats["no_face"] += 1
                continue
            crop.save(DATA / rel, quality=92)
            ff = {"gender": FF_GENDER[row["gender"]], "age": FF_AGE[row["age"]], "race": FF_RACE[row["race"]]}
            # Only heads that labels.json still defines (race is not one: EU AI Act).
            labels = {k: CLASSES[k].index(v) for k, v in ff.items() if k in CLASSES}
            out.write(json.dumps({"path": rel, "split": "val" if split == "validation" else "train",
                                  "source": "fairface", "labels": labels}) + "\n")
            stats["kept"] += 1
            if stats["kept"] % 500 == 0:
                print(stats, flush=True)

    # Browser captures: tag "key=value" is a human label; "free" carries none.
    for path in sorted((DATA / "captures").glob("*/*.jpg")):
        rel = str(path.relative_to(DATA))
        if rel in done:
            continue
        tag = path.parent.name
        labels = {}
        if "=" in tag:
            key, value = tag.split("=", 1)
            if key in CLASSES and value in CLASSES[key]:
                labels[key] = CLASSES[key].index(value)
        # Every 10th capture is held out for validation.
        split = "val" if zlib.crc32(rel.encode()) % 10 == 0 else "train"
        out.write(json.dumps({"path": rel, "split": split, "source": "capture", "labels": labels}) + "\n")
        stats["kept"] += 1
    out.close()
    print("done", stats)


if __name__ == "__main__":
    main()
