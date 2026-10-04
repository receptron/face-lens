"""Collect openly licensed upper-body photos and cut the clothing crop the browser uses.

Openverse is queried with license=by,cc0,pdm only (the model is published under CC BY 4.0).
Each photo goes through MediaPipe; the crop is the square below the chin defined in
labels.json `clothing.crop` (the same geometry as packages/face-lens). Crops that are mostly
outside the photo are dropped.

    python teacher/fetch_clothing.py          # → data/clothing/{crops/, manifest.jsonl}
"""

import hashlib
import json
import sys
import time
import urllib.parse
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_tongue import download, get_json, license_ok  # noqa: E402
from prepare import make_landmarker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "teacher/data/clothing"
CROP = json.loads((ROOT / "labels.json").read_text())["clothing"]["crop"]

GARMENTS = [
    "t-shirt", "graphic t-shirt", "striped shirt", "plaid shirt", "button-up shirt", "dress shirt",
    "polo shirt", "blouse", "sweater", "knit sweater", "cardigan", "turtleneck", "hoodie",
    "sweatshirt", "jacket", "denim jacket", "leather jacket", "coat", "winter coat", "suit",
    "blazer", "dress", "sundress", "tank top", "sleeveless top", "uniform", "jersey", "kimono",
]
# Round 2: more garment words, and plain people photos (Bonsai decides the garment).
GARMENTS_2 = [
    "tee shirt", "jumper", "pullover", "fleece", "parka", "trench coat", "puffer jacket", "windbreaker",
    "bomber jacket", "vest", "waistcoat", "tuxedo", "gown", "camisole", "henley shirt", "flannel shirt",
    "oxford shirt", "linen shirt", "sports jersey", "lab coat", "scrubs", "overalls", "raincoat",
    "track jacket", "crop top", "halter top", "shirt and tie", "sweater vest", "poncho", "apron",
]
PEOPLE = [
    "portrait", "headshot", "speaker", "selfie", "street portrait", "office worker", "student",
    "graduation", "concert crowd", "festival people", "market vendor", "teacher", "scientist",
    "engineer", "musician", "chef", "volunteer", "tourist", "conference", "interview",
    "smiling woman", "smiling man", "young man", "young woman", "old man", "old woman",
    "family photo", "friends", "team photo", "commuter", "barista", "nurse", "artist portrait",
]
# Short queries: Openverse matches titles and tags, so long phrases return almost nothing.
WHO = ["", "man ", "woman "]


def queries():
    for g in GARMENTS:
        for who in WHO:
            yield f"{who}{g}"
    for g in GARMENTS_2:
        for who in WHO:
            yield f"{who}{g}"
    yield from PEOPLE


def candidates():
    """All candidates so far; queries already run (recorded in queries_done.json) are skipped."""
    path = OUT / "candidates.json"
    done_path = OUT / "queries_done.json"
    out = json.loads(path.read_text()) if path.exists() else []
    done = set(json.loads(done_path.read_text())) if done_path.exists() else {c["query"] for c in out}
    seen = {c["url"] for c in out}
    for q in queries():
        if q in done:
            continue
        for page in range(1, 13):
            params = {"q": q, "page": page, "page_size": 20, "license": "by,cc0,pdm"}
            try:
                d = get_json("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(params))
            except Exception as e:  # noqa: BLE001
                print(f"openverse {q!r} p{page}: {e}", flush=True)
                break
            for r in d.get("results", []):
                url = r.get("url")
                lic = f"cc-{r.get('license')} {r.get('license_version') or ''}".strip()
                if r.get("license") in ("cc0", "pdm"):
                    lic = "cc0" if r["license"] == "cc0" else "public domain"
                if not url or url in seen or not license_ok(lic):
                    continue
                seen.add(url)
                out.append({"source": f"openverse/{r.get('source')}", "title": r.get("title"),
                            "page": r.get("foreign_landing_url"), "url": url, "license": lic,
                            "creator": r.get("creator"), "query": q})
            if page >= d.get("page_count", 0):
                break
            time.sleep(2)
        done.add(q)
        path.write_text(json.dumps(out, indent=1))
        done_path.write_text(json.dumps(sorted(done)))
        print(f"{q!r}: {len(out)} total", flush=True)
    return out


def torso_crop(landmarker, img: Image.Image):
    """Mirror of clothingBox() in packages/face-lens. Returns (crop, visible fraction) or None."""
    rgb = np.asarray(img.convert("RGB"))
    r = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not r.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    xs = [p.x * w for p in r.face_landmarks[0]]
    ys = [p.y * h for p in r.face_landmarks[0]]
    face_h = max(ys) - min(ys)
    if face_h < 40:
        return None
    size = face_h * CROP["scale"]
    x0 = (min(xs) + max(xs)) / 2 - size / 2
    y0 = max(ys)
    vis_w = max(0.0, min(w, x0 + size) - max(0.0, x0))
    vis_h = max(0.0, min(h, y0 + size) - max(0.0, y0))
    visible = (vis_w * vis_h) / (size * size)
    if visible < CROP["minVisible"]:
        return None
    out = CROP["saveSize"]
    k = out / size
    canvas = Image.new("RGB", (out, out))
    scaled = img.convert("RGB").resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.BILINEAR)
    canvas.paste(scaled, (round(-x0 * k), round(-y0 * k)))
    return canvas, visible


def main():
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    cands = candidates()
    print(f"{len(cands)} candidates", flush=True)
    landmarker = make_landmarker()
    manifest = OUT / "manifest.jsonl"
    done = {json.loads(l)["key"] for l in manifest.read_text().splitlines() if l} if manifest.exists() else set()
    tried = OUT / "tried.txt"
    tried_keys = set(tried.read_text().split()) if tried.exists() else set()
    kept = 0
    with manifest.open("a") as mf, tried.open("a") as tf:
        for i, c in enumerate(cands):
            key = hashlib.sha1(c["url"].encode()).hexdigest()[:16]
            if key in done or key in tried_keys:
                continue
            tf.write(key + "\n")
            img = download(c["url"])
            if img is None or min(img.size) < 200:
                continue
            res = torso_crop(landmarker, img)
            if res is None:
                continue
            crop, visible = res
            crop.save(OUT / "crops" / f"{key}.jpg", quality=92)
            # Every 10th crop is held out for validation.
            split = "val" if int(key, 16) % 10 == 0 else "train"
            mf.write(json.dumps({"key": key, "path": f"crops/{key}.jpg", "split": split,
                                 "visible": round(visible, 2), **c}) + "\n")
            mf.flush()
            kept += 1
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(cands)} tried, {kept} new crops", flush=True)
    print(f"done, {kept} new crops", flush=True)


if __name__ == "__main__":
    main()
