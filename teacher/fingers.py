"""Finger counting from MediaPipe hand landmarks, and its check against HaGRID.

The rule is a straight port target for packages/face-lens/src/hands.ts — keep them identical.
It uses world landmarks (meters, hand-centred), so it does not depend on hand rotation or size.

    python teacher/fingers.py            # accuracy + handedness check on the HaGRID sample
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAGRID = ROOT / "teacher/data/hagrid"

# Joints: (MCP, PIP, DIP, TIP) for index..pinky; thumb (CMC, MCP, IP, TIP).
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
THUMB = (1, 2, 3, 4)

# Tuned by grid search on half of a 3,888-hand HaGRID sample; 83% exact count on the other half.
STRAIGHT_DEG = 70.0   # max total bend at PIP + DIP for a finger to count as up
THUMB_DEG = 100.0     # max bend at MCP + IP for the thumb (loose: reach decides)
THUMB_REACH = 1.1     # thumb tip distance from THUMB_REF, relative to palm width


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _angle(u, v):
    nu = math.sqrt(sum(x * x for x in u)) or 1e-9
    nv = math.sqrt(sum(x * x for x in v)) or 1e-9
    c = max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v)) / (nu * nv)))
    return math.degrees(math.acos(c))


def _dist(a, b):
    return math.dist(a, b)


THUMB_REF = 13        # landmark the thumb tip must reach away from (13 = ring-finger MCP)
PINCH = 0.75          # thumb tip–index tip distance, relative to palm width, that counts as a pinch


def fingers_up(p):
    """p: 21 (x, y, z) world landmarks. Returns {finger: bool} for thumb..pinky."""
    out = {}
    wrist = p[0]
    palm = _dist(p[5], p[17])
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        bend = _angle(_sub(p[pip], p[mcp]), _sub(p[dip], p[pip])) + _angle(_sub(p[dip], p[pip]), _sub(p[tip], p[dip]))
        out[name] = bend < STRAIGHT_DEG and _dist(p[tip], wrist) > _dist(p[pip], wrist)
    cmc, mcp, ip, tip = THUMB
    bend = _angle(_sub(p[mcp], p[cmc]), _sub(p[ip], p[mcp])) + _angle(_sub(p[ip], p[mcp]), _sub(p[tip], p[ip]))
    thumb = bend < THUMB_DEG and _dist(p[tip], p[THUMB_REF]) > THUMB_REACH * palm
    # Thumb and index tips touching ("OK", pinch): neither is a raised finger.
    if _dist(p[4], p[8]) < PINCH * palm:
        thumb = False
        out["index"] = False
    return {"thumb": thumb, **out}


# Fingers up per HaGRID gesture (no_gesture and ambiguous ones are skipped).
EXPECTED = {
    "fist": 0, "one": 1, "mute": 1, "like": 1, "dislike": 1,
    "peace": 2, "peace_inverted": 2, "two_up": 2, "two_up_inverted": 2, "call": 2,
    "three": 3, "three2": 3, "ok": 3, "four": 4,
    "palm": 5, "stop": 5, "stop_inverted": 5,
}


def load_hagrid():
    """Yields (image_path, gesture, leading_hand, bbox[x,y,w,h] normalized)."""
    ann_dir = next(HAGRID.rglob("ann_train_val"), None)
    if ann_dir is None:
        raise SystemExit("HaGRID annotations not found")
    images = {p.stem: p for p in HAGRID.rglob("*.jpg")}
    for ann_file in sorted(ann_dir.glob("*.json")):
        gesture_dir = ann_file.stem
        for image_id, a in json.loads(ann_file.read_text()).items():
            if image_id not in images:
                continue
            for bbox, label in zip(a["bboxes"], a["labels"]):
                if label == gesture_dir:
                    yield images[image_id], label, a.get("leading_hand"), bbox


def hand_crop(img, bbox, out=256):
    """Square crop around the hand box, 2.2x its size, so the hand is webcam-sized."""
    from PIL import Image

    w, h = img.size
    cx, cy = (bbox[0] + bbox[2] / 2) * w, (bbox[1] + bbox[3] / 2) * h
    size = max(bbox[2] * w, bbox[3] * h) * 2.2
    box = (int(cx - size / 2), int(cy - size / 2), int(cx + size / 2), int(cy + size / 2))
    return img.crop(box).resize((out, out), Image.Resampling.BILINEAR)


def main():
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "teacher/models/hand_landmarker.task")),
        num_hands=2))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    per_class = defaultdict(int)
    samples = []
    for path, gesture, lead, bbox in load_hagrid():
        if gesture not in EXPECTED or per_class[gesture] >= limit:
            continue
        per_class[gesture] += 1
        samples.append((path, gesture, lead, bbox))

    results = []
    for path, gesture, lead, bbox in samples:
        crop = hand_crop(Image.open(path).convert("RGB"), bbox)
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop)))
        if not r.hand_landmarks:
            results.append((gesture, None, lead, None))
            continue
        # The hand nearest the crop centre is the annotated one.
        i = min(range(len(r.hand_landmarks)),
                key=lambda k: sum((p.x - 0.5) ** 2 + (p.y - 0.5) ** 2 for p in r.hand_landmarks[k]))
        world = [(p.x, p.y, p.z) for p in r.hand_world_landmarks[i]]
        up = fingers_up(world)
        results.append((gesture, sum(up.values()), lead, r.handedness[i][0].category_name.lower()))

    found = [x for x in results if x[1] is not None]
    ok = sum(1 for g, n, _, _ in found if n == EXPECTED[g])
    print(f"detected {len(found)}/{len(results)}  count exact {ok}/{len(found)} = {ok / max(1, len(found)):.3f}")
    by = defaultdict(Counter)
    for g, n, _, _ in found:
        by[g][n] += 1
    for g in EXPECTED:
        if by[g]:
            tot = sum(by[g].values())
            print(f"  {g:16s} want {EXPECTED[g]}  acc {by[g][EXPECTED[g]] / tot:.2f}  {dict(sorted(by[g].items()))}")
    pairs = Counter((lead, h) for _, n, lead, h in found if lead in ("left", "right"))
    print("handedness (HaGRID leading_hand, MediaPipe label on the unmirrored photo):", dict(pairs))


if __name__ == "__main__":
    main()
