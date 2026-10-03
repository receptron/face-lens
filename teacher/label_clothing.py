"""Bonsai soft labels for the clothing crops (style + pattern), resumable.

    python teacher/label_clothing.py   # data/clothing/manifest.jsonl → data/clothing/teacher.jsonl
"""

import json
import time
from pathlib import Path

from PIL import Image

from bonsai import LABELS, Teacher

DATA = Path(__file__).resolve().parent / "data/clothing"
INTRO = ("The image shows the area just below a person's chin: their upper-body clothing. "
         "Answer every question about that clothing.")


def main():
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text().splitlines() if l]
    out_path = DATA / "teacher.jsonl"
    done = {json.loads(l)["path"] for l in out_path.read_text().splitlines() if l} if out_path.exists() else set()
    todo = sorted((r for r in rows if r["path"] not in done), key=lambda r: r["split"] != "val")
    print(f"{len(done)} labeled, {len(todo)} to go", flush=True)
    teacher = Teacher(LABELS["clothing"]["heads"], INTRO)
    start = time.time()
    with out_path.open("a") as out:
        for i, r in enumerate(todo, 1):
            probs = teacher.label(Image.open(DATA / r["path"]))
            out.write(json.dumps({"path": r["path"], "probs": probs}) + "\n")
            out.flush()
            if i % 50 == 0:
                rate = (time.time() - start) / i
                print(f"{i}/{len(todo)}  {rate:.2f}s/img  eta {rate * (len(todo) - i) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
