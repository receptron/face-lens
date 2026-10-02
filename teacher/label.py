"""Run the Bonsai teacher over every crop in data/manifest.jsonl.

Resumable: soft labels are appended to data/teacher.jsonl as they are made, and
crops already in it are skipped. Validation crops go first so a partial run is
already usable for evaluation.

    python teacher/label.py            # everything
    python teacher/label.py --limit 500
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from bonsai import Teacher

DATA = Path(__file__).resolve().parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text().splitlines() if l]
    out_path = DATA / "teacher.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["path"] for l in out_path.read_text().splitlines() if l}
    todo = [r for r in rows if r["path"] not in done]
    todo.sort(key=lambda r: (r["split"] != "val", r["source"] != "capture"))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} labeled, {len(todo)} to go", flush=True)

    teacher = Teacher()
    start = time.time()
    with out_path.open("a") as out:
        for i, row in enumerate(todo, 1):
            probs = teacher.label(Image.open(DATA / row["path"]))
            out.write(json.dumps({"path": row["path"], "probs": probs}) + "\n")
            out.flush()
            if i % 50 == 0:
                rate = (time.time() - start) / i
                print(f"{i}/{len(todo)}  {rate:.2f}s/img  eta {rate * (len(todo) - i) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
