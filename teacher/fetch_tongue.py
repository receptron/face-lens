"""Collect openly licensed tongue-out photos from the web as training crops.

1. Candidates: Wikimedia Commons "People sticking out the tongue" (recursive, minus
   topless / suggestive / art subcategories) and Openverse searches (CC-licensed).
2. Crop each with the browser's MediaPipe crop; images without a face drop out.
3. Bonsai checks each crop. tongue P >= YES → captures/tongue=yes/, P <= NO → captures/tongue=no/
   (hard negatives: faces from tongue searches with no tongue visible). Between: discarded.
   Its other answers (emotion, hair, eyes) go to data/teacher-web.jsonl for training.

Source, license and creator of every kept file are in data/web/sources.jsonl.

    python teacher/fetch_tongue.py
"""

import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import crop_face, make_landmarker  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
WEB = DATA / "web"
UA = "face-lens/0.1 (+https://github.com/receptron/face-lens)"
YES, NO = 0.7, 0.15
# Only permissive licenses (CC BY, CC0, public domain): the trained model is published under
# CC BY 4.0, so no non-commercial, no-derivatives or share-alike (BY-SA, GFDL) sources.
def license_ok(name):
    words = re.split(r"[- ]+", (name or "").strip().lower())
    if not words or words == [""]:
        return False
    if any(w in ("nc", "nd", "sa") for w in words) or words[0] == "gfdl":
        return False
    return words[:2] in (["cc", "by"], ["cc", "cc0"]) or words[0] == "cc0" \
        or words[:2] in (["public", "domain"], ["no", "restrictions"])


COMMONS_ROOT = "Category:People sticking out the tongue"
COMMONS_SKIP = re.compile(r"topless|suggestive|in art|nude|naked|sexual", re.I)
OPENVERSE_QUERIES = [
    "sticking out tongue", "tongue out", "tongue out selfie", "girl sticking out tongue",
    "boy sticking out tongue", "man sticking out tongue", "woman sticking out tongue",
    "child tongue out", "funny face tongue", "silly face tongue", "blowing raspberry face",
]


def get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(30 * (attempt + 1))
                continue
            raise


def commons_candidates():
    api = "https://commons.wikimedia.org/w/api.php?"
    seen_cats, queue, out = set(), [(COMMONS_ROOT, 0)], []
    while queue:
        cat, depth = queue.pop()
        if cat in seen_cats or COMMONS_SKIP.search(cat):
            continue
        seen_cats.add(cat)
        cont = {}
        while True:
            q = {"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": cat,
                 "gcmlimit": "200", "gcmtype": "file|subcat", "prop": "imageinfo",
                 "iiprop": "url|extmetadata|mime", "iiurlwidth": "1024", **cont}
            d = get_json(api + urllib.parse.urlencode(q))
            for p in d.get("query", {}).get("pages", {}).values():
                if p.get("ns") == 14:
                    if depth < 3:
                        queue.append((p["title"], depth + 1))
                    continue
                info = (p.get("imageinfo") or [{}])[0]
                if not info.get("mime", "").startswith("image/jpeg") and not info.get("mime", "").startswith("image/png"):
                    continue
                meta = info.get("extmetadata", {})
                out.append({
                    "source": "wikimedia-commons", "title": p["title"], "page": info.get("descriptionurl"),
                    "url": info.get("thumburl") or info.get("url"),
                    "license": meta.get("LicenseShortName", {}).get("value"),
                    "creator": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:200],
                })
            if "continue" not in d:
                break
            cont = d["continue"]
            time.sleep(0.5)
    print(f"commons: {len(out)} files from {len(seen_cats)} categories", flush=True)
    return out


def openverse_candidates():
    out = []
    for query in OPENVERSE_QUERIES:
        for page in range(1, 13):
            q = {"q": query, "page": page, "page_size": 20, "license_type": "all-cc"}
            try:
                d = get_json("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(q))
            except urllib.error.HTTPError as e:
                print(f"openverse {query!r} p{page}: HTTP {e.code}", flush=True)
                break
            for r in d.get("results", []):
                out.append({
                    "source": f"openverse/{r.get('source')}", "title": r.get("title"),
                    "page": r.get("foreign_landing_url"), "url": r.get("url"),
                    "license": f"cc-{r.get('license')} {r.get('license_version') or ''}".strip(),
                    "creator": r.get("creator"),
                })
            if page >= d.get("page_count", 0):
                break
            time.sleep(3)
    print(f"openverse: {len(out)} results", flush=True)
    return out


def download(url: str):
    """Polite fetch: Wikimedia's upload servers ask for ~1 request/s and answer 429 otherwise."""
    for attempt in range(4):
        if "wikimedia.org" in url:
            time.sleep(1.2)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                img = Image.open(io.BytesIO(r.read()))
                img.load()
                return img
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(60 * (attempt + 1))
                continue
            print(f"skip {url[:80]}: HTTP {e.code}", flush=True)
            return None
        except Exception as e:  # noqa: BLE001 — a dead link is just skipped
            print(f"skip {url[:80]}: {e}", flush=True)
            return None
    return None


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    cand_path = WEB / "candidates.json"
    if cand_path.exists():
        candidates = json.loads(cand_path.read_text())
    else:
        candidates = commons_candidates() + openverse_candidates()
        uniq = {c["url"]: c for c in candidates if c.get("url")}
        candidates = list(uniq.values())
        cand_path.write_text(json.dumps(candidates, indent=1))
    print(f"{len(candidates)} unique candidates", flush=True)

    # Download + crop.
    landmarker = make_landmarker()
    crops_dir = WEB / "crops"
    crops_dir.mkdir(exist_ok=True)
    cropped = []
    for i, c in enumerate(candidates):
        key = hashlib.sha1(c["url"].encode()).hexdigest()[:16]
        dest = crops_dir / f"{key}.jpg"
        if dest.exists():
            cropped.append((key, c))
            continue
        img = download(c["url"])
        if img is None:
            continue
        if min(img.size) < 160:
            continue
        crop = crop_face(landmarker, img)
        if crop is None:
            continue
        crop.save(dest, quality=92)
        cropped.append((key, c))
        if (i + 1) % 100 == 0:
            print(f"downloaded {i + 1}/{len(candidates)}, faces {len(cropped)}", flush=True)
    print(f"{len(cropped)} face crops", flush=True)

    # Verify with Bonsai.
    from bonsai import Teacher

    teacher = Teacher()
    verdicts = {}
    vpath = WEB / "verify.jsonl"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            r = json.loads(line)
            verdicts[r["key"]] = r["probs"]
    with vpath.open("a") as vf:
        for key, c in cropped:
            if key in verdicts:
                continue
            probs = teacher.label(Image.open(crops_dir / f"{key}.jpg"))
            verdicts[key] = probs
            vf.write(json.dumps({"key": key, "probs": probs}) + "\n")
            vf.flush()

    kept = {"yes": 0, "no": 0, "unsure": 0}
    sources = (WEB / "sources.jsonl").open("w")
    teacher_web = (DATA / "teacher-web.jsonl").open("w")
    for key, c in cropped:
        if not license_ok(c.get("license")):
            kept["license"] = kept.get("license", 0) + 1
            continue
        p_yes = verdicts[key]["tongue"][1]
        tag = "yes" if p_yes >= YES else "no" if p_yes <= NO else None
        if tag is None:
            kept["unsure"] += 1
            continue
        kept[tag] += 1
        tag_dir = DATA / "captures" / f"tongue={tag}"
        tag_dir.mkdir(parents=True, exist_ok=True)
        name = f"web-{key}.jpg"
        (tag_dir / name).write_bytes((crops_dir / f"{key}.jpg").read_bytes())
        rel = f"captures/tongue={tag}/{name}"
        sources.write(json.dumps({"path": rel, "p_tongue": round(p_yes, 3), **c}) + "\n")
        teacher_web.write(json.dumps({"path": rel, "probs": verdicts[key]}) + "\n")
    print("kept", kept, flush=True)


if __name__ == "__main__":
    main()
