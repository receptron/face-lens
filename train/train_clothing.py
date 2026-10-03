"""Train and export the clothing-style student (style + pattern) from Bonsai soft labels.

Input: teacher/data/clothing/{manifest,teacher}.jsonl, crops from teacher/fetch_clothing.py.
Output: apps/demo/public/models/clothing.{onnx,json}.

    python train/train_clothing.py --epochs 15
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from export import halve_weights
from train import MEAN, STD, WebcamLook, soft_ce

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "teacher/data/clothing"
RUNS = ROOT / "train/runs"
OUT = ROOT / "apps/demo/public/models"
SPEC = json.loads((ROOT / "labels.json").read_text())["clothing"]
HEADS = {k: v["classes"] for k, v in SPEC["heads"].items()}
BACKBONE = "mobilenetv4_conv_small.e2400_r224_in1k"
SIZE = 224


class BottomCut:
    """A webcam often shows only the top of the torso: randomly black out the lower part."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            w, h = img.size
            keep = int(h * random.uniform(0.5, 1.0))
            img = img.copy()
            img.paste((0, 0, 0), (0, keep, w, h))
        return img


def transforms(train: bool):
    if not train:
        return T.Compose([T.Resize(SIZE), T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
    return T.Compose([
        BottomCut(),
        T.RandomResizedCrop(SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3),  # no hue/saturation: color is not a label here, but keep it honest
        WebcamLook(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def load_rows():
    teacher = {}
    for line in (DATA / "teacher.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            teacher[r["path"]] = r["probs"]
    rows = []
    for line in (DATA / "manifest.jsonl").read_text().splitlines():
        if line:
            r = json.loads(line)
            if r["path"] in teacher:
                rows.append({"path": r["path"], "split": r["split"],
                             "targets": {k: torch.tensor(v) for k, v in teacher[r["path"]].items() if k in HEADS}})
    return rows


class Crops(Dataset):
    def __init__(self, rows, train):
        self.rows, self.tf = rows, transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(Image.open(DATA / r["path"]).convert("RGB")), r["targets"]


class Student(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, len(c)) for k, c in HEADS.items()})

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return tuple(self.heads[k](f) for k in HEADS)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hits = {k: 0 for k in HEADS}
    n = 0
    for x, t in loader:
        out = model(x.to(device))
        for k, o in zip(HEADS, out):
            hits[k] += (o.argmax(-1).cpu() == t[k].argmax(-1)).sum().item()
        n += x.shape[0]
    return {k: round(hits[k] / max(1, n), 3) for k in HEADS}


def export(model):
    model = model.cpu().eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    path = OUT / "clothing.onnx"
    torch.onnx.export(model, (x,), str(path), input_names=["input"], output_names=list(HEADS),
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(x)
    halve_weights(path)
    got = ort.InferenceSession(str(path)).run(None, {"input": x.numpy()})
    for name, a, b in zip(HEADS, got, ref):
        assert int(np.argmax(a)) == int(b.argmax()), f"{name}: fp16 weights changed the answer"
    crop = {k: SPEC["crop"][k] for k in ("scale", "minVisible")}
    meta = {"inputSize": SIZE, "mean": list(MEAN), "std": list(STD), "heads": HEADS, "crop": crop}
    (OUT / "clothing.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and clothing.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    print(f"train {len(train_rows)}  val {len(val_rows)}", flush=True)

    # Inverse-sqrt class weights from the teacher's soft counts.
    weights = {}
    for k, classes in HEADS.items():
        counts = torch.full((len(classes),), 1.0) + sum(r["targets"][k] for r in train_rows)
        w = (counts.sum() / counts).sqrt()
        weights[k] = (w / w.mean()).clamp(0.25, 5.0).to(device)

    train_dl = DataLoader(Crops(train_rows, True), batch_size=64, shuffle=True, num_workers=6,
                          persistent_workers=True, drop_last=True)
    val_dl = DataLoader(Crops(val_rows, False), batch_size=128, num_workers=4)
    model = Student().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)
    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, t in train_dl:
            out = model(x.to(device))
            loss = sum(soft_ce(o, t[k].to(device), weights[k]).mean() for k, o in zip(HEADS, out))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {running / len(train_dl):.3f}  val {acc}  {time.time() - t0:.0f}s", flush=True)
        if sum(acc.values()) > best:
            best = sum(acc.values())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            (RUNS / "clothing.json").write_text(json.dumps({"epoch": epoch + 1, "val": acc}, indent=2))
    torch.save(best_state, RUNS / "clothing.pt")
    model.load_state_dict(best_state)
    export(model)


if __name__ == "__main__":
    main()
