"""Train the browser student: one MobileNetV4 backbone, one head per learned attribute.

Targets per crop and head, most trusted first:
  1. a capture tag (tongue=yes …)        → hard label
  2. FairFace human label (gender/age/race) → hard label
  3. Bonsai teacher soft label           → full distribution (distillation)
Heads with no target for a crop are masked out of the loss.

    python train/train.py --epochs 12
    python train/train.py --init train/runs/round1.pt --epochs 4   # fine-tune
    python train/export.py            # → public/models/student.{onnx,json}
"""

import argparse
import json
import random
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "teacher/data"
RUNS = ROOT / "train/runs"
LABELS = json.loads((ROOT / "labels.json").read_text())
HEADS = {k: v["classes"] for k, v in LABELS["learned"].items()}
BACKBONE = "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k"
SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_rows():
    teacher = {}
    path = DATA / "teacher.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line:
                r = json.loads(line)
                teacher[r["path"]] = r["probs"]
    rows = []
    for line in (DATA / "manifest.jsonl").read_text().splitlines():
        if not line:
            continue
        r = json.loads(line)
        targets = {}
        for key, classes in HEADS.items():
            if key in r["labels"]:
                t = torch.zeros(len(classes))
                t[r["labels"][key]] = 1.0
                targets[key] = t
            elif r["path"] in teacher and key in teacher[r["path"]]:
                targets[key] = torch.tensor(teacher[r["path"]][key])
        if targets:
            rows.append({"path": r["path"], "split": r["split"], "targets": targets, "source": r["source"]})
    return rows


class WebcamLook:
    """Blur and JPEG-ish softness so FairFace photos look more like a webcam."""

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        return img


def transforms(train: bool):
    if not train:
        return T.Compose([T.Resize(SIZE), T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])
    # No hue jitter: hair and eye color are labels.
    return T.Compose([
        T.RandomResizedCrop(SIZE, scale=(0.8, 1.0), ratio=(0.92, 1.08)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
        WebcamLook(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


class Crops(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        self.tf = transforms(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tf(Image.open(DATA / r["path"]).convert("RGB"))
        targets, mask = {}, {}
        for key, classes in HEADS.items():
            t = r["targets"].get(key)
            targets[key] = t if t is not None else torch.zeros(len(classes))
            mask[key] = torch.tensor(t is not None, dtype=torch.float32)
        return x, targets, mask


class Student(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=pretrained, num_classes=0)
        dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.drop = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({k: nn.Linear(dim, len(c)) for k, c in HEADS.items()})

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return {k: h(f) for k, h in self.heads.items()}


def soft_ce(logits, target, weight=None):
    loss = -(target * F.log_softmax(logits, -1))
    if weight is not None:
        loss = loss * weight
    return loss.sum(-1)


def class_weights(rows):
    """Inverse-sqrt frequency, so rare classes (tongue=yes, red hair) are not ignored."""
    weights = {}
    for key, classes in HEADS.items():
        counts = torch.full((len(classes),), 1.0)
        for r in rows:
            if key in r["targets"]:
                counts += r["targets"][key]
        w = counts.sum() / counts / len(classes)
        weights[key] = (w.sqrt() / w.sqrt().mean()).clamp(0.25, 6.0)
    return weights


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hits = {k: 0.0 for k in HEADS}
    total = {k: 0.0 for k in HEADS}
    for x, targets, mask in loader:
        out = model(x.to(device))
        for k in HEADS:
            m = mask[k].bool()
            if m.any():
                pred = out[k].argmax(-1).cpu()[m]
                ref = targets[k][m].argmax(-1)
                hits[k] += (pred == ref).sum().item()
                total[k] += m.sum().item()
    return {k: round(hits[k] / total[k], 3) if total[k] else None for k in HEADS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", help="start from this checkpoint (fine-tune) instead of ImageNet weights")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    coverage = {k: sum(k in r["targets"] for r in train_rows) for k in HEADS}
    print(f"train {len(train_rows)}  val {len(val_rows)}  per-head train coverage {coverage}", flush=True)

    weights = {k: w.to(device) for k, w in class_weights(train_rows).items()}
    train_dl = DataLoader(Crops(train_rows, True), batch_size=args.batch, shuffle=True, num_workers=6,
                          persistent_workers=True, drop_last=True)
    val_dl = DataLoader(Crops(val_rows, False), batch_size=128, num_workers=4)

    model = Student(pretrained=not args.init)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model = model.to(device)
    params = [
        {"params": model.backbone.parameters(), "lr": args.lr * 0.3},
        {"params": model.heads.parameters(), "lr": args.lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.02)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=steps, pct_start=0.1)

    RUNS.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for x, targets, mask in train_dl:
            out = model(x.to(device))
            loss = 0.0
            for k in HEADS:
                m = mask[k].to(device)
                if m.sum() == 0:
                    continue
                per = soft_ce(out[k], targets[k].to(device), weights[k])
                loss = loss + (per * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        acc = evaluate(model, val_dl, device)
        score = sum(v for v in acc.values() if v is not None)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {running / len(train_dl):.3f}  val {acc}  "
              f"{time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), RUNS / "best.pt")
            (RUNS / "best.json").write_text(json.dumps({"epoch": epoch + 1, "val": acc}, indent=2))
    print("best", (RUNS / "best.json").read_text())


if __name__ == "__main__":
    main()
