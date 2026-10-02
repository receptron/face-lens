"""Export the trained student to ONNX for onnxruntime-web, plus the meta the page reads.

    python train/export.py   # train/runs/best.pt → public/models/student.{onnx,json}
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from train import HEADS, MEAN, RUNS, SIZE, STD, Student

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public/models"


class Wrapped(torch.nn.Module):
    """Named tuple outputs, in HEADS order, so ONNX output names match the head keys."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[k] for k in HEADS)


def main():
    model = Student(pretrained=False)
    model.load_state_dict(torch.load(RUNS / "best.pt", map_location="cpu"))
    model.eval()
    wrapped = Wrapped(model).eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    path = OUT / "student.onnx"
    torch.onnx.export(wrapped, (x,), str(path), input_names=["input"], output_names=list(HEADS),
                      opset_version=17, dynamo=False)

    # The exported graph must agree with PyTorch before the page gets it.
    sess = ort.InferenceSession(str(path))
    got = sess.run(None, {"input": x.numpy()})
    with torch.no_grad():
        ref = wrapped(x)
    for name, a, b in zip(HEADS, got, ref):
        err = float(np.abs(a - b.numpy()).max())
        assert err < 1e-3, f"{name}: max diff {err}"

    meta = {"inputSize": SIZE, "mean": list(MEAN), "std": list(STD), "heads": HEADS}
    (OUT / "student.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and student.json")


if __name__ == "__main__":
    main()
