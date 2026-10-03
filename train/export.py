"""Export the trained student to ONNX for onnxruntime-web, plus the meta the page reads.

    python train/export.py   # train/runs/best.pt → apps/demo/public/models/student.{onnx,json}
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

from train import HEADS, MEAN, RUNS, SIZE, STD, Student

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "apps/demo/public/models"


class Wrapped(torch.nn.Module):
    """Named tuple outputs, in HEADS order, so ONNX output names match the head keys."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[k] for k in HEADS)


def halve_weights(path: Path):
    """Store float32 weights as float16 and Cast them back at load.

    The file is half the size, while every op still runs in float32, so the model behaves
    the same on WebGPU and on the WASM fallback (which lacks most float16 kernels).
    """
    model = onnx.load(str(path))
    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims) < 1024:
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
        init.CopyFrom(half)
        casts.append(helper.make_node("Cast", [init.name], [init.name[: -len("_fp16")]], to=onnx.TensorProto.FLOAT))
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save(model, str(path))


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

    halve_weights(path)
    got = ort.InferenceSession(str(path)).run(None, {"input": x.numpy()})
    for name, a, b in zip(HEADS, got, ref):
        p, q = torch.softmax(torch.from_numpy(a), -1), torch.softmax(b, -1)
        err = float((p - q).abs().max())
        assert err < 0.01, f"{name} after fp16 weights: max prob diff {err}"
        assert int(p.argmax()) == int(q.argmax()), f"{name}: fp16 weights changed the answer"

    labels = json.loads((ROOT / "labels.json").read_text())
    crop = {k: labels["crop"][k] for k in ("scale", "shiftY")}
    meta = {"inputSize": SIZE, "mean": list(MEAN), "std": list(STD), "heads": HEADS, "crop": crop}
    (OUT / "student.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and student.json")


if __name__ == "__main__":
    main()
