"""Ternary Bonsai 2 27B as a teacher: soft labels for every learned attribute.

One prefill per image (the image plus all questions), then for each attribute
the prompt is extended by "<key>:" and the next-token distribution over the
option letters is read straight from the logits. No free text is generated.
The argmax letter is appended before moving on, so later answers see earlier
ones, exactly as if the model had written them.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "teacher/models/bonsai2-27b-mlx"
LABELS = json.loads((ROOT / "labels.json").read_text())
LETTERS = "ABCDEFGHIJ"

sys.path.insert(0, str(PACK / "runtime"))


# Attributes the teacher labels; gender, age and race come from FairFace's human labels.
KEYS = [k for k, spec in LABELS["learned"].items() if spec.get("source") == "teacher"]


def build_question() -> str:
    lines = [
        "Look at the person's face in the image that follows and answer every question.",
        "Judge only from what is visible. When unsure, pick the closest option.",
        "",
    ]
    for key in KEYS:
        spec = LABELS["learned"][key]
        opts = ", ".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(spec["classes"]))
        lines.append(f"{key}: {spec['question']} Options: {opts}")
    lines += ["", "Reply with one letter per line, in this exact format:"]
    lines += [f"{key}: <letter>" for key in KEYS]
    return "\n".join(lines)


def copy_cache(cache):
    """Fresh cache objects holding the same (immutable) arrays, so the prefix can be reused."""
    out = []
    for c in cache:
        n = type(c)(**({"size": len(c.state)} if hasattr(c, "cache") else {}))
        n.state = list(c.state) if hasattr(c, "cache") else c.state
        out.append(n)
    return out


class Teacher:
    """The question text comes before the image, so its prefill is done once and reused."""

    def __init__(self):
        from mlx_vlm.models import cache as cache_mod
        from vision_artifact import load_vl_model

        self.cache_mod = cache_mod
        self.model, self.processor, self.config = load_vl_model(PACK)
        self.tok = self.processor.tokenizer
        self.prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_question()},
                        {"type": "image"},
                        {"type": "text", "text": "Answer now."},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Each option letter must be exactly one token when it follows "key:".
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tok.encode(f" {letter}", add_special_tokens=False)
            assert len(ids) == 1, f"' {letter}' is {ids}"
            self.letter_ids.append(ids[0])
        self.prefix_cache = None
        self.prefix_len = None

    def _ids(self, text: str) -> mx.array:
        return mx.array([self.tok.encode(text, add_special_tokens=False)])

    def label(self, image: Image.Image) -> dict[str, list[float]]:
        from mlx_vlm.utils import prepare_inputs

        inputs = prepare_inputs(self.processor, images=[image.convert("RGB")], prompts=[self.prompt])
        ids = inputs["input_ids"]
        lm = self.model.language_model
        lm._position_ids = None
        lm._rope_deltas = None
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
        # Embeds the whole prompt (vision tower included) and sets the mRoPE positions for all of it.
        emb = self.model.get_input_embeddings(ids, inputs["pixel_values"], mask=inputs["attention_mask"], **extra)
        positions = lm._position_ids
        embeds = emb.inputs_embeds

        if self.prefix_cache is None:
            start = ids[0].tolist().index(self.config["vision_start_token_id"])
            self.prefix_len = start
            self.prefix_cache = self.cache_mod.make_prompt_cache(lm)
            lm(ids[:, :start], inputs_embeds=embeds[:, :start], cache=self.prefix_cache,
               position_ids=positions[..., :start])
            mx.eval([c.state for c in self.prefix_cache])
        p = self.prefix_len
        cache = copy_cache(self.prefix_cache)
        lm(ids[:, p:], inputs_embeds=embeds[:, p:], cache=cache, position_ids=positions[..., p:])

        out: dict[str, list[float]] = {}
        prefix = ""
        for key in KEYS:
            n = len(LABELS["learned"][key]["classes"])
            logits = lm(self._ids(f"{prefix}{key}:"), cache=cache).logits[0, -1]
            probs = mx.softmax(logits[mx.array(self.letter_ids[:n])].astype(mx.float32)).tolist()
            out[key] = [round(v, 4) for v in probs]
            best = max(range(n), key=lambda i: probs[i])
            # Feed the chosen letter so the next answer is conditioned on it.
            lm(self._ids(f" {LETTERS[best]}"), cache=cache)
            prefix = "\n"
        return out


def main():
    import time

    teacher = Teacher()
    for path in sys.argv[1:]:
        t = time.time()
        probs = teacher.label(Image.open(path))
        dt = time.time() - t
        print(f"{path}  ({dt:.2f}s)")
        for key, p in probs.items():
            classes = LABELS["learned"][key]["classes"]
            top = sorted(zip(classes, p), key=lambda x: -x[1])[:3]
            print(f"  {key:8s} " + "  ".join(f"{c} {v:.2f}" for c, v in top))


if __name__ == "__main__":
    main()
