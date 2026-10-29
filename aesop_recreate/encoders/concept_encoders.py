"""Concept encoders: camera frame -> concepts ("object predicate" strings), the first stage of AESOP's pipeline.

  owl          OWL-ViT scores every vocabulary concept phrase ("bus stop on the sidewalk", sim/concepts.CONCEPTS)
               against every box; each box keeps its best concept, concepts reaching OWL_MIN_SCORE are kept (best
               score over boxes), at most OWL_MAX_CONCEPTS per frame, highest first
  qwen_closed  Qwen3-VL picks concepts from the vocabulary (QWEN_CLOSED_PROMPT)
  qwen_open    Qwen3-VL names objects and where they are freely (QWEN_OPEN_PROMPT)
The ground-truth concepts come from sim/concepts.py. The concepts then go through sim.concepts.describe() and a text
embedder (encoders/text_embedders.py).
"""
from __future__ import annotations

import re
from typing import Sequence

import torch
from PIL import Image

from sim.concepts import CONCEPTS, OBJECTS, PREDICATES

OWL_MODEL = "google/owlvit-base-patch32"
OWL_MIN_SCORE = 0.05
OWL_MAX_CONCEPTS = 15
QWEN_MAX_CONCEPTS = 15
QWEN_MAX_NEW_TOKENS = 200

# both prompts ask Qwen to enumerate only what it sees, each object-and-place once, and to stop there: the cap is a
# ceiling, not a target (an earlier wording made it pad every answer to 15 lines with repeats)
QWEN_CLOSED_PROMPT = (
    "You are labelling the front camera of a car driving through a town.\n"
    f"Objects: {', '.join(OBJECTS)}.\n"
    f"Locations: {', '.join(PREDICATES)}.\n"
    "Enumerate the objects you can actually see in the image, each as 'object location' using only the words above. "
    "Write one numbered line per object and location, list each combination once, and stop when you have listed what "
    f"you see: a simple scene needs only a few labels, and never write more than {QWEN_MAX_CONCEPTS}. No other text."
)
QWEN_OPEN_PROMPT = (
    "You are labelling the front camera of a car driving through a town.\n"
    "Enumerate the objects you can actually see in the image, each with where it is, as a short 'object location' "
    "phrase (for example 'car on the road'). Write one numbered line per object, list each once, and stop when you "
    f"have listed what you see: a simple scene needs only a few labels, and never write more than {QWEN_MAX_CONCEPTS}. "
    "No other text."
)


def _clean(line: str) -> str:
    line = re.sub(r"^[\s\-\*\d\.\)]+", "", line.lower())          # list markers: "1. ", "- "
    return re.sub(r"[^a-z0-9 \-]", "", line.replace("_", " ")).strip()


def parse_closed(text: str) -> list[str]:
    """Vocabulary concepts in a closed-prompt answer; lines that are not object + predicate from the lists are dropped."""
    out = []
    for line in text.splitlines():
        c = _clean(line)
        obj = max((o for o in OBJECTS if c.startswith(o)), key=len, default=None)
        pred = next((p for p in PREDICATES if c.endswith(p)), None)
        if obj and pred and f"{obj} {pred}" not in out:
            out.append(f"{obj} {pred}")
    return out[:QWEN_MAX_CONCEPTS]


def parse_open(text: str) -> list[str]:
    """Free 'object location' phrases of an open-prompt answer, de-duplicated, at most 8 words each."""
    out = []
    for line in re.split(r"[\n,;]", text):
        c = _clean(line)
        if c and len(c.split()) <= 8 and c not in out:
            out.append(c)
    return out[:QWEN_MAX_CONCEPTS]


class OwlConceptEncoder:
    """OWL-ViT over the closed concept vocabulary; the concept phrases are embedded once."""

    def __init__(self, model_name: str = OWL_MODEL, device: str | None = None):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        with torch.no_grad():
            tok = self.processor(text=CONCEPTS, return_tensors="pt", padding=True).to(self.device)
            q = self.model.owlvit.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            self.queries = q if torch.is_tensor(q) else q.pooler_output

    @torch.no_grad()
    def encode_batch(self, frames: Sequence[Image.Image]) -> list[list[dict]]:
        """Per frame: [{concept, object, predicate, score, box (x0, y0, x1, y1 pixels)}], best first."""
        px = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
        fmap, _ = self.model.image_embedder(pixel_values=px)
        b, h, w, d = fmap.shape
        feats = fmap.reshape(b, h * w, d)
        q = self.queries[None].expand(b, -1, -1)
        logits, _ = self.model.class_predictor(feats, q, torch.ones(q.shape[:2], dtype=torch.bool, device=self.device))
        boxes = self.model.box_predictor(feats, fmap)
        best, which = torch.sigmoid(logits).max(-1)                      # each box's best concept
        out = []
        for i, f in enumerate(frames):
            fw, fh = f.size
            keep: dict[int, tuple[float, int]] = {}
            for j in torch.nonzero(best[i] >= OWL_MIN_SCORE).flatten().tolist():
                c, s = int(which[i, j]), float(best[i, j])
                if s > keep.get(c, (0.0, -1))[0]:
                    keep[c] = (s, j)
            rows = []
            for c, (s, j) in sorted(keep.items(), key=lambda kv: -kv[1][0])[:OWL_MAX_CONCEPTS]:
                cx, cy, bw, bh = boxes[i, j].tolist()
                obj, pred = next((o, p) for o in OBJECTS for p in PREDICATES if f"{o} {p}" == CONCEPTS[c])
                rows.append({"concept": CONCEPTS[c], "object": obj, "predicate": pred, "score": round(s, 4),
                             "box": [round((cx - bw / 2) * fw, 1), round((cy - bh / 2) * fh, 1), round((cx + bw / 2) * fw, 1), round((cy + bh / 2) * fh, 1)]})
            out.append(rows)
        return out


class QwenConceptEncoder:
    """Qwen3-VL (multimodal-ood-detection's QwenVLEncoder, greedy and batched) with the closed and open concept prompts."""

    def __init__(self, **kwargs):
        from encoders.qwen_encoder import QwenVLEncoder

        self.vlm = QwenVLEncoder(**kwargs)

    def encode_batch(self, frames: Sequence[Image.Image]) -> list[dict]:
        """Per frame: {"closed": concepts, "open": phrases, "raw_closed": answer, "raw_open": answer}."""
        closed, _, _ = self.vlm.generate(frames, QWEN_CLOSED_PROMPT, QWEN_MAX_NEW_TOKENS)
        opened, _, _ = self.vlm.generate(frames, QWEN_OPEN_PROMPT, QWEN_MAX_NEW_TOKENS)
        return [{"closed": parse_closed(c), "open": parse_open(o), "raw_closed": c, "raw_open": o} for c, o in zip(closed, opened)]


CONCEPT_ENCODERS = {"owl": OwlConceptEncoder, "qwen": QwenConceptEncoder}
