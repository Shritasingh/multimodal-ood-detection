"""Concept encoders: camera frame -> concepts ("object predicate" strings), the first stage of AESOP's pipeline.

  owl          OWL-ViT, object first: every box (one per image patch) is scored against each vocabulary object's
               phrasings (OWL_QUERIES), each queried as is and as "photo of ..." (OWL_FORMS; caption-like queries score
               far higher than bare words, e.g. "pedestrian" ~0.01 vs "a photo of a person" ~0.45), the object's score
               being its best query; each box keeps its best object
               and that score; boxes overlapping a higher-scoring one by IoU > OWL_NMS_IOU are dropped
               (class-agnostic NMS: neighbouring patches predict near-duplicate boxes of one object); boxes reaching
               OWL_MIN_SCORE get the location whose "object predicate" phrase (sim/concepts.CONCEPTS) scores highest
               on them, and are kept, highest first, until OWL_MAX_CONCEPTS distinct concepts
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
OWL_MIN_SCORE = 0.15   # object-first scores run higher; 0.15 doubled recall at the same box precision on 240 frames
OWL_MAX_CONCEPTS = 15
OWL_NMS_IOU = 0.5
# query phrasings per vocabulary object, chosen by their OWL-ViT scores on the objects' boxes in the nominal runs (the
# object's score on a box is its best query); each phrasing is queried as is and as "photo of ..." (OWL_FORMS)
OWL_QUERIES = {
    "car": ["car"], "van": ["van"], "truck": ["truck"], "bus": ["bus", "minibus"],
    "motorcycle": ["motorcycle", "motorbike"], "bicycle": ["bicycle", "bike"],
    "pedestrian": ["pedestrian", "person walking"], "cyclist": ["cyclist", "person riding a bicycle"],
    "man": ["man"], "woman": ["woman"], "person": ["a person", "person"],
    "traffic light": ["traffic light", "traffic signal"], "speed limit sign": ["round road sign", "speed limit sign"],
    "pole": ["lamp pole", "metal pole"], "street lamp": ["street lamp", "lamp post", "street light"],
    "trash can": ["rubbish bin", "trash can"], "garbage bag": ["garbage bag", "trash bag"],
    "recycling bin": ["waste container", "recycling bin"], "mailbox": ["mail box", "post box"],
    "bus stop": ["bus stop with shelter", "bus stop shelter"], "cardboard box": ["cardboard box"],
    "bench": ["wooden bench", "bench"], "plant pot": ["planter box", "planter"], "fire hydrant": ["red hydrant", "fire hydrant"],
    "billboard": ["information board", "billboard"], "kiosk": ["booth", "kiosk"], "bike rack": ["bike rack", "bicycle rack"],
    "chair": ["chair"], "table": ["table"], "umbrella": ["umbrella", "parasol"], "fence": ["garden fence", "fence"],
    "wall": ["low garden wall", "garden wall"], "guard rail": ["crash barrier", "guard rail"],
    "building": ["house", "house front", "building"], "bridge": ["bridge"], "tree": ["tree"], "grass": ["grass", "lawn"],
    "water": ["water", "river"], "hay bale": ["hay bale", "straw bale"], "ambulance": ["ambulance"],
    "police car": ["police car"], "warning sign": ["warning sign", "warning triangle"]}
OWL_FORMS = ["{}", "photo of {}"]
OWL_MASS_NOUNS = {"grass", "lawn", "water"}   # no article after "photo of"


def owl_query_text(form: str, phrasing: str) -> str:
    """One OWL-ViT query: the phrasing in a form, with "a" / "an" after "photo of" for countable phrasings."""
    if form == "{}" or phrasing in OWL_MASS_NOUNS or phrasing.startswith("a "):
        return form.format(phrasing)
    return form.format(("an " if phrasing[0] in "aeiou" else "a ") + phrasing)


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
    """OWL-ViT over the closed vocabulary, object first; the object phrasings and concept phrases are embedded once."""

    def __init__(self, model_name: str = OWL_MODEL, device: str | None = None):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.queries = self._embed(CONCEPTS)                                     # "object predicate" phrases
        phrasing = [(o, owl_query_text(f, q)) for o in OBJECTS for q in OWL_QUERIES[o] for f in OWL_FORMS]
        self.object_queries = self._embed([q for _, q in phrasing])               # one per phrasing and form
        self.phrasing_object = torch.tensor([OBJECTS.index(o) for o, _ in phrasing], device=self.device)
        # CONCEPTS index of each (object, predicate)
        self.concept_index = torch.tensor([[CONCEPTS.index(f"{o} {p}") for p in PREDICATES] for o in OBJECTS], device=self.device)

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> torch.Tensor:
        tok = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        q = self.model.owlvit.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        return torch.nn.functional.normalize(q if torch.is_tensor(q) else q.pooler_output, dim=-1)

    def _scores(self, feats, queries):
        q = queries[None].expand(feats.shape[0], -1, -1)
        logits, _ = self.model.class_predictor(feats, q, torch.ones(q.shape[:2], dtype=torch.bool, device=self.device))
        return torch.sigmoid(logits)

    @torch.no_grad()
    def encode_batch(self, frames: Sequence[Image.Image]) -> list[list[dict]]:
        """Per frame: one row per kept box, [{concept, object, predicate, score, box (x0, y0, x1, y1 pixels)}], best first;
        a concept may have several boxes (two parked cars)."""
        from torchvision.ops import box_convert, nms

        px = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
        fmap, _ = self.model.image_embedder(pixel_values=px)
        b, h, w, d = fmap.shape
        feats = fmap.reshape(b, h * w, d)
        boxes = self.model.box_predictor(feats, fmap)
        per_phrasing = self._scores(feats, self.object_queries)
        per_object = torch.zeros(b, h * w, len(OBJECTS), device=self.device).scatter_reduce(
            2, self.phrasing_object.expand(b, h * w, -1), per_phrasing, "amax")
        best, which = per_object.max(-1)                                  # each box's best object
        concept_scores = self._scores(feats, self.queries)
        out = []
        for i, f in enumerate(frames):
            fw, fh = f.size
            cand = torch.nonzero(best[i] >= OWL_MIN_SCORE).flatten()
            xyxy = box_convert(boxes[i, cand], "cxcywh", "xyxy") * torch.tensor([fw, fh, fw, fh], device=self.device)
            rows, concepts = [], set()
            for k in nms(xyxy, best[i, cand], OWL_NMS_IOU).tolist():      # surviving boxes, highest score first
                j, o = int(cand[k]), int(which[i, cand[k]])
                obj, pred = OBJECTS[o], PREDICATES[int(concept_scores[i, j, self.concept_index[o]].argmax())]
                c = f"{obj} {pred}"
                if c not in concepts and len(concepts) == OWL_MAX_CONCEPTS:
                    continue
                concepts.add(c)
                rows.append({"concept": c, "object": obj, "predicate": pred, "score": round(float(best[i, j]), 4),
                             "box": [round(v, 1) for v in xyxy[k].tolist()]})
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
