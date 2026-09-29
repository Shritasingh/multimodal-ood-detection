"""E_Semantic: open-vocabulary detection -> z_LLM.

Runs OWL-ViT open-vocabulary detection against a bank of candidate labels
(swap in Grounded-SAM/Detic later if needed), then discards the boxes --
the numeric embedding is a score-weighted mean of text embeddings of the
labels actually detected in a frame (BERT by default; any preset from
encoders/text_embedder.py via text_embedder=...), not a pooled vision feature. This
makes the embedding space directly about scene semantics ("what's in the
frame") rather than pixel statistics, so cosine similarity compares label
content instead of visual appearance. The detected {label, box, score}
triples are still returned in `extras` for inspection/visualization.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from encoders.text_embedder import TextEmbedder

DEFAULT_MODEL = "google/owlvit-base-patch32"
DEFAULT_TEXT_EMBEDDER = "bert"  # the original embedder; kept as default so existing results stay comparable

# CARLA's own CityObjectLabel semantic-segmentation taxonomy (every tag
# except the non-visual catch-alls Any/NONE/Other/Dynamic/Static), used
# as-is rather than trimmed to one map/pipeline, so the vocab stays valid
# if the town or spawn config changes later.
NOMINAL_LABELS = [
    "road", "road line", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
    "bridge", "rail track", "guard rail", "train", "water", "ground",
]

# Objects sim/anomalies.py's NOVEL_OBJECT_BLUEPRINTS can inject (traffic
# cone/plastic bag/cardboard box/mattress), plus other debris/hazard
# concepts that shouldn't appear in a nominal CARLA scene.
ANOMALY_LABELS = [
    "traffic cone", "plastic bag", "cardboard box", "mattress", "debris",
    "pothole", "animal", "fallen tree branch", "construction barrier", "tire",
]

DEFAULT_CANDIDATE_LABELS = NOMINAL_LABELS + ANOMALY_LABELS


class OwlVitSemanticEncoder:
    name = "semantic"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        text_embedder: str | TextEmbedder = DEFAULT_TEXT_EMBEDDER,
        text_presets: dict | None = None,
        candidate_labels: Sequence[str] = DEFAULT_CANDIDATE_LABELS,
        min_score: float = 0.025,
        top_n: int = 5,
        score_threshold: float | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.candidate_labels = list(candidate_labels)
        self.min_score = min_score
        self.top_n = top_n
        self.score_threshold = score_threshold

        # a preset name (bert / minilm / clip / text_presets entry) or an already-loaded embedder to share
        self.text = text_embedder if isinstance(text_embedder, TextEmbedder) else TextEmbedder.from_preset(text_embedder, text_presets, device=self.device)

        # Label vocabulary is fixed, so embed it once rather than per frame.
        label_embeds = self.text.embed(self.candidate_labels)
        self.label_embeddings = dict(zip(self.candidate_labels, label_embeds))
        self.embedding_dim = label_embeds.shape[-1]
        # Frames with zero detections above threshold get this embedding.
        # A zero vector would break cosine similarity (division by a
        # zero norm), so use the text embedding of an explicit "empty
        # scene" phrase instead -- a real, normalizable direction that
        # nominal empty-road frames should cluster around.
        self._empty_embedding = self.text.embed(["no objects detected"])[0]

    def _labels_to_embedding(self, detections: list[dict]) -> np.ndarray:
        """Score-weighted mean of the (precomputed) text embeddings of the
        labels detected in one frame. Boxes are intentionally ignored."""
        if not detections:
            return self._empty_embedding
        vecs = np.stack([self.label_embeddings[d["label"]] for d in detections])
        weights = np.array([d["score"] for d in detections])
        return np.average(vecs, axis=0, weights=weights)

    @torch.no_grad()
    def _forward(self, frame: Image.Image):
        inputs = self.processor(text=[self.candidate_labels], images=frame, return_tensors="pt").to(self.device)
        return self.model(**inputs)

    @torch.no_grad()
    def label_scores(self, frame: Image.Image | None = None, outputs=None) -> np.ndarray:
        """Best-box sigmoid score for every candidate label, with no top_n / min_score cut: the frame's
        raw per-label evidence, normalised into a distribution over labels by the semantic_dist scorer."""
        outputs = outputs if outputs is not None else self._forward(frame)
        return torch.sigmoid(outputs.logits[0]).max(dim=0).values.float().cpu().numpy()

    @torch.no_grad()
    def _detect(self, frame: Image.Image, outputs=None):
        """Detections for one frame: per-box thresholding if score_threshold is set (the original behaviour), else the top_n labels reaching min_score."""
        outputs = outputs if outputs is not None else self._forward(frame)
        detections = self._thresholded(outputs, frame) if self.score_threshold is not None else self._top_labels(outputs, frame)
        return detections, self._labels_to_embedding(detections)

    def _top_labels(self, outputs, frame: Image.Image) -> list[dict]:
        """The top_n labels by best-box score that reach min_score, each with the box that scored it."""
        best, best_box = torch.sigmoid(outputs.logits[0]).max(dim=0)
        w, h = frame.size
        detections = []
        for label_idx in best.argsort(descending=True)[: self.top_n].tolist():
            if best[label_idx] < self.min_score:
                break
            cx, cy, bw, bh = outputs.pred_boxes[0, best_box[label_idx]].tolist()
            detections.append({
                "label": self.candidate_labels[label_idx],
                "score": float(best[label_idx]),
                "box": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
            })
        return detections

    def _thresholded(self, outputs, frame: Image.Image) -> list[dict]:
        """Every box's best label with score above score_threshold."""
        target_sizes = torch.tensor([frame.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.score_threshold
        )[0]
        return [
            {"label": self.candidate_labels[i], "score": float(sc), "box": [float(x) for x in box]}
            for i, sc, box in zip(results["labels"].tolist(), results["scores"].tolist(), results["boxes"].tolist())
        ]

    def encode(self, obs_history: Sequence[Image.Image]) -> tuple[np.ndarray, dict[str, Any]]:
        """obs_history: list of PIL Images; we detect on the most recent frame
        (semantic anomalies like a novel object are typically evaluated
        per-frame rather than pooled over time) and also embed the whole
        window for the numeric score. Returns (embedding, extras)."""
        latest = obs_history[-1]
        detections, latest_embed = self._detect(latest)

        all_embeds = [latest_embed]
        for frame in obs_history[:-1]:
            _, emb = self._detect(frame)
            all_embeds.append(emb)
        pooled = np.mean(all_embeds, axis=0)

        return pooled, {
            "detections": detections,
            "candidate_labels": self.candidate_labels,
            "latest_frame_embedding": latest_embed,
        }
