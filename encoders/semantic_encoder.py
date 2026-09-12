"""E_Semantic: open-vocabulary detection -> z_LLM.

Runs OWL-ViT open-vocabulary detection against a bank of candidate labels
(swap in Grounded-SAM/Detic later if needed) and return both:
  - a fixed-size embedding (pooled CLIP-style image/text features)
  - the actual detected {label, box, score} triples in `extras`
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

DEFAULT_MODEL = "google/owlvit-base-patch32"

# Minimal starter vocabulary for CARLA driving scenes. Expand/replace with a
# scene-specific ontology or swap in fully open-vocab captioning later.
DEFAULT_CANDIDATE_LABELS = [
    "car", "pedestrian", "cyclist", "traffic cone", "traffic light",
    "road sign", "plastic bag", "cardboard box", "debris", "pothole",
    "animal", "fallen tree branch", "construction barrier", "tire",
]


class OwlVitSemanticEncoder:
    name = "semantic"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        candidate_labels: Sequence[str] = DEFAULT_CANDIDATE_LABELS,
        score_threshold: float = 0.1,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.candidate_labels = list(candidate_labels)
        self.score_threshold = score_threshold

    @torch.no_grad()
    def _detect(self, frame: Image.Image):
        inputs = self.processor(
            text=[self.candidate_labels], images=frame, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([frame.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.score_threshold
        )[0]
        detections = [
            {
                "label": self.candidate_labels[label_idx],
                "score": float(score),
                "box": [float(x) for x in box],
            }
            for label_idx, score, box in zip(
                results["labels"].tolist(), results["scores"].tolist(), results["boxes"].tolist()
            )
        ]
        # outputs.image_embeds is a per-patch grid (1, H, W, D), not a pooled
        # vector -- mean-pool over the spatial grid to get a fixed-size
        # embedding for the numeric OOD scorer.
        patch_embeds = outputs.image_embeds.squeeze(0).detach().cpu().numpy()
        image_embeds = patch_embeds.reshape(-1, patch_embeds.shape[-1]).mean(axis=0)
        return detections, image_embeds

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
