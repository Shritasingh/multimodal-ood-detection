"""E_Vision: DINOv2 features.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL = "facebook/dinov2-base"


class DinoVisionEncoder:
    name = "vision"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def _embed_frames(self, frames: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # CLS token pooled representation per frame.
        cls = outputs.last_hidden_state[:, 0, :]
        return cls.detach().cpu().numpy()

    def encode(self, obs_history: Sequence[Image.Image]) -> tuple[np.ndarray, dict[str, Any]]:
        """obs_history: list of PIL Images (egocentric RGB frames, oldest->newest).
        Returns (embedding, extras)."""
        per_frame = self._embed_frames(obs_history)
        # Mean-pool over the temporal window; keep per-frame embeddings too
        # since a single flared frame can be diluted by mean pooling over a
        # long history.
        pooled = per_frame.mean(axis=0)
        return pooled, {"per_frame_embeddings": per_frame}
