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
        hidden = outputs.last_hidden_state
        # CLS (global summary, DINO's own training target) concatenated with
        # the mean of all patch tokens (background/texture context CLS can
        # underweight) -- reported to beat CLS alone on retrieval/classification
        # benchmarks in the DINOv2 paper. index 0 is CLS, 1: are patch tokens
        # (this checkpoint has no register tokens to exclude).
        cls = hidden[:, 0, :]
        patch_mean = hidden[:, 1:, :].mean(dim=1)
        combined = torch.cat([cls, patch_mean], dim=-1)
        return combined.detach().cpu().numpy()

    def encode(self, obs_history: Sequence[Image.Image]) -> tuple[np.ndarray, dict[str, Any]]:
        """obs_history: list of PIL Images (egocentric RGB frames, oldest->newest).
        Returns (embedding, extras)."""
        per_frame = self._embed_frames(obs_history)
        # Mean-pool over the temporal window; keep per-frame embeddings too
        # since a single flared frame can be diluted by mean pooling over a
        # long history.
        pooled = per_frame.mean(axis=0)
        return pooled, {"per_frame_embeddings": per_frame}


class DinoPatchEncoder:
    """DINOv2 patch tokens (no CLS, no center crop): one embedding per patch, a 16x21 grid for 224x294 input."""
    name = "vision_patch"
    PATCH = 14

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, image_size: tuple[int, int] = (224, 294)):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        h, w = image_size
        assert h % self.PATCH == 0 and w % self.PATCH == 0, "image_size must be multiples of 14"
        self.grid = (h // self.PATCH, w // self.PATCH)
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, do_center_crop=False, size={"height": h, "width": w}
        )
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval().to(self.dtype)

    @torch.no_grad()
    def encode_batch(self, frames: Sequence[Image.Image]) -> np.ndarray:
        """(B, n_patches, D) float16 patch tokens, row-major over the grid."""
        px = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        hidden = self.model(pixel_values=px).last_hidden_state
        return hidden[:, 1:, :].cpu().numpy().astype(np.float16)

    def encode(self, obs_history: Sequence[Image.Image]) -> tuple[np.ndarray, dict[str, Any]]:
        """Patch embeddings (N, D) of the latest frame and the grid shape."""
        return self.encode_batch([obs_history[-1]])[0], {"grid": self.grid}
