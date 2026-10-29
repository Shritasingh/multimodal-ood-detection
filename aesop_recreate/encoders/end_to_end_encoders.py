"""End-to-end encoders: camera frame -> embedding directly, no concepts (AESOP's image baselines).

  clip_image    CLIP ViT-B/32 image embedding, L2-normalised (512-d)
  vision_patch  DINOv2-base patch tokens, multimodal-ood-detection's DinoPatchEncoder (16 x 21 patches x 768-d);
                scored with its patch bank (worst 5% patches, k = 1 cosine)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image


class ClipImageEncoder:
    name = "clip_image"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode_batch(self, frames: Sequence[Image.Image]) -> np.ndarray:
        px = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device)
        v = self.model.get_image_features(pixel_values=px)
        v = v if torch.is_tensor(v) else v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy()


def load_end_to_end(name: str):
    """clip_image or vision_patch, ready to encode_batch()."""
    if name == "clip_image":
        return ClipImageEncoder()
    from encoders.vision_encoder import DinoPatchEncoder

    return DinoPatchEncoder()


END_TO_END_ENCODERS = ("clip_image", "vision_patch")
