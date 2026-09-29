"""Label text embedders shared by the semantic (OWL-ViT) and Qwen label encoders.

Presets (select in config under text_embedding.embedders; add or override under text_embedding.presets):
  bert   -- raw BERT, mean-pooled; the semantic encoder's original embedder (not trained for similarity)
  minilm -- sentence-transformer, contrastively trained so cosine distance means semantic difference
  clip   -- CLIP's text tower: trained against images, so labels of visually similar things sit close
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection

TEXT_EMBEDDERS = {
    "bert": {"model": "bert-base-uncased", "kind": "mean_pool"},
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2", "kind": "mean_pool"},
    "clip": {"model": "openai/clip-vit-base-patch32", "kind": "clip", "template": "a photo of a {}"},
}


class TextEmbedder:
    """L2-normalised label embeddings, cached per string. kind='mean_pool' averages the last hidden
    state over tokens (BERT, MiniLM); kind='clip' uses CLIP's projected text embedding, with the
    label wrapped in `template` (CLIP was trained on captions, not bare nouns)."""

    def __init__(self, model: str = TEXT_EMBEDDERS["minilm"]["model"], kind: str = "mean_pool", template: str = "{}", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kind, self.template, self.model_name = kind, template, model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        cls = CLIPTextModelWithProjection if kind == "clip" else AutoModel
        self.model = cls.from_pretrained(model).to(self.device).eval()
        self.dim = self.model.config.projection_dim if kind == "clip" else self.model.config.hidden_size
        self.cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_preset(cls, name: str, extra: dict | None = None, **kwargs) -> "TextEmbedder":
        presets = {**TEXT_EMBEDDERS, **(extra or {})}
        if name not in presets:
            raise ValueError(f"unknown text embedder {name!r}; known: {sorted(presets)}")
        return cls(**{**presets[name], **kwargs})

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            inp = self.tokenizer([self.template.format(t) for t in new], return_tensors="pt", padding=True, truncation=True).to(self.device)
            if self.kind == "clip":
                v = self.model(**inp).text_embeds
            else:
                h = self.model(**inp).last_hidden_state
                mask = inp["attention_mask"].unsqueeze(-1).float()
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v, dim=-1)
            self.cache.update(zip(new, v.float().cpu().numpy()))
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
