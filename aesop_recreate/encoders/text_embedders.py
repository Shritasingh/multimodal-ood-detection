"""Text embedders: a frame's description (sim.concepts.describe()) -> one L2-normalised vector, AESOP's second stage.

  minilm      sentence-transformers/all-MiniLM-L6-v2, mean-pooled (384-d)
  mpnet       sentence-transformers/all-mpnet-base-v2, mean-pooled (768-d); AESOP's MPNet
  bert_large  bert-large-uncased, mean-pooled (1024-d)
  e5_mistral  intfloat/e5-mistral-7b-instruct in 8-bit (bitsandbytes, ~7.3 GB), last-token pooling (4096-d)
The mean-pooled ones are multimodal-ood-detection's TextEmbedder with these presets. Every embedder caches its
vectors per description, so repeated descriptions (common for ground-truth concepts) are embedded once.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

TEXT_PRESETS = {
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2", "kind": "mean_pool"},
    "mpnet": {"model": "sentence-transformers/all-mpnet-base-v2", "kind": "mean_pool"},
    "bert_large": {"model": "bert-large-uncased", "kind": "mean_pool"},
}
E5_MISTRAL = "intfloat/e5-mistral-7b-instruct"
E5_TASK = "Given a driving scene observed by a runtime monitor, retrieve scenes with the same objects and locations"
TEXT_EMBEDDERS = (*TEXT_PRESETS, "e5_mistral")


class E5MistralEmbedder:
    """e5-mistral-7b-instruct in 8-bit: last-token pooling of the instruction-prefixed text with EOS appended, as in
    the model card. Same embed() interface as TextEmbedder."""

    def __init__(self, model: str = E5_MISTRAL, task: str = E5_TASK, batch_size: int = 8):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        self.model_name, self.task, self.batch_size = model, task, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0}).eval()
        self.dim = self.model.config.hidden_size
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(new), self.batch_size):
            chunk = new[i : i + self.batch_size]
            ids = self.tokenizer([f"Instruct: {self.task}\nQuery: {t}" for t in chunk], max_length=511, truncation=True,
                                 add_special_tokens=True, padding=False)["input_ids"]
            batch = self.tokenizer.pad({"input_ids": [x + [self.tokenizer.eos_token_id] for x in ids]}, padding=True, return_tensors="pt").to(0)
            h = self.model(**batch).last_hidden_state
            last = batch["attention_mask"].sum(1) - 1                    # right padding: the EOS position
            v = torch.nn.functional.normalize(h[torch.arange(len(chunk)), last].float(), dim=-1)
            self.cache.update(zip(chunk, v.cpu().numpy()))
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


def load_text_embedder(name: str):
    """One of TEXT_EMBEDDERS, ready to embed()."""
    if name == "e5_mistral":
        return E5MistralEmbedder()
    from encoders.text_embedder import TextEmbedder

    return TextEmbedder.from_preset(name, TEXT_PRESETS)
