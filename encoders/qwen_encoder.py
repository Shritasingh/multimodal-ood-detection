"""E_Qwen: Qwen3-VL as a semantic encoder, three ways.

1. qwen_labels -- the model lists the object categories in a frame (open
   vocabulary, no candidate label set); each label is embedded with a text
   embedder (encoders/text_embedder.py) and scored by cosine distance to the nearest label Qwen
   produced on the nominal runs.
2. qwen_judge_labels / qwen_judge_score -- one JSON prompt returns both an
   object list (scored like 1) and the model's own 0-10 anomaly score.
3. qwen_hidden -- decoder hidden states captured during the labels pass
   (last prompt token and mean over the generated answer, at a few layers);
   scored by kNN in a nominal-fitted whitened PCA space.

Generation is greedy and batched (left padding), so reruns are deterministic.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from encoders.text_embedder import TEXT_EMBEDDERS, TextEmbedder  # noqa: F401  (re-exported for callers)

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
LABELS_PROMPT = (
    "List every distinct object category visible in this driving scene, "
    "as a comma-separated list of short noun phrases. No other text."
)
JUDGE_PROMPT = (
    "You are the perception monitor of a self-driving car; this is its front camera. "
    "Answer in JSON with keys: "
    "\"objects\" (every distinct object category visible, as short noun phrases), "
    "\"unusual_objects\" (objects that would not normally be on or near a road), "
    "\"anomaly_score\" (integer 0-10, how unusual this scene is for normal urban driving). "
    "No other text."
)

MAX_LABELS = 20  # guards against repetition loops ("road surface ... variation variation")
MAX_LABEL_WORDS = 5


def clean_labels(items: Sequence[str]) -> list[str]:
    """Lower-cased, stripped, de-duplicated labels in first-seen order, dropping empties and run-on phrases."""
    out: list[str] = []
    for x in items:
        x = re.sub(r"[^a-z0-9 \-]", "", str(x).lower().replace("_", " ")).strip()  # "picket_fence" -> "picket fence"
        if x and len(x.split()) <= MAX_LABEL_WORDS and x not in out:
            out.append(x)
    return out[:MAX_LABELS]


def parse_label_list(text: str) -> list[str]:
    return clean_labels(text.split(","))


def parse_judge(text: str) -> tuple[list[str], float]:
    """(objects + unusual_objects, anomaly_score / 10); NaN score if the JSON does not parse."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    labels = clean_labels(list(j.get("objects") or []) + list(j.get("unusual_objects") or []))
    score = j.get("anomaly_score")
    return labels, float(score) / 10 if isinstance(score, (int, float)) else float("nan")


class QwenVLEncoder:
    name = "qwen"

    def __init__(self, model_name: str = DEFAULT_MODEL, layers: Sequence[int] = (9, 18, 27, 36), device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16).to(self.device).eval()
        self.layers = list(layers)
        tok = self.processor.tokenizer
        self.stop_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
        gen_eos = self.model.generation_config.eos_token_id
        self.stop_ids |= set(gen_eos if isinstance(gen_eos, list) else [gen_eos])

    def _inputs(self, frames: Sequence[Image.Image], prompt: str):
        convs = [[{"role": "user", "content": [{"type": "image", "image": f}, {"type": "text", "text": prompt}]}] for f in frames]
        return self.processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(self.device)

    @torch.no_grad()
    def generate(self, frames: Sequence[Image.Image], prompt: str, max_new_tokens: int, hidden: bool = False):
        """Greedy answers for a batch of frames. With hidden=True also returns
        (last_prompt, answer_mean): (B, len(layers), D) float16 decoder hidden
        states at the last prompt position and averaged over the answer tokens."""
        inp = self._inputs(frames, prompt)
        n_in = inp["input_ids"].shape[1]
        out = self.model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False,
            output_hidden_states=hidden, return_dict_in_generate=True,
        )
        gen = out.sequences[:, n_in:]
        texts = [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]
        if not hidden:
            return texts, None, None

        hs = out.hidden_states  # step 0: prefill (layers+1, each (B, n_in, D)); step s>0: input = generated token s-1
        last_prompt = torch.stack([hs[0][l][:, -1] for l in self.layers], dim=1).float()
        # an answer token counts until (excluding) the first stop token
        is_stop = torch.zeros_like(gen, dtype=torch.bool)
        for i in self.stop_ids:
            is_stop |= gen == i
        valid = torch.cumsum(is_stop.int(), dim=1) == 0  # (B, n_gen)
        B, D = gen.shape[0], last_prompt.shape[-1]
        acc = torch.zeros(B, len(self.layers), D, device=last_prompt.device)
        n = torch.zeros(B, 1, 1, device=last_prompt.device)
        for s in range(1, len(hs)):
            m = valid[:, s - 1].float().view(B, 1, 1)
            acc += torch.stack([hs[s][l][:, -1] for l in self.layers], dim=1).float() * m
            n += m
        answer_mean = acc / n.clamp(min=1)
        return texts, last_prompt.half().cpu().numpy(), answer_mean.half().cpu().numpy()
