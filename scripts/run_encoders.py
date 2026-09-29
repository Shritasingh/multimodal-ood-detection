"""Config-driven encoding step: runs vision/semantic/physics over the
reference run and every anomaly run, and caches the raw per-frame (or, for
semantic, per-detected-label) embeddings to <out_path>/<encoder>*.npz.

Physics keeps a 2-frame window: its yaw-rate features (ego's and each
tracked other agent's) are finite differences and are undefined -- not just
"less accurate", literally zero/meaningless -- from a single timestep, so
frame 0 of every run has no physics embedding.

Usage:
    .venv/bin/python scripts/run_encoders.py [config/encoders_v2.json] [--anomaly-only]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
# optional first CLI arg: a different config file, e.g. config/encoders_qwen.json (run lists come from config/data.json)
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "config" / "encoders.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_metrics import active_encoders as expand_encoders, eval_runs, load_config
from encoders.qwen_encoder import JUDGE_PROMPT, LABELS_PROMPT, QwenVLEncoder, TextEmbedder, parse_judge, parse_label_list
from encoders.physics_encoder import GroundTruthPhysicsEncoder, RolePhysicsEncoder
from encoders.semantic_encoder import OwlVitSemanticEncoder
from encoders.vision_encoder import DinoPatchEncoder, DinoVisionEncoder


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_run(data_path: str):
    data_path = resolve_path(data_path)
    frame_paths = sorted((data_path / "frames").glob("*.jpg"))
    states = [json.loads(line) for line in (data_path / "states.jsonl").read_text().splitlines()]
    assert len(frame_paths) == len(states), (
        f"frame/state count mismatch in {data_path}: {len(frame_paths)} frames vs {len(states)} states"
    )
    return frame_paths, states


def encode_vision(frame_paths, encoder: DinoVisionEncoder, desc: str) -> tuple[np.ndarray, np.ndarray]:
    embeds = [encoder.encode([Image.open(p).convert("RGB")])[0] for p in tqdm(frame_paths, desc=desc)]
    return np.stack(embeds), np.arange(len(frame_paths))


def encode_physics(states, encoder: GroundTruthPhysicsEncoder, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """2-frame window [t-1, t] so accel/yaw-rate/drift are real finite
    differences, not identically zero -- frame 0 has no prior frame, so
    ticks start at 1."""
    embeds = []
    for t in tqdm(range(1, len(states)), desc=desc):
        embedding, _ = encoder.encode([states[t - 1], states[t]])
        embeds.append(embedding)
    return np.stack(embeds), np.arange(1, len(states))


def encode_vision_patch(frame_paths, encoder: DinoPatchEncoder, desc: str, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(T, n_patches, D) float16 patch tokens of every frame, cached without pooling."""
    out = []
    for i in tqdm(range(0, len(frame_paths), batch), desc=desc):
        out.append(encoder.encode_batch([Image.open(p).convert("RGB") for p in frame_paths[i : i + batch]]))
    return np.concatenate(out), np.arange(len(frame_paths))


def encode_physics_roles(states, encoder: RolePhysicsEncoder, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """Role-slot embeddings from a window of encoder.window states (3, or the configured history), so the
    first tick scored is window - 1."""
    w = encoder.window
    embeds = [encoder.encode(states[t - w + 1 : t + 1])[0] for t in tqdm(range(w - 1, len(states)), desc=desc)]
    return np.stack(embeds), np.arange(w - 1, len(states))


def detect_semantic(frame_paths, encoder: OwlVitSemanticEncoder, desc: str, with_scores: bool = False):
    """Every individually detected label, kept separate rather than pooled
    into one vector per frame -- a frame with 3 detections contributes 3
    rows, a frame with 0 contributes none. frame_idx says which frame each
    row came from, so per-frame scoring can be reconstructed later without
    re-running OWL-ViT. Labels are embedded afterwards, once per configured
    text embedder. with_scores also returns every frame's best-box score for
    every candidate label (T, n_labels), from the same forward pass."""
    frame_idx, labels, scores, label_scores = [], [], [], []
    for t, p in enumerate(tqdm(frame_paths, desc=desc)):
        img = Image.open(p).convert("RGB")
        outputs = encoder._forward(img)
        detections, _ = encoder._detect(img, outputs)
        if with_scores:
            label_scores.append(encoder.label_scores(outputs=outputs))
        for d in detections:
            frame_idx.append(t)
            labels.append(d["label"])
            scores.append(d["score"])
    return np.array(frame_idx), np.array(labels), np.array(scores), (np.stack(label_scores) if with_scores else None)


QWEN_ENCODERS = ("qwen_labels", "qwen_hidden", "qwen_judge_labels", "qwen_judge_score")


def label_rows(per_frame: list[tuple[int, list[str]]], embedder: TextEmbedder) -> dict:
    """One row per (frame, label), in the same layout as semantic.npz so run_metrics scores it the same way."""
    rows = [(t, l) for t, labels in per_frame for l in labels]
    labels = [l for _, l in rows]
    return {"embeddings": embedder.embed(labels), "frame_idx": np.array([t for t, _ in rows]), "labels": np.array(labels)}


def split_tag(name: str) -> tuple[str, str | None]:
    """'qwen_labels@clip' -> ('qwen_labels', 'clip'); untagged names -> (name, None)."""
    base, _, tag = name.partition("@")
    return base, tag or None


DEFAULT_TEXT_EMBEDDER = {"semantic": "bert"}  # everything else defaults to minilm


def text_embedder(models: dict, tag: str | None, text_cfg: dict, base: str) -> TextEmbedder:
    """The label embedder for an encoder variant (tag), else the first of text_embedding.embedders,
    else the encoder's default (bert for semantic, minilm for qwen). Each preset is loaded once and shared."""
    name = tag or (text_cfg.get("embedders") or [DEFAULT_TEXT_EMBEDDER.get(base, "minilm")])[0]
    cache = models.setdefault("text", {})
    if name not in cache:
        cache[name] = TextEmbedder.from_preset(name, text_cfg.get("presets"))
    return cache[name]


def saved_texts(out_dir: Path, suffix: str, want: set[str]) -> dict | None:
    """{tick: {prompt: output}} from an earlier run's qwen_texts<suffix>.jsonl, if it has every wanted prompt."""
    path = out_dir / f"qwen_texts{suffix}.jsonl"
    if not path.exists():
        return None
    by_tick: dict[int, dict] = {}
    for r in map(json.loads, path.read_text().splitlines()):
        by_tick.setdefault(r["tick"], {})[r["prompt"]] = r["output"]
    return by_tick if by_tick and all(want <= set(v) for v in by_tick.values()) else None


def encode_qwen(frame_paths, active: list[str], models: dict, p: dict, stride: int, out_dir: Path, suffix: str, desc: str) -> None:
    """Every stride-th frame: a labels pass (options 1 and 3 share it; hidden states are captured during
    that generation) and/or a judge pass (option 2). Raw model outputs go to qwen_texts<suffix>.jsonl.
    Label encoders may carry an embedder tag (qwen_labels@clip). With reuse_texts, label/score encoders
    are rebuilt from the saved answers instead of re-running Qwen (not possible for qwen_hidden)."""
    bases = {split_tag(a)[0] for a in active}
    want_labels = bool(bases & {"qwen_labels", "qwen_hidden"})
    want_judge = bool(bases & {"qwen_judge_labels", "qwen_judge_score"})
    want = {k for k, on in (("labels", want_labels), ("judge", want_judge)) if on}
    reuse = saved_texts(out_dir, suffix, want) if p.get("reuse_texts") and "qwen_hidden" not in bases else None
    labels, judge, scores, last, ans, log = [], [], [], [], [], []
    if reuse is not None:
        ticks = sorted(reuse)
        for t in ticks:
            if want_labels:
                labels.append((t, parse_label_list(reuse[t]["labels"])))
            if want_judge:
                lab, sc = parse_judge(reuse[t]["judge"])
                judge.append((t, lab))
                scores.append(sc)
        _save_qwen(active, models, p, np.array(ticks), labels, judge, scores, last, ans, out_dir, suffix)
        print(f"  qwen{suffix}: {len(ticks)} frames re-parsed from saved answers -> {out_dir}")
        return

    if "qwen" not in models:
        models["qwen"] = QwenVLEncoder(**{k: p[k] for k in ("model_name", "layers") if k in p})
    enc = models["qwen"]
    ticks = list(range(0, len(frame_paths), stride))
    bs = p.get("batch_size", 6)
    for i in tqdm(range(0, len(ticks), bs), desc=desc):
        tb = ticks[i : i + bs]
        frames = [Image.open(frame_paths[t]).convert("RGB") for t in tb]
        if want_labels:
            texts, lp, am = enc.generate(frames, LABELS_PROMPT, p.get("max_new_tokens_labels", 96), hidden="qwen_hidden" in bases)
            labels += [(t, parse_label_list(x)) for t, x in zip(tb, texts)]
            log += [{"tick": t, "prompt": "labels", "output": x} for t, x in zip(tb, texts)]
            if lp is not None:
                last.append(lp)
                ans.append(am)
        if want_judge:
            texts, _, _ = enc.generate(frames, JUDGE_PROMPT, p.get("max_new_tokens_judge", 160))
            for t, x in zip(tb, texts):
                lab, sc = parse_judge(x)
                judge.append((t, lab))
                scores.append(sc)
            log += [{"tick": t, "prompt": "judge", "output": x} for t, x in zip(tb, texts)]

    _save_qwen(active, models, p, np.array(ticks), labels, judge, scores, last, ans, out_dir, suffix)
    with open(out_dir / f"qwen_texts{suffix}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in log)
    print(f"  qwen{suffix}: {len(ticks)} frames (stride {stride}) -> {out_dir}")


def _save_qwen(active, models, p, ticks, labels, judge, scores, last, ans, out_dir: Path, suffix: str) -> None:
    """One npz per active qwen encoder; label encoders are embedded with their tagged (or default) embedder."""
    for name in active:
        base, tag = split_tag(name)
        if base in ("qwen_labels", "qwen_judge_labels"):
            emb = text_embedder(models, tag, p.get("text_embedding", {}), base)
            rows = label_rows(labels if base == "qwen_labels" else judge, emb)
            np.savez(out_dir / f"{name}{suffix}.npz", **rows, ticks=ticks, text_embedder=np.array([emb.model_name]))
        elif base == "qwen_hidden" and last:
            np.savez(out_dir / f"{name}{suffix}.npz", last_prompt=np.concatenate(last), answer_mean=np.concatenate(ans),
                     layers=np.array(models["qwen"].layers), ticks=ticks)
        elif base == "qwen_judge_score":
            np.savez(out_dir / f"{name}{suffix}.npz", embeddings=np.array(scores)[:, None], ticks=ticks)


def encode_run(run_name: str, data_path: str, out_path: str, active_encoders: list[str], models: dict, suffix: str,
               qwen_params: dict | None = None, text_cfg: dict | None = None) -> None:
    """suffix: '_nominal' for the reference run's feature pool, '' for an
    anomaly run's own raw embeddings."""
    print(f"=== {run_name} ===")
    frame_paths, states = load_run(data_path)
    out_dir = resolve_path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "vision" in active_encoders:
        embeddings, ticks = encode_vision(frame_paths, models["vision"], f"{run_name}/vision")
        out_file = out_dir / f"vision{suffix}.npz"
        np.savez(out_file, embeddings=embeddings, ticks=ticks)
        print(f"  vision{suffix}: {embeddings.shape} -> {out_file}")

    if "vision_patch" in active_encoders:
        patches, ticks = encode_vision_patch(frame_paths, models["vision_patch"], f"{run_name}/vision_patch")
        out_file = out_dir / f"vision_patch{suffix}.npz"
        np.savez(out_file, embeddings=patches, ticks=ticks)
        print(f"  vision_patch{suffix}: {patches.shape} -> {out_file}")

    for name in ("physics_roles", "physics_hist"):  # physics_hist = role embedding + an H-state history window
        if name in active_encoders:
            embeddings, ticks = encode_physics_roles(states, models[name], f"{run_name}/{name}")
            out_file = out_dir / f"{name}{suffix}.npz"
            np.savez(out_file, embeddings=embeddings, ticks=ticks)
            print(f"  {name}{suffix}: {embeddings.shape} (window {models[name].window}) -> {out_file}")

    if "physics" in active_encoders:
        embeddings, ticks = encode_physics(states, models["physics"], f"{run_name}/physics")
        out_file = out_dir / f"physics{suffix}.npz"
        np.savez(out_file, embeddings=embeddings, ticks=ticks)
        print(f"  physics{suffix}: {embeddings.shape} -> {out_file}")

    qwen_active = [e for e in active_encoders if split_tag(e)[0] in QWEN_ENCODERS]
    if qwen_active:
        p = qwen_params or {}
        stride = p.get("stride_reference", 10) if suffix == "_nominal" else p.get("stride_eval", 5)
        encode_qwen(frame_paths, qwen_active, models, p, stride, out_dir, suffix, f"{run_name}/qwen")

    semantic = [e for e in active_encoders if split_tag(e)[0] == "semantic"]
    want_dist = "semantic_dist" in active_encoders
    if semantic or want_dist:
        frame_idx, labels, scores, label_scores = detect_semantic(frame_paths, models["semantic"], f"{run_name}/semantic", with_scores=want_dist)
        if want_dist:
            out_file = out_dir / f"semantic_dist{suffix}.npz"
            np.savez(out_file, embeddings=label_scores, ticks=np.arange(len(frame_paths)), label_names=np.array(models["semantic"].candidate_labels))
            print(f"  semantic_dist{suffix}: {label_scores.shape} -> {out_file}")
        for name in semantic:
            emb = text_embedder(models, split_tag(name)[1], text_cfg or {}, "semantic")
            embeddings = emb.embed(labels.tolist())
            out_file = out_dir / f"{name}{suffix}.npz"
            np.savez(out_file, embeddings=embeddings, frame_idx=frame_idx, labels=labels, scores=scores, text_embedder=np.array([emb.model_name]))
            print(f"  {name}{suffix}: {embeddings.shape} ({emb.model_name}) -> {out_file}")


def main():
    config = load_config(CONFIG_PATH)
    active_encoders = expand_encoders(config)

    models = {}
    if "vision" in active_encoders:
        models["vision"] = DinoVisionEncoder()
    if "vision_patch" in active_encoders:
        models["vision_patch"] = DinoPatchEncoder()
    if "physics_roles" in active_encoders:
        models["physics_roles"] = RolePhysicsEncoder(**config.get("physics_roles_encoder_params", {}))
    if "physics_hist" in active_encoders:
        models["physics_hist"] = RolePhysicsEncoder(**{"history": 10, **config.get("physics_hist_encoder_params", {})})
    text_cfg = config.get("text_embedding", {})
    if any(split_tag(e)[0] in ("semantic", "semantic_dist") for e in active_encoders):
        sp = config.get("semantic_encoder_params", {})
        # share the loaded embedder with the pipeline unless semantic_encoder_params names its own
        sp.setdefault("text_embedder", text_embedder(models, None, text_cfg, "semantic"))
        models["semantic"] = OwlVitSemanticEncoder(**sp, text_presets=text_cfg.get("presets"))
    if "physics" in active_encoders:
        models["physics"] = GroundTruthPhysicsEncoder()
    # Qwen and its text embedders load lazily, only when needed
    qwen_params = {**config.get("qwen_encoder_params", {}), "text_embedding": text_cfg}

    if "--anomaly-only" not in sys.argv:
        for ref in config.get("reference_runs") or [config["reference_run"]]:
            encode_run(ref["run_name"], ref["data_path"], ref["out_path"], active_encoders, models, suffix="_nominal", qwen_params=qwen_params, text_cfg=text_cfg)

    for run in eval_runs(config):
        encode_run(run["run_name"], run["data_path"], run["out_path"], active_encoders, models, suffix="", qwen_params=qwen_params, text_cfg=text_cfg)


if __name__ == "__main__":
    main()
