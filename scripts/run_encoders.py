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
# optional first CLI arg: a different config file, e.g. config/encoders_v2.json
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "config" / "encoders.json"

sys.path.insert(0, str(REPO_ROOT))
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
    """Role-slot embeddings from a 3-state window, so ticks start at 2."""
    w = encoder.window
    embeds = [encoder.encode(states[t - w + 1 : t + 1])[0] for t in tqdm(range(w - 1, len(states)), desc=desc)]
    return np.stack(embeds), np.arange(w - 1, len(states))


def encode_semantic(frame_paths, encoder: OwlVitSemanticEncoder, desc: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every individually detected label's own BERT embedding, kept
    separate rather than pooled into one vector per frame -- a frame with
    3 detections contributes 3 rows, a frame with 0 contributes none.
    frame_idx says which frame each row came from, so per-frame scoring
    can be reconstructed later without re-running OWL-ViT."""
    embeds, frame_idx, labels, scores = [], [], [], []
    for t, p in enumerate(tqdm(frame_paths, desc=desc)):
        detections, _ = encoder._detect(Image.open(p).convert("RGB"))
        for d in detections:
            embeds.append(encoder.label_embeddings[d["label"]])
            frame_idx.append(t)
            labels.append(d["label"])
            scores.append(d["score"])
    return np.stack(embeds), np.array(frame_idx), np.array(labels), np.array(scores)


def encode_run(run_name: str, data_path: str, out_path: str, active_encoders: list[str], models: dict, suffix: str) -> None:
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

    if "physics_roles" in active_encoders:
        embeddings, ticks = encode_physics_roles(states, models["physics_roles"], f"{run_name}/physics_roles")
        out_file = out_dir / f"physics_roles{suffix}.npz"
        np.savez(out_file, embeddings=embeddings, ticks=ticks)
        print(f"  physics_roles{suffix}: {embeddings.shape} -> {out_file}")

    if "physics" in active_encoders:
        embeddings, ticks = encode_physics(states, models["physics"], f"{run_name}/physics")
        out_file = out_dir / f"physics{suffix}.npz"
        np.savez(out_file, embeddings=embeddings, ticks=ticks)
        print(f"  physics{suffix}: {embeddings.shape} -> {out_file}")

    if "semantic" in active_encoders:
        embeddings, frame_idx, labels, scores = encode_semantic(frame_paths, models["semantic"], f"{run_name}/semantic")
        out_file = out_dir / f"semantic{suffix}.npz"
        np.savez(out_file, embeddings=embeddings, frame_idx=frame_idx, labels=labels, scores=scores)
        print(f"  semantic{suffix}: {embeddings.shape} -> {out_file}")


def main():
    config = json.loads(CONFIG_PATH.read_text())
    active_encoders = config["encoders"]

    models = {}
    if "vision" in active_encoders:
        models["vision"] = DinoVisionEncoder()
    if "vision_patch" in active_encoders:
        models["vision_patch"] = DinoPatchEncoder()
    if "physics_roles" in active_encoders:
        models["physics_roles"] = RolePhysicsEncoder()
    if "semantic" in active_encoders:
        models["semantic"] = OwlVitSemanticEncoder(**config.get("semantic_encoder_params", {}))
    if "physics" in active_encoders:
        models["physics"] = GroundTruthPhysicsEncoder()

    if "--anomaly-only" not in sys.argv:
        for ref in config.get("reference_runs") or [config["reference_run"]]:
            encode_run(ref["run_name"], ref["data_path"], ref["out_path"], active_encoders, models, suffix="_nominal")

    for run in config["anomaly_runs"]:
        encode_run(run["run_name"], run["data_path"], run["out_path"], active_encoders, models, suffix="")


if __name__ == "__main__":
    main()
