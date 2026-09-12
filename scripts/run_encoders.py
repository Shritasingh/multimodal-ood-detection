"""Run all three encoders over a collected run
(frames/ + states.jsonl) and cache per-timestep embeddings to data/<run>/embeddings/

Usage:
    .venv/bin/python scripts/run_encoders.py --run-name nominal_town03 --history-len 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from encoders.physics_encoder import GroundTruthPhysicsEncoder
from encoders.semantic_encoder import OwlVitSemanticEncoder
from encoders.vision_encoder import DinoVisionEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--history-len", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="only featurize every Nth tick, to save compute")
    p.add_argument(
        "--encoders",
        nargs="+",
        default=["vision", "semantic", "physics"],
        choices=["vision", "semantic", "physics"],
    )
    p.add_argument("--relative-range", type=float, default=40.0, help="physics encoder: max range (m) for tracked other agents")
    p.add_argument("--dt", type=float, default=0.1, help="physics encoder: seconds/tick, must match collection's --fixed-delta")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / "data"
    run_dir = data_root / args.run_name
    frame_paths = sorted((run_dir / "frames").glob("*.jpg"))
    states = [json.loads(line) for line in (run_dir / "states.jsonl").read_text().splitlines()]
    assert len(frame_paths) == len(states), (
        f"frame/state count mismatch: {len(frame_paths)} frames vs {len(states)} states"
    )

    out_dir = run_dir / "embeddings"
    out_dir.mkdir(exist_ok=True)

    encoders = {}
    if "vision" in args.encoders:
        encoders["vision"] = DinoVisionEncoder()
    if "semantic" in args.encoders:
        encoders["semantic"] = OwlVitSemanticEncoder()
    if "physics" in args.encoders:
        encoders["physics"] = GroundTruthPhysicsEncoder(relative_range_m=args.relative_range, dt=args.dt)

    results = {name: [] for name in encoders}
    result_ticks = []

    H = args.history_len
    for t in tqdm(range(H - 1, len(frame_paths), args.stride)):
        result_ticks.append(t)
        window = range(t - H + 1, t + 1)

        if "vision" in encoders or "semantic" in encoders:
            frames = [Image.open(frame_paths[i]).convert("RGB") for i in window]
        if "vision" in encoders:
            embedding, _ = encoders["vision"].encode(frames)
            results["vision"].append(embedding)
        if "semantic" in encoders:
            embedding, _ = encoders["semantic"].encode(frames)
            results["semantic"].append(embedding)
        if "physics" in encoders:
            state_window = [states[i] for i in window]
            embedding, _ = encoders["physics"].encode(state_window)
            results["physics"].append(embedding)

    for name, embeds in results.items():
        arr = np.stack(embeds)
        np.savez(out_dir / f"{name}.npz", embeddings=arr, ticks=np.array(result_ticks))
        print(f"{name}: {arr.shape} -> {out_dir / f'{name}.npz'}")


if __name__ == "__main__":
    main()
