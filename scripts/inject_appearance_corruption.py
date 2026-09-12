"""Recreate a visually-corrupted copy of an already-collected
run without recollecting. 

Usage:
    .venv/bin/python scripts/inject_appearance_corruption.py \
        --src-run nominal_run --dst-run anomaly_flare \
        --corruption flare --onset-frame 800 --duration 150
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # path to sim/ package (repo root)
from sim.anomalies import VISUAL_CORRUPTIONS, inject_visual


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-run", required=True, help="run name under data/ to read frames from")
    p.add_argument("--dst-run", required=True, help="run name under data/ to write the corrupted copy to")
    p.add_argument("--corruption", choices=list(VISUAL_CORRUPTIONS), default="flare")
    p.add_argument("--onset-frame", type=int, required=True)
    p.add_argument("--duration", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / "data"
    src_dir = data_root / args.src_run
    dst_dir = data_root / args.dst_run
    (dst_dir / "frames").mkdir(parents=True, exist_ok=True)

    # Ground-truth agent state is unaffected by an appearance-only anomaly.
    if (src_dir / "states.jsonl").exists():
        shutil.copy(src_dir / "states.jsonl", dst_dir / "states.jsonl")
    for frame_path in sorted((src_dir / "frames").glob("*.jpg")):
        shutil.copy(frame_path, dst_dir / "frames" / frame_path.name)

    fields = inject_visual(dst_dir, args.corruption, args.onset_frame, args.duration, seed=args.seed)
    (dst_dir / "anomalies.json").write_text(json.dumps({"anomaly_type": "visual", **fields}, indent=2))
    print(f"Wrote {fields['n_frames']} frames to {dst_dir} (anomaly window [{fields['onset_frame']}, {fields['offset_frame']}))")


if __name__ == "__main__":
    main()
