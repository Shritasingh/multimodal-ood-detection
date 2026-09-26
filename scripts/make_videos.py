"""Stitches each run's frames/*.jpg into an mp4, saved alongside the frames.

Usage:
    .venv/bin/python scripts/make_videos.py [run_name ...]
    .venv/bin/python scripts/make_videos.py --start 600 --end 800 [run_name ...]

With no run names, does every directory under data/ that has a frames/
subfolder. --start/--end clip to a frame index range (inclusive start,
exclusive end) and suffix the output filename so the full-run video, if
already made, isn't overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FPS = 10  # matches --fixed-delta 0.1 in run_sim.py


def make_video(run_dir: Path, start: int | None, end: int | None) -> None:
    frame_paths = sorted((run_dir / "frames").glob("*.jpg"))
    if not frame_paths:
        print(f"skip {run_dir.name}: no frames")
        return
    suffix = ""
    if start is not None or end is not None:
        lo, hi = start or 0, end if end is not None else len(frame_paths)
        frame_paths = frame_paths[lo:hi]
        suffix = f"_{lo}-{hi}"
        if not frame_paths:
            print(f"skip {run_dir.name}: no frames in range [{lo}, {hi})")
            return
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    out_path = run_dir / f"{run_dir.name}{suffix}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for p in frame_paths:
        writer.write(cv2.imread(str(p)))
    writer.release()
    print(f"wrote {out_path} ({len(frame_paths)} frames @ {FPS}fps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_names", nargs="*")
    p.add_argument("--start", type=int, default=None, help="first frame index to include")
    p.add_argument("--end", type=int, default=None, help="frame index to stop before")
    args = p.parse_args()

    if args.run_names:
        run_dirs = [DATA_DIR / name for name in args.run_names]
    else:
        run_dirs = sorted(d for d in DATA_DIR.iterdir() if (d / "frames").is_dir())
    for run_dir in run_dirs:
        make_video(run_dir, args.start, args.end)


if __name__ == "__main__":
    main()
