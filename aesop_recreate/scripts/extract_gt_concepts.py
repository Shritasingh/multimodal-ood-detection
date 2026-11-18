"""Ground-truth concepts for every recorded frame (sim/concepts.py), offline from the saved segmentation, depth and
camera pose; the lanes come from the town's saved OpenDRIVE map, so no CARLA server is needed.

    .venv/bin/python aesop_recreate/scripts/extract_gt_concepts.py              # every run in data/semantic24
    .venv/bin/python aesop_recreate/scripts/extract_gt_concepts.py t2_haybale   # some runs

The kerb-strip rule is on (--kerb-sidewalk-m, default sim/concepts.KERB_SIDEWALK_M = 1.5 m): an object with no lane
under it this close to a driving lane's edge is on the sidewalk.

Writes data/semantic24/<run>/concepts_gt_raw.jsonl: {"t", "instances": [{id, object, predicate, px, box, xyz, lane, tag,
src}]} per tick. Hand corrections are kept apart (gt_overrides.json, from the dashboard's Images tab), so re-running
this after a rule change keeps them.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT.parent)]       # sim/ merges with multimodal-ood-detection's
from sim import concepts as K, semantic24 as P  # noqa: E402

OUT = ROOT / "data" / P.DATASET


def extract(run_dir: Path, kerb_m: float | None = None) -> str:
    import carla

    info = json.loads((run_dir / "run_info.json").read_text())
    town = info["town"]
    towns = run_dir.parent / "_towns"   # the dataset's saved maps
    cmap = carla.Map(town, (towns / f"{town}.xodr").read_text())
    env = K.EnvIndex(json.loads((towns / f"{town}_env.json").read_text()))
    actors = {int(k): v for k, v in json.loads((run_dir / "actors.json").read_text()).items()}
    cams = [json.loads(l) for l in (run_dir / "camera.jsonl").read_text().splitlines()]
    with open(run_dir / "concepts_gt_raw.jsonl", "w") as f:
        for cam in cams:
            t = cam["t"]
            inst = cv2.imread(str(run_dir / "inst" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(str(run_dir / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 100.0
            f.write(json.dumps({"t": t, "instances": K.frame_instances(inst, depth, cam, info["camera"]["fov"], cmap, env, actors, kerb_m)}) + "\n")
    return f"{run_dir.name}: {len(cams)} frames"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names (default: every recorded run)")
    ap.add_argument("--data-dir", type=Path, default=OUT, help=f"dataset directory (default {OUT.relative_to(ROOT.parent)})")
    ap.add_argument("--kerb-sidewalk-m", type=float, default=K.KERB_SIDEWALK_M,
                    help="kerb strip rule: no lane under it within this of a driving lane -> on the sidewalk")
    args = ap.parse_args()
    out = args.data_dir.resolve()
    runs = [out / n for n in args.runs] if args.runs else sorted(d for d in out.iterdir() if (d / "camera.jsonl").exists())
    with ProcessPoolExecutor(8) as ex:
        for msg in ex.map(partial(extract, kerb_m=args.kerb_sidewalk_m), runs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
