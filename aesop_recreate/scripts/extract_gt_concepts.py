"""Ground-truth concepts for every recorded frame (sim/concepts.py), offline from the saved segmentation, depth and
camera pose; the lanes come from the town's saved OpenDRIVE map, so no CARLA server is needed.

    .venv/bin/python scripts/extract_gt_concepts.py              # every run in data/semantic24
    .venv/bin/python scripts/extract_gt_concepts.py t2_umbrella  # some runs

Writes data/semantic24/<run>/concepts_gt_raw.jsonl: {"t", "instances": [{id, object, predicate, px, box, xyz, lane, tag,
src}]} per tick. Hand corrections are kept apart (gt_overrides.json, from the dashboard's Labels tab), so re-running
this after a rule change keeps them.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT.parent)]       # sim/ merges with multimodal-ood-detection's
from sim import concepts as K  # noqa: E402

OUT = ROOT / "data" / "semantic24"


def extract(run_dir: Path) -> str:
    import carla

    info = json.loads((run_dir / "run_info.json").read_text())
    town = info["town"]
    cmap = carla.Map(town, (OUT / "_towns" / f"{town}.xodr").read_text())
    env = K.EnvIndex(json.loads((OUT / "_towns" / f"{town}_env.json").read_text()))
    actors = {int(k): v for k, v in json.loads((run_dir / "actors.json").read_text()).items()}
    cams = [json.loads(l) for l in (run_dir / "camera.jsonl").read_text().splitlines()]
    with open(run_dir / "concepts_gt_raw.jsonl", "w") as f:
        for cam in cams:
            t = cam["t"]
            inst = cv2.imread(str(run_dir / "inst" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(str(run_dir / "depth" / f"{t:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 100.0
            f.write(json.dumps({"t": t, "instances": K.frame_instances(inst, depth, cam, info["camera"]["fov"], cmap, env, actors)}) + "\n")
    return f"{run_dir.name}: {len(cams)} frames"


def main():
    names = sys.argv[1:]
    runs = [OUT / n for n in names] if names else sorted(d for d in OUT.iterdir() if (d / "camera.jsonl").exists())
    with ProcessPoolExecutor(8) as ex:
        for msg in ex.map(extract, runs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
