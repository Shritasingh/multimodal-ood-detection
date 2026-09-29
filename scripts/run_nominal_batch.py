"""Record nominal runs that differ only in seed, then verify frame/state and agent counts: run_nominal_batch.py [--seeds 1-10] (needs CARLA)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def check_run(name: str, cfg: dict) -> str:
    """Empty string if the run is complete with the requested agent counts, else what is wrong."""
    d = REPO_ROOT / "data" / name
    problems = []
    n_frames = len(list((d / "frames").glob("*.jpg")))
    n_states = len((d / "states.jsonl").read_text().splitlines()) if (d / "states.jsonl").exists() else 0
    if n_frames != cfg["n_ticks"] or n_states != cfg["n_ticks"]:
        problems.append(f"{n_frames} frames / {n_states} states, expected {cfg['n_ticks']}")
    info_path = d / "run_info.json"
    if not info_path.exists():
        problems.append("no run_info.json")
    else:
        info = json.loads(info_path.read_text())
        if info["n_vehicles_spawned"] != cfg["n_background_vehicles"] or info["n_walkers_spawned"] != cfg["n_background_walkers"]:
            problems.append(f"spawned {info['n_vehicles_spawned']} vehicles / {info['n_walkers_spawned']} walkers, "
                            f"expected {cfg['n_background_vehicles']} / {cfg['n_background_walkers']}")
    return "; ".join(problems)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1-10", help="e.g. 1-10 or 1-3,7")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "sim_nominal.json")
    p.add_argument("--prefix", default="nominal_seed")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())

    report = []
    for seed in parse_seeds(args.seeds):
        name = f"{args.prefix}{seed:02d}"
        if (REPO_ROOT / "data" / name / "states.jsonl").exists() and not args.overwrite:
            problem = check_run(name, cfg)
            print(f"=== {name}: already recorded ({'ok' if not problem else problem}), skipping")
            report.append((name, seed, 0.0, problem or "ok (existing)"))
            continue
        print(f"=== {name} (seed {seed})", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_sim.py"), "--config", str(args.config), "--run-name", name, "--seed", str(seed)]
        rc = subprocess.run(cmd).returncode
        problem = f"run_sim exited {rc}" if rc else check_run(name, cfg)
        report.append((name, seed, time.time() - t0, problem or "ok"))

    print("\nrun                seed  minutes  status")
    for name, seed, secs, status in report:
        print(f"{name:18s} {seed:4d}  {secs / 60:7.1f}  {status}")
    if any(not r[3].startswith("ok") for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
