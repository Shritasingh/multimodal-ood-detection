"""Score nominal reference runs against the full nominal pool, so they can be inspected like any scored run.

The run is scored as is: its own frames are part of the pool, so frame-matching scores (vision patch, physics kNN)
sit near zero. Scores go to <out_path>/<output_stem>.csv like run_metrics.py writes for eval runs, and the analysis
scripts and dashboard then pick the run up.

    .venv/bin/python scripts/score_reference_run.py nominal_seed10                         # config/encoders.json
    .venv/bin/python scripts/score_reference_run.py nominal_seed10 --config config/encoders_qwen.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_metrics as rm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="reference run names from the data config")
    ap.add_argument("--config", default=str(rm.REPO_ROOT / "config" / "encoders.json"))
    args = ap.parse_args()

    config = rm.load_config(args.config)
    refs = {r["run_name"]: r for r in config["reference_runs"]}
    unknown = [r for r in args.runs if r not in refs]
    if unknown:
        ap.error(f"not reference runs in {args.config}: {', '.join(unknown)}")
    active, params = rm.active_encoders(config), rm.params_from_config(config)
    series, stem = rm.series_of(active, params), config.get("output_stem", "anomaly_scores")
    for name in args.runs:
        print(f"=== {name} (against the full pool, itself included) ===")
        nominal, inv_cov = rm.build_reference([r["out_path"] for r in refs.values()], active, params)
        scores = rm.score_run(refs[name]["out_path"], nominal, inv_cov, active, series, params, suffix="_nominal")
        rm.write_scores_csv(rm.resolve_path(refs[name]["out_path"]) / f"{stem}.csv", series, scores)
        del nominal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
