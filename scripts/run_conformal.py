"""Conformal score thresholds per scored series (leave-one-run-out, or a held-out nominal run), then TPR/FPR/TNR/AUROC per anomaly run: run_conformal.py [config] [--alpha 0.05] [--calibration-run RUN --negative-run RUN]."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_metrics as rm

ROOT = rm.REPO_ROOT


def calibration_scores(config: dict, params: dict) -> dict[str, np.ndarray]:
    """Score every reference run against the pool of the other runs (leave-one-run-out): nominal scores per series key."""
    import torch

    refs = [r["out_path"] for r in config["reference_runs"]]
    series = rm.series_of(config["encoders"], params)
    out = {key: [] for key, _, _ in series}
    for i, path in enumerate(refs):
        nominal, inv_cov = rm.build_reference(refs[:i] + refs[i + 1 :], config["encoders"], params)
        for key, name, metric in series:
            _, vals = rm.score_encoder(name, nominal[name], rm.load_npz(path, f"{name}_nominal"), params[name], metric, inv_cov.get(name))
            out[key] += [v for v in vals if not np.isnan(v)]
        del nominal
        torch.cuda.empty_cache()
        print(f"  calibration {i + 1}/{len(refs)} done ({path})", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest nominal score; a score above it is flagged and nominal data is flagged with probability <= alpha."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def load_series(csv_path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(csv_path)))
    return {c: np.array([float(r[c]) if r[c] else np.nan for r in rows]) for c in rows[0]}


def confusion(v: np.ndarray, t: np.ndarray, pos: np.ndarray, neg: np.ndarray, tau: float) -> dict:
    """Counts and rates of `score > tau` over the positive and negative tick masks."""
    ok = ~np.isnan(v)
    p, n = pos & ok, neg & ok
    tp, fp = int((v[p] > tau).sum()), int((v[n] > tau).sum())
    tpr, fpr = tp / max(p.sum(), 1), fp / max(n.sum(), 1)
    hits = t[p & (np.nan_to_num(v) > tau)]
    return {"P": int(p.sum()), "N": int(n.sum()), "TP": tp, "FN": int(p.sum()) - tp, "FP": fp, "TN": int(n.sum()) - fp,
            "TPR": tpr, "FPR": fpr, "TNR": 1 - fpr, "BAcc": (tpr + 1 - fpr) / 2 if p.any() else None,
            "delay": int(hits[0] - t[p][0]) if len(hits) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(ROOT / "config" / "encoders_seeds.json"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--calibration-run", help="calibrate on this held-out nominal run's saved scores instead of leave-one-run-out")
    ap.add_argument("--negative-run", help="a second held-out nominal run: independent negatives for FPR/TNR and AUROC")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    params = rm.params_from_config(config)
    stem = config.get("output_stem", "anomaly_scores")
    cal_path = ROOT / "data" / config.get("calibration_cache", "conformal_calibration_seeds.npz")

    if args.calibration_run:
        c = load_series(ROOT / "data" / args.calibration_run / "embeddings" / f"{stem}.csv")
        cal = {k: c[k][~np.isnan(c[k])] for k in c if k != "tick"}
    elif cal_path.exists() and not args.recalibrate:
        cal = dict(np.load(cal_path))
    else:
        cal = calibration_scores(config, params)
        np.savez(cal_path, **cal)
    tau = {k: conformal_threshold(v, args.alpha) for k, v in cal.items()}
    print(f"alpha = {args.alpha}: thresholds from {args.calibration_run or 'leave-one-run-out'} nominal ticks")
    for k, v in tau.items():
        print(f"  {k:30s} tau = {v:.4f}   ({len(cal[k])} calibration scores)")

    from sklearn.metrics import roc_auc_score

    def auroc(pos_scores, neg_scores):
        pos_scores, neg_scores = pos_scores[~np.isnan(pos_scores)], neg_scores[~np.isnan(neg_scores)]
        if not len(pos_scores) or not len(neg_scores):
            return None
        return float(roc_auc_score(np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))], np.r_[pos_scores, neg_scores]))

    calib = load_series(ROOT / "data" / args.calibration_run / "embeddings" / f"{stem}.csv") if args.calibration_run else None
    neg = load_series(ROOT / "data" / args.negative_run / "embeddings" / f"{stem}.csv") if args.negative_run else None
    results = {}
    for run in config["anomaly_runs"]:
        d = load_series(ROOT / run["out_path"] / f"{stem}.csv")
        t = d["tick"]
        pos, neg_mask = (t >= run["onset_tick"]) & (t < run["offset_tick"]), t < run["onset_tick"]
        results[run["run_name"]] = {k: confusion(d[k], t, pos, neg_mask, tau[k]) for k in tau}
        for k in tau:
            r = results[run["run_name"]][k]
            if neg is not None:
                nv = neg[k][~np.isnan(neg[k])]
                r["FPR_indep"], r["TNR_indep"] = float((nv > tau[k]).mean()), float((nv <= tau[k]).mean())
                r["AUROC_vs_neg"] = auroc(d[k][pos], neg[k])
            if calib is not None:
                r["AUROC_same_ticks"] = auroc(d[k][pos], calib[k][np.isin(calib["tick"], t[pos])])
    f = lambda x: "  -  " if x is None else f"{x:5.2f}"
    print(f"\n{'run':20s} {'series':30s} {'TPR':>5s} {'FPR':>5s} {'TNR':>5s} {'AUROC':>6s} {'AUROC':>6s}   (FPR/TNR on {args.negative_run or 'pre-onset ticks'}; AUROC vs same-tick {args.calibration_run or 'n/a'} | vs {args.negative_run or 'n/a'})")
    for name, res in results.items():
        for k, m in res.items():
            fpr, tnr = (m["FPR_indep"], m["TNR_indep"]) if "FPR_indep" in m else (m["FPR"], m["TNR"])
            print(f"{name:20s} {k:30s} {f(m['TPR'] if m['P'] else None):>5s} {f(fpr):>5s} {f(tnr):>5s} {f(m.get('AUROC_same_ticks')):>6s} {f(m.get('AUROC_vs_neg')):>6s}")
    out = ROOT / "data" / (f"conformal_calib_{args.calibration_run}.json" if args.calibration_run else config.get("conformal_results", "conformal_seeds.json"))
    out.write_text(json.dumps({"alpha": args.alpha, "thresholds": tau, "results": results}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
