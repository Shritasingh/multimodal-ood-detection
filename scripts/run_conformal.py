"""Run-level conformal thresholds per scored series, then run-level detection results.

Frames within a run are strongly correlated, so they are not exchangeable and a quantile over pooled frames
gives no valid guarantee. Following UNISafe (Seo et al., 2025, Sec. 5.2) the unit of calibration is a run:

  1. each nominal reference run is scored against the pool of the other runs (leave-one-run-out);
  2. each run is summarised by Q = its (1 - alpha_trans) quantile of frame scores (at most an alpha_trans
     fraction of its frames lie above Q);
  3. the threshold tau is the ceil((1 - alpha_cal)(N + 1))-th smallest Q over the N reference runs.

Guarantee (runs exchangeable): with probability >= 1 - alpha_cal, a new nominal run has at most an alpha_trans
fraction of its frames above tau. With N runs, alpha_cal must be >= 1/(N + 1) or tau is infinite.

Results per run: a nominal (negative) run is falsely flagged if its Q exceeds tau; an anomaly run is detected if
its anomaly window's Q exceeds tau (more than alpha_trans of the window above tau). Also reported: fraction of
window frames above tau, fraction of the whole run's frames above tau (run_frac_above), delay from onset to the first frame above tau, and same-frame AUROC vs the negative run
(threshold-free frame ranking). --frames restores the old pooled-frame thresholds (alpha = --alpha).

    run_conformal.py [config] [--alpha-trans 0.05] [--alpha-cal 0.1] [--recalibrate] [--negative-run RUN] [--frames]
"""
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


def calibration_scores(config: dict, params: dict) -> dict[str, list[np.ndarray]]:
    """Per series key, one array of frame scores per calibration run. With calibration_runs in the data config
    (split conformal) these are those runs' own score csvs, already scored against the reference pool by
    run_metrics.py; otherwise every reference run is scored against the pool of the other runs (leave-one-run-out)."""
    if config.get("calibration_runs"):
        stem = config.get("output_stem", "anomaly_scores")
        cal: dict[str, list[np.ndarray]] = {}
        for run in config["calibration_runs"]:
            d = load_series(ROOT / run["out_path"] / f"{stem}.csv")
            for k, v in d.items():
                if k != "tick":
                    cal.setdefault(k, []).append(v[~np.isnan(v)])
        return cal
    import torch

    refs = [r["out_path"] for r in config["reference_runs"]]
    active = rm.active_encoders(config)
    series = rm.series_of(active, params)
    out: dict[str, list[np.ndarray]] = {key: [] for key, _, _ in series}
    for i, path in enumerate(refs):
        nominal, inv_cov = rm.build_reference(refs[:i] + refs[i + 1 :], active, params)
        for key, name, metric in series:
            _, vals = rm.score_encoder(name, nominal[name], rm.load_npz(path, f"{name}_nominal"), params[name], metric, inv_cov.get(name))
            v = np.asarray(vals, dtype=float)
            out[key].append(v[~np.isnan(v)])
        del nominal
        torch.cuda.empty_cache()
        print(f"  calibration {i + 1}/{len(refs)} done ({path})", flush=True)
    return out


def save_cache(path: Path, cal: dict[str, list[np.ndarray]]) -> None:
    np.savez(path, **{f"{k}::{i}": v for k, runs in cal.items() for i, v in enumerate(runs)})


def load_cache(path: Path) -> dict[str, list[np.ndarray]] | None:
    """Per-run calibration scores, or None if the cache is missing or in the old pooled-frame format."""
    if not path.exists():
        return None
    d = np.load(path)
    if not all("::" in k for k in d.files):
        return None
    out: dict[str, dict[int, np.ndarray]] = {}
    for k in d.files:
        key, i = k.rsplit("::", 1)
        out.setdefault(key, {})[int(i)] = d[k]
    return {k: [v[i] for i in sorted(v)] for k, v in out.items()}


def order_stat(x: np.ndarray, q: float) -> float:
    """The ceil(q * n)-th smallest value: at most a (1 - q) fraction of x lies above it."""
    x = np.sort(np.asarray(x, dtype=float))
    return float(x[min(len(x), max(1, math.ceil(q * len(x)))) - 1]) if len(x) else float("nan")


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; inf if n is too small for this alpha."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def load_series(csv_path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(csv_path)))
    return {c: np.array([float(r[c]) if r[c] else np.nan for r in rows]) for c in rows[0]}


def auroc_same_ticks(v: np.ndarray, t: np.ndarray, mask: np.ndarray, neg: dict | None, key: str) -> float | None:
    """Frames in `mask` vs the negative run at the same frames (threshold-free ranking check)."""
    from sklearn.metrics import roc_auc_score

    if neg is None:
        return None
    nmap = {int(a): b for a, b in zip(neg["tick"], neg[key]) if not np.isnan(b)}
    pairs = [(x, nmap[int(tt)]) for x, tt in zip(v[mask], t[mask]) if not np.isnan(x) and int(tt) in nmap]
    if not pairs or len({s for p in pairs for s in p}) < 2:
        return 0.5 if pairs else None
    p, n = zip(*pairs)
    return float(roc_auc_score([1] * len(p) + [0] * len(n), list(p) + list(n)))


def auroc_vs_pool(window: np.ndarray, pool: list[np.ndarray]) -> float | None:
    """Frames of the anomaly window vs every frame of the held-out nominal runs (threshold-free ranking check)."""
    from sklearn.metrics import roc_auc_score

    p = window[~np.isnan(window)]
    n = np.concatenate([x[~np.isnan(x)] for x in pool]) if pool else np.array([])
    if not len(p) or not len(n):
        return None
    return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))


def summary_table(config: dict, results: dict, tau: dict) -> dict:
    """Run-level detection rate per anomaly type / kind and false-alarm rate on the held-out runs, per series."""
    groups: dict[str, list[str]] = {}
    for run in config["anomaly_runs"]:
        t, k = run.get("anomaly_type", "anomaly"), run.get("anomaly_kind")
        groups.setdefault(t, []).append(run["run_name"])
        if k:
            groups.setdefault(f"{t}/{k}", []).append(run["run_name"])
    groups["false alarms"] = [r["run_name"] for r in config.get("negative_runs", []) if r["run_name"] in results]
    return {g: {key: float(np.mean([results[r][key]["flagged"] for r in runs])) for key in tau} | {"n_runs": len(runs)} for g, runs in groups.items() if runs}


def run_result(v: np.ndarray, t: np.ndarray, mask: np.ndarray, tau: float, alpha_trans: float) -> dict:
    """Run-level decision on the frames in `mask`: Q, flagged, fraction above tau, first frame above tau."""
    ok = mask & ~np.isnan(v)
    x = v[ok]
    q = order_stat(x, 1 - alpha_trans)
    above = t[ok][x > tau]
    return {"n_frames": int(ok.sum()), "Q": q, "flagged": bool(q > tau), "frac_above": float((x > tau).mean()) if len(x) else None,
            "first_above": int(above[0]) if len(above) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(ROOT / "config" / "encoders.json"))
    ap.add_argument("--alpha-trans", type=float, help="max fraction of a nominal run's frames above tau (default: config conformal.alpha_trans or 0.05)")
    ap.add_argument("--alpha-cal", type=float, help="probability a nominal run breaks that bound (default: config conformal.alpha_cal or 0.1)")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--negative-run", help="held-out nominal run for the false-alarm rate (default: the first of data.json's negative_runs)")
    ap.add_argument("--frames", action="store_true", help="old behaviour: one quantile over pooled frames (not a valid guarantee)")
    ap.add_argument("--alpha", type=float, default=0.05, help="--frames only: frame-level alpha")
    args = ap.parse_args()
    config = rm.load_config(args.config)
    cc = config.get("conformal", {})
    a_trans = args.alpha_trans if args.alpha_trans is not None else cc.get("alpha_trans", 0.05)
    a_cal = args.alpha_cal if args.alpha_cal is not None else cc.get("alpha_cal", 0.1)
    if args.negative_run is None and config.get("negative_runs"):
        args.negative_run = config["negative_runs"][0]["run_name"]
    params = rm.params_from_config(config)
    stem = config.get("output_stem", "anomaly_scores")
    cal_path = ROOT / "data" / config.get("calibration_cache", "conformal_calibration_seeds.npz")

    split = bool(config.get("calibration_runs"))
    cal = None if args.recalibrate or split else load_cache(cal_path)
    if cal is None:
        cal = calibration_scores(config, params)
        if not split:
            save_cache(cal_path, cal)
    n_runs = len(next(iter(cal.values())))
    if args.frames:
        tau = {k: conformal_threshold(np.concatenate(v), args.alpha) for k, v in cal.items()}
        run_q = {}
        print(f"FRAME-LEVEL (old): alpha = {args.alpha} over pooled nominal frames; no valid guarantee (frames are correlated)")
    else:
        run_q = {k: [order_stat(v, 1 - a_trans) for v in runs] for k, runs in cal.items()}
        tau = {k: conformal_threshold(np.array(q), a_cal) for k, q in run_q.items()}
        print(f"RUN-LEVEL: {n_runs} calibration runs, alpha_trans = {a_trans}, alpha_cal = {a_cal}")
        print(f"  guarantee: P(a new nominal run has more than {a_trans:.0%} of its frames above tau) <= {a_cal:.0%}")
        if math.ceil((n_runs + 1) * (1 - a_cal)) > n_runs:
            print(f"  WARNING: alpha_cal = {a_cal} needs >= {math.ceil(1 / a_cal) - 1} runs; tau is infinite (nothing can be flagged)")
    for k, v in tau.items():
        extra = f"   run scores: {', '.join(f'{q:.3g}' for q in run_q[k])}" if run_q else ""
        print(f"  {k:30s} tau = {v:.4g}{extra}")

    out_paths = {r["run_name"]: ROOT / r["out_path"] for r in config.get("negative_runs", []) + config.get("calibration_runs", [])}
    out_of = lambda name: out_paths.get(name, ROOT / "data" / name / "embeddings")  # noqa: E731
    neg = load_series(out_of(args.negative_run) / f"{stem}.csv") if args.negative_run else None
    # same-frame AUROC needs the anomaly runs' replay twin (same seed, so frame t is the same place on the route):
    # always the first negative run (nominal seed 11), whichever run --negative-run uses for the false-alarm rate
    # a randomised batch has no replay twin; frames are compared with every held-out nominal run's frames instead
    twin_name = None if split else config["negative_runs"][0]["run_name"] if config.get("negative_runs") else args.negative_run
    negs = {r["run_name"]: load_series(ROOT / r["out_path"] / f"{stem}.csv") for r in config.get("negative_runs", [])} if split else {}
    twin = load_series(out_of(twin_name) / f"{stem}.csv") if twin_name else None
    results = {}
    for run in config["anomaly_runs"]:
        d = load_series(ROOT / run["out_path"] / f"{stem}.csv")
        t = d["tick"]
        win = (t >= run["onset_tick"]) & (t < run["offset_tick"])
        results[run["run_name"]] = {}
        for k in tau:
            r = run_result(d[k], t, win, tau[k], a_trans)
            r["delay"] = int(r["first_above"] - run["onset_tick"]) if r["first_above"] is not None else None
            r["AUROC_same_ticks"] = auroc_same_ticks(d[k], t, win, twin, k)
            if negs:
                r["AUROC_vs_negatives"] = auroc_vs_pool(d[k][win], [n[k] for n in negs.values()])
            whole = run_result(d[k], t, np.ones(len(t), bool), tau[k], a_trans)  # every frame of the run, not just the window
            r["run_frac_above"], r["run_flagged"] = whole["frac_above"], whole["flagged"]
            results[run["run_name"]][k] = r
    for name, n in (negs.items() if negs else ([(args.negative_run, neg)] if neg is not None else [])):
        results[name] = {k: run_result(n[k], n["tick"], np.ones(len(n["tick"]), bool), tau[k], a_trans) for k in tau}

    f = lambda x, w=5: " " * (w - 1) + "-" if x is None else (f"{x:{w}.2f}" if isinstance(x, float) else f"{x:>{w}}")  # noqa: E731
    print(f"\n{'run':20s} {'series':30s} {'flag':>5s} {'frac>tau':>8s} {'delay':>5s} {'AUROC':>6s}   (flag = run-level decision; false alarms on {args.negative_run}; AUROC = same frames vs {twin_name})")
    for name, res in results.items():
        for k, m in res.items():
            print(f"{name:20s} {k:30s} {('YES' if m['flagged'] else 'no'):>5s} {f(m['frac_above'], 8)} {f(m.get('delay'))} {f(m.get('AUROC_same_ticks'), 6)}")
    summary = summary_table(config, results, tau)
    if split:
        print(f"\nrun-level detection rate per anomaly type (false alarms = held-out nominal runs flagged)")
        print(f"{'group':28s} {'n':>3s} " + " ".join(f"{k[:14]:>14s}" for k in tau))
        for g, row in summary.items():
            print(f"{g:28s} {row['n_runs']:3d} " + " ".join(f"{row[k]:14.2f}" for k in tau))
    out = ROOT / "data" / config.get("conformal_results", "conformal_seeds.json")
    out.write_text(json.dumps({"mode": "frames" if args.frames else "runs", "alpha_trans": a_trans, "alpha_cal": a_cal, "alpha_frames": args.alpha if args.frames else None,
                               "n_calibration_runs": n_runs, "negative_run": args.negative_run, "auroc_twin_run": twin_name, "thresholds": tau,
                               "calibration_run_scores": run_q, "summary": summary, "results": results}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
