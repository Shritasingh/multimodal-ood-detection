"""Score the semantic24 embeddings and set run-level conformal thresholds (one encoder config = one reference pool).

    .venv/bin/python aesop_recreate/scripts/score_semantic24.py aesop_recreate/config/encoders_semantic24.json        # withheld routes
    .venv/bin/python aesop_recreate/scripts/score_semantic24.py aesop_recreate/config/encoders_semantic24_incl.json   # included routes
    ... [--recalibrate]   (the leave-one-run-out calibration scores are cached; recompute after re-encoding)

Metrics per encoder (config `encoders`: {name: {"metrics": [...]}}):
  cosine_k<N>   mean cosine distance to the N nearest nominal frames (AESOP's kNN score)
  mahalanobis   Mahalanobis distance to the nominal mean, Ledoit-Wolf shrunk covariance (a Gaussian fit of the pool)
  patch         vision_patch: multimodal-ood-detection's DINOv2 patch score (worst 5% patches, k=1 cosine to the patch bank)
Test runs (anomaly + negative) are scored against the whole pool and written to <out_path>/<output_stem>.csv. The
thresholds follow multimodal-ood-detection's run_conformal.py (run level): each reference run is scored against the
other reference runs, summarised by Q = its (1 - alpha_trans) frame quantile, and tau is the conformal quantile of
those Qs. Per anomaly run and series: flagged (window Q > tau), fraction of window frames above tau (frame TPR),
delay, AUROC of window frames vs the held-out nominal run's frames, and vs the calibration frames. Per negative run:
flagged (false alarm) and fraction of frames above tau (frame FPR). Results go to data/<conformal_results>, in
run_conformal.py's layout so the dashboard's Summary tab reads them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
MAIN = ROOT.parent
sys.path[:0] = [str(ROOT), str(MAIN), str(MAIN / "scripts")]
import run_conformal as rc  # noqa: E402
import run_metrics as rm  # noqa: E402


def series_of(cfg: dict) -> list[tuple[str, str, str]]:
    """(csv column, encoder, metric) for every scored series."""
    return [(f"{name}/{m}", name, m) for name, p in cfg["encoders"].items() for m in p["metrics"]]


def load(out_paths: list[str], name: str) -> dict:
    return rm.load_reference(out_paths, name)


class Pool:
    """One reference pool's fitted state per encoder (normalised rows, Gaussian fit, patch bank), built lazily."""

    def __init__(self, out_paths: list[str]):
        self.out_paths, self.cache = out_paths, {}

    def get(self, name: str, metric: str):
        key = (name, "patch" if metric == "patch" else "maha" if metric == "mahalanobis" else "cos")
        if key not in self.cache:
            if key[1] == "patch":
                self.cache[key] = {"bank": rm.load_patch_bank(self.out_paths, f"{name}_nominal")}
            else:
                x = load(self.out_paths, f"{name}_nominal")["embeddings"].astype(np.float64)
                if key[1] == "cos":
                    self.cache[key] = x / np.linalg.norm(x, axis=1, keepdims=True)
                else:
                    from sklearn.covariance import LedoitWolf

                    lw = LedoitWolf().fit(x)
                    self.cache[key] = (lw.location_, lw.precision_)
        return self.cache[key]


def score(pool: Pool, name: str, metric: str, query: dict) -> np.ndarray:
    """Per-frame scores of one run's embeddings (query) against the pool, higher = more anomalous."""
    if metric == "patch":
        return np.array(rm.score_vision_patch(pool.get(name, metric), query, top_frac=0.05, k=1)[1])
    q = query["embeddings"].astype(np.float64)
    if metric == "mahalanobis":
        mu, prec = pool.get(name, metric)
        d = q - mu
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, prec, d), 0, None))
    k = int(metric.split("_k")[1])
    ref = pool.get(name, metric)
    dist = 1.0 - (q / np.linalg.norm(q, axis=1, keepdims=True)) @ ref.T
    k = min(k, dist.shape[1])
    return np.partition(dist, k - 1, axis=1)[:, :k].mean(1)


def score_runs(pool: Pool, runs: list[dict], series: list, suffix: str = "") -> dict[str, dict[str, np.ndarray]]:
    """{run: {"tick": ticks, column: scores}}."""
    out = {}
    for r in runs:
        d = {}
        for key, name, metric in series:
            q = rm.load_npz(r["out_path"], f"{name}{suffix}")
            d["tick"], d[key] = q["ticks"].astype(float), score(pool, name, metric, q)
        out[r["run_name"]] = d
    return out


def calibrate(cfg: dict, series: list, cache: Path, recalibrate: bool) -> dict[str, list[np.ndarray]]:
    """Leave-one-run-out frame scores of every reference run, per series (cached)."""
    if cache.exists() and not recalibrate:
        got = rc.load_cache(cache)
        if got is not None and all(k in got for k, _, _ in series):
            return got
    refs = cfg["reference_runs"]
    cal: dict[str, list[np.ndarray]] = {k: [] for k, _, _ in series}
    for i, r in enumerate(refs):
        pool = Pool([x["out_path"] for x in refs if x is not r])
        for key, v in score_runs(pool, [r], series, "_nominal")[r["run_name"]].items():
            if key != "tick":
                cal[key].append(v)
        print(f"  calibration {i + 1}/{len(refs)}: {r['run_name']}", flush=True)
    rc.save_cache(cache, cal)
    return cal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--recalibrate", action="store_true")
    args = ap.parse_args()
    cfg = rm.load_config(rm.resolve_path(args.config))
    series = series_of(cfg)
    stem = cfg["output_stem"]
    a_trans, a_cal = cfg["conformal"]["alpha_trans"], cfg["conformal"]["alpha_cal"]

    pool = Pool([r["out_path"] for r in cfg["reference_runs"]])
    tests = score_runs(pool, cfg["anomaly_runs"] + cfg["negative_runs"], series)
    for r in cfg["anomaly_runs"] + cfg["negative_runs"]:
        d = tests[r["run_name"]]
        rm.write_scores_csv(rm.resolve_path(r["out_path"]) / f"{stem}.csv", series,
                            {k: (d["tick"].astype(int).tolist(), d[k].tolist()) for k, _, _ in series})

    cal = calibrate(cfg, series, MAIN / "data" / cfg["calibration_cache"], args.recalibrate)
    run_q = {k: [rc.order_stat(v, 1 - a_trans) for v in runs] for k, runs in cal.items()}
    tau = {k: rc.conformal_threshold(np.array(q), a_cal) for k, q in run_q.items()}
    print(f"RUN-LEVEL: {len(cfg['reference_runs'])} calibration runs, alpha_trans = {a_trans}, alpha_cal = {a_cal}")

    negs = {r["run_name"]: tests[r["run_name"]] for r in cfg["negative_runs"]}
    results = {}
    for r in cfg["anomaly_runs"]:
        d = tests[r["run_name"]]
        t = d["tick"]
        win = (t >= r["onset_tick"]) & (t < r["offset_tick"])
        results[r["run_name"]] = {}
        for k in tau:
            m = rc.run_result(d[k], t, win, tau[k], a_trans)
            m["delay"] = int(m["first_above"] - r["onset_tick"]) if m["first_above"] is not None else None
            m["AUROC_vs_negatives"] = rc.auroc_vs_pool(d[k][win], [n[k] for n in negs.values()])
            m["AUROC_vs_calibration"] = rc.auroc_vs_pool(d[k][win], cal[k])
            whole = rc.run_result(d[k], t, np.ones(len(t), bool), tau[k], a_trans)
            m["run_frac_above"], m["run_flagged"] = whole["frac_above"], whole["flagged"]
            results[r["run_name"]][k] = m
    for name, n in negs.items():
        results[name] = {k: rc.run_result(n[k], n["tick"], np.ones(len(n["tick"]), bool), tau[k], a_trans) for k in tau}
    summary = rc.summary_table(cfg, results, tau)

    w = max(len(k) for k in tau)
    print(f"\n{'series':{w}s}  tau      " + "  ".join(f"{r['run_name'][3:]:>16s}" for r in cfg["anomaly_runs"] + cfg["negative_runs"]))
    for k in tau:
        cells = []
        for r in cfg["anomaly_runs"] + cfg["negative_runs"]:
            m = results[r["run_name"]][k]
            au = m.get("AUROC_vs_negatives")
            cells.append(f"{'FLAG' if m['flagged'] else '-':>4s} {m['frac_above'] or 0:4.2f} {'' if au is None else f'{au:.2f}':>5s}")
        print(f"{k:{w}s}  {tau[k]:7.4f}  " + "  ".join(f"{c:>16s}" for c in cells))
    out = MAIN / "data" / cfg["conformal_results"]
    out.write_text(json.dumps({"mode": "runs", "alpha_trans": a_trans, "alpha_cal": a_cal, "n_calibration_runs": len(cfg["reference_runs"]),
                               "negative_run": cfg["negative_runs"][0]["run_name"], "auroc_twin_run": None, "thresholds": tau,
                               "calibration_run_scores": run_q, "summary": summary, "results": results}, indent=1, default=float))
    print(f"\nwrote {out}  (cells: run flagged, fraction of window / run frames above tau, AUROC vs held-out nominal frames)")


if __name__ == "__main__":
    main()
