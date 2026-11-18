"""The Experiment tab's results for the scored semantic24 corpus (build_corpus.py, score.py): the sweeps of
analysis/dashboard.html's Experiment tab, under score.py's protocol (ground-truth cache, run-level conformal tau,
test runs scored from raw OWL-ViT and from ground-truth labels).

    .venv/bin/python aesop_recreate/scripts/experiment.py     # both pools

Writes _results/experiment_<pool>.json:
  setup           per concept source (gt = corrected ground truth, owl = raw OWL-ViT), the pool runs in config order:
                  each run's concepts per point and the distinct concepts in the cache after adding it
  full            per series <source>@<embedder>/<metric>: tau and the metrics on all 4 test runs (score.py's)
  per_anomaly     the same with the test runs cut to t2_test_nominal + one anomaly run, at the same tau
  pool_sweep      the whole protocol on a random draw of SIZES pool runs (DRAWS draws; the included pool draws Town01 and
                  Town02 runs in equal numbers): tau from the drawn runs' leave-one-out scores, the test runs against
                  their cache
  corpus_sweep    every pool run, but the cache is a random FRACS share of the pool's points (DRAWS draws); coverage =
                  the share of the pool's distinct ground-truth concepts those points hold
  roc             the full pool's ROC curve, tau swept: tau(p) = the conformal rank of the Q_r, Q_r = each pool run's
                  p-th percentile of its leave-one-out scores, for p from p_lo (the highest p <= ROC_START at which every
                  anomaly window is detected) to 100 in ROC_STEP steps; above tau(100), up to the highest window peak
                  (where TPR reaches 0), and below tau(p_lo) if needed, the curve continues through the window peaks
                  themselves (p = null), just above and just below each, so every TPR step is exact. TP / FN / FP / TN
                  as the accuracy's (Elhafsi et al.): a window is a TP if any of its points is above tau, else a FN; a
                  nominal point above tau is a FP, else a TN. AUROC = the area under the curve, closed at (0, 0) and
                  (1, 1) (TPR stays 0 / 1 beyond its ends); it replaces score.py's AUROC in full ("AUROC_mw" keeps
                  that one, the Mann-Whitney statistic of the same sets, as a check)
  threshold_sweep the roc points with a p: TPR, FPR and accuracy vs p (95 = the protocol's 1 - alpha_frame)
Tau with fewer runs: the conformal rank ceil((N + 1)(1 - alpha_run)) exceeds N below N = 19, so the sweeps take the
largest Q_r there (marked "capped"), the strictest threshold the draw allows.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT.parent), str(ROOT.parent / "scripts")]
from encoders.text_embedders import TEXT_EMBEDDERS  # noqa: E402
from metrics import _ledoit_wolf, _maha, _t, metrics_at, window_ids  # noqa: E402
from score import ALPHA_FRAME, ALPHA_RUN, DEV, KS, METRICS, TEST_SOURCES, finite, load  # noqa: E402
from sim import semantic24 as P  # noqa: E402

SIZES = (2, 5, 10, 15, 20)
DRAWS = 5
FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ROC_START = 75.0   # the swept quantiles start here (lower if not every window is detected there), as AESOP's Table VI
ROC_STEP = 0.5


def tau_of(q: np.ndarray) -> tuple[float, bool]:
    """(conformal tau over the run scores q, whether the rank was capped at the largest)."""
    rank = math.ceil((len(q) + 1) * (1 - ALPHA_RUN))
    return float(np.sort(q)[min(rank, len(q)) - 1]), rank > len(q)


def roc(loo: list[np.ndarray], test: np.ndarray, y: np.ndarray) -> dict:
    """The ROC curve with tau swept over the per-run quantiles and the window peaks (module docstring, roc)."""
    test = test.astype(np.float64)   # float32 scores would round "just below a peak" back onto the peak
    neg = test[y < 0]
    peaks = np.array([test[y == w].max() for w in np.unique(y[y >= 0])])
    rate = lambda t: (float((peaks > t).mean()), float((neg > t).mean()))  # noqa: E731
    pcts = [round(float(p), 2) for p in np.arange(0, 100 + ROC_STEP / 2, ROC_STEP)]
    tau_p = {p: tau_of(np.array([np.percentile(v, p, method="higher") for v in loo]))[0] for p in pcts}
    p_lo = max((p for p in pcts if p <= ROC_START and rate(tau_p[p])[0] == 1.0), default=0.0)
    eps = 1e-9 * max(1.0, float(np.abs(peaks).max()))
    lo, hi = min(tau_p[p_lo], float(peaks.min()) - eps), float(peaks.max())
    cand = [(tau_p[p], p) for p in pcts if p >= p_lo] + [(float(pk) + d, None) for pk in peaks for d in (0.0, -eps)]
    pts = []
    for t, p in sorted({c for c in cand if lo <= c[0] <= hi}, key=lambda c: (-c[0], c[1] is None, -(c[1] or 0))):
        tpr, fpr = rate(t)
        pts.append({"tau": t, "p": p, "TPR": tpr, "FPR": fpr, "accuracy": (tpr + 1 - fpr) / 2})
    x = [0.0] + [r["FPR"] for r in pts] + [1.0]   # tau falling: both rates rise; closed where they stay 0 / 1
    t = [0.0] + [r["TPR"] for r in pts] + [1.0]
    auroc = float(sum((x[i + 1] - x[i]) * (t[i + 1] + t[i]) / 2 for i in range(len(x) - 1)))
    return {"p_lo": p_lo, "AUROC": auroc, "points": pts}


class Embedder:
    """One text model's vectors: the pool's ground-truth points and the test runs' points per source, with every
    cosine similarity precomputed, so any subset of the cache is a mask."""

    def __init__(self, data: Path, emb: str, ref: list[str], tests: list[str]):
        vecs = [load(data, n, emb, "gt") for n in ref]
        self.X = np.concatenate(vecs)
        self.groups = np.concatenate([np.full(len(v), i) for i, v in enumerate(vecs)])
        self.tests = {src: [load(data, n, emb, src) for n in tests] for src in TEST_SOURCES}
        X = torch.nn.functional.normalize(_t(self.X).float(), dim=-1)
        self.pp = X @ X.T
        self.pt = {src: torch.nn.functional.normalize(_t(np.concatenate(v)).float(), dim=-1) @ X.T for src, v in self.tests.items()}

    @staticmethod
    def _knn(sim: torch.Tensor) -> dict[str, np.ndarray]:
        top = sim.topk(max(KS), dim=1).values
        return {f"cosine_k{k}": (1 - top[:, :k].mean(1)).cpu().numpy() for k in KS}

    def scores(self, runs: list[int], mask: np.ndarray, metrics: list[str]) -> tuple[dict, dict]:
        """({metric: [leave-one-out scores of each run in runs]}, {metric: {source: test scores}}): the cache is the
        points in mask of the runs in runs; a run's own points are left out of its cache."""
        cache = mask & np.isin(self.groups, runs)
        cols = torch.from_numpy(np.flatnonzero(cache)).to(DEV)
        cgroups = torch.from_numpy(self.groups[cache]).to(DEV)
        loo = {m: [] for m in metrics}
        for r in runs:
            rows = torch.from_numpy(np.flatnonzero(self.groups == r)).to(DEV)
            s = self.pp[rows][:, cols].clone()
            s[:, cgroups == r] = -torch.inf
            for m, v in self._knn(s).items():
                if m in loo:
                    loo[m].append(v)
        test = {m: {} for m in metrics}
        for src, pt in self.pt.items():
            for m, v in self._knn(pt[:, cols]).items():
                if m in test:
                    test[m][src] = v
        if "mahalanobis" in metrics:
            X = _t(self.X).double()
            for r in runs:
                loo["mahalanobis"].append(_maha(*_ledoit_wolf(X[_t(cache & (self.groups != r))]), X[_t(self.groups == r)]))
            fit = _ledoit_wolf(X[_t(cache)])
            test["mahalanobis"] = {src: _maha(*fit, _t(np.concatenate(v)).double()) for src, v in self.tests.items()}
        return loo, test


def draws_of(ref: list[str], n: int, pool: str, rng: random.Random) -> list[list[int]]:
    """DRAWS random sets of n pool runs (indices into ref); the included pool takes Town01 and Town02 runs equally."""
    idx = list(range(len(ref)))
    if n >= len(ref):
        return [idx]
    out = []
    for _ in range(DRAWS):
        if pool == "included":
            t1 = [i for i in idx if ref[i].startswith("t1_")]
            t2 = [i for i in idx if not ref[i].startswith("t1_")]
            out.append(sorted(rng.sample(t1, n // 2) + rng.sample(t2, n - n // 2)))
        else:
            out.append(sorted(rng.sample(idx, n)))
    return out


def experiment(data: Path, dataset: str, pool: str, embedders: list[str]) -> dict:
    S = json.loads((data / "_results" / f"score_{pool}.json").read_text())
    ref, tests, wins = S["reference_runs"], S["test_runs"], S["windows"]
    cfg = json.loads((ROOT / "config" / f"data_{dataset}.json").read_text())
    kinds = {r["run_name"]: r["anomaly_kind"] for r in cfg["anomaly_runs"]}
    rows = {n: [json.loads(l) for l in (data / n / "corpus" / "frames.jsonl").read_text().splitlines()] for n in ref + tests}
    # the test points' window ids, concatenated in test-run order (the same for both sources)
    y, first = [], 0
    for n in tests:
        yi, first = window_ids(np.array([r["t"] for r in rows[n]]), wins.get(n, []), first)
        y.append(yi)
    y = np.concatenate(y)

    def evaluate(loo: list[np.ndarray], test: dict, pct: float = 100 * (1 - ALPHA_FRAME)) -> dict:
        tau, capped = tau_of(np.array([np.percentile(v, pct, method="higher") for v in loo]))
        return {src: {**metrics_at(tau, t, y), "capped": capped} for src, t in test.items()}

    out = {"pool": pool, "reference_runs": ref, "test_runs": tests, "windows": wins,
           "protocol": {**S["protocol"], "pct": 100 * (1 - ALPHA_FRAME), "sizes": list(SIZES), "draws": DRAWS, "fracs": list(FRACS),
                        "roc_start": ROC_START, "roc_step": ROC_STEP, "scoring": "per injection"},
           "setup": {"per_run": {}}, "full": {}, "per_anomaly": {}, "pool_sweep": {}, "corpus_sweep": {}, "roc": {}, "threshold_sweep": {}}
    for src, field in (("gt", "gt"), ("owl", "raw")):
        seen, per = set(), []
        for i, n in enumerate(ref, 1):
            sets = [set(r[field]) for r in rows[n]]
            seen |= set().union(*sets)
            per.append({"i": i, "run": n, "per_frame": float(np.mean([len(s) for s in sets])), "cumulative_distinct": len(seen)})
        out["setup"]["per_run"][src] = per
    pool_concepts = set().union(*(set(r["gt"]) for n in ref for r in rows[n]))
    for emb in embedders:
        for metric in METRICS:
            m = S["embedders"][emb][metric]
            for src in TEST_SOURCES:
                key = f"{src}@{emb}/{metric}"
                out["full"][key] = {**m["metrics"][src]["total"]["conformal"], "tau": m["tau"]}
                out["per_anomaly"][key] = {kind: m["metrics"][src][kind]["conformal"] for kind in m["metrics"][src] if kind != "total"}
        E = Embedder(data, emb, ref, tests)
        rng = random.Random(0)
        everything = np.ones(len(E.X), bool)
        for n in SIZES:
            for d, runs in enumerate(draws_of(ref, n, pool, rng)):
                loo, test = E.scores(runs, everything, METRICS)
                for metric in METRICS:
                    for src, r in evaluate(loo[metric], test[metric]).items():
                        out["pool_sweep"].setdefault(f"{src}@{emb}/{metric}", []).append({"n_runs": n, "draw": d, **r})
            print(f"{pool} / {emb}: pool size {n} done", flush=True)
        all_runs = list(range(len(ref)))
        for frac in FRACS:
            for d in range(DRAWS if frac < 1 else 1):
                mask = np.zeros(len(E.X), bool)
                mask[rng.sample(range(len(E.X)), round(frac * len(E.X)))] = True
                kept = [r for n in ref for r in rows[n]]
                coverage = len(set().union(*(set(kept[i]["gt"]) for i in np.flatnonzero(mask)))) / len(pool_concepts)
                metrics = [m for m in METRICS if m != "mahalanobis"]
                loo, test = E.scores(all_runs, mask, metrics)
                for metric in metrics:
                    for src, r in evaluate(loo[metric], test[metric]).items():
                        out["corpus_sweep"].setdefault(f"{src}@{emb}/{metric}", []).append({"share": frac, "draw": d, "coverage": coverage, **r})
        loo, test = E.scores(all_runs, everything, METRICS)
        for metric in METRICS:
            for src, t in test[metric].items():
                key = f"{src}@{emb}/{metric}"
                c = out["roc"][key] = roc(loo[metric], t, y)
                out["full"][key]["AUROC_mw"] = out["full"][key]["AUROC"]
                out["full"][key]["AUROC"] = c["AUROC"]
                out["threshold_sweep"][key] = [{"pct": r["p"], "TPR": r["TPR"], "FPR": r["FPR"], "accuracy": r["accuracy"]}
                                               for r in c["points"] if r["p"] is not None]
        print(f"{pool} / {emb}: done", flush=True)
        del E
        torch.cuda.empty_cache()
    out["kinds"] = {n: kinds[n] for n in tests if n in kinds}
    (data / "_results" / f"experiment_{pool}.json").write_text(json.dumps(finite(out)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=P.DATASET)
    ap.add_argument("--pools", default="withheld,included")
    ap.add_argument("--embedders", default=",".join(TEXT_EMBEDDERS))
    args = ap.parse_args()
    data = ROOT / "data" / args.dataset
    for pool in args.pools.split(","):
        experiment(data, args.dataset, pool, args.embedders.split(","))


if __name__ == "__main__":
    main()
