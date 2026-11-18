"""Scores, conformal thresholds and metrics of the semantic24 corpus (scripts/build_corpus.py), per pool and text model.

    .venv/bin/python aesop_recreate/scripts/score.py                    # both pools
    .venv/bin/python aesop_recreate/scripts/score.py --pools withheld

Cache = the pool's nominal runs, encoded from their (corrected) ground truth. Scores, per point:
  cosine_k<k>   mean cosine distance (1 - cos) to the k nearest cache points (k = 1, 3, 5, 10)
  mahalanobis   distance to a Ledoit-Wolf Gaussian of the cache
Pool runs are scored run-level leave-one-out (against the cache of the other pool runs); test runs against the whole
cache, twice: their raw OWL-ViT prompts (owl@) and their ground-truth prompts (gt@).

Threshold, run-level conformal (UNISafe-style, Seo et al. 2025): Q_r = the (1 - ALPHA_FRAME) quantile of pool run r's
leave-one-out scores; tau = the ceil((N + 1)(1 - ALPHA_RUN))-th smallest of the N Q_r (N = 20: the largest). For
comparison the results also keep AESOP's tau: the 95th percentile of all pool points' leave-one-out scores pooled.
Metrics as Elhafsi et al. / AESOP at the conformal tau: each anomaly window is one positive, detected if any of its points
scores above tau; every other test point one negative; accuracy = (TPR + TNR) / 2; "total" = t2_test_nominal + every
anomaly run, per anomaly kind = t2_test_nominal + that run. Windows: config/data_<dataset>.json (the ground truth's).

Writes, per pool (withheld / included):
  <run>/corpus/scores_<pool>.csv    tick, <source>@<embedder>/<metric> (pool runs: gt; test runs: owl and gt)
  <run>/corpus/knn_<pool>.json      {source: {embedder: {t: [[run, t, cos], ...]}}}: the 5 nearest cache points (leave-one-out for pool runs)
  _results/score_<pool>.json        per embedder and metric: Q_r, tau (conformal, AESOP), per test source the metrics
                                    (total / per anomaly) and per run the fraction of points above tau
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT.parent), str(ROOT.parent / "scripts")]
from encoders.text_embedders import TEXT_EMBEDDERS  # noqa: E402
from metrics import DEV, mahalanobis_scores, metrics_at, window_ids  # noqa: E402
from sim import semantic24 as P  # noqa: E402

KS = (1, 3, 5, 10)
METRICS = [f"cosine_k{k}" for k in KS] + ["mahalanobis"]
KNN_SHOWN = 5
ALPHA_FRAME = 0.05
ALPHA_RUN = 0.05
AESOP_PCT = 95.0
TEST_SOURCES = ("owl", "gt")   # test runs are scored from their raw OWL-ViT concepts and from their ground truth


def load(data: Path, name: str, emb: str, src: str) -> np.ndarray:
    return np.load(data / name / "corpus" / f"{emb}@{src}.npy")


def nearest(cache: np.ndarray, query: np.ndarray, groups: np.ndarray | None, q_group: int | None, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(cosine similarities, cache indices) of the k nearest cache points of every query point; cache points of the
    query's own run (groups == q_group) are left out."""
    X = torch.nn.functional.normalize(torch.from_numpy(cache).to(DEV).float(), dim=-1)
    Q = torch.nn.functional.normalize(torch.from_numpy(query).to(DEV).float(), dim=-1)
    s = Q @ X.T
    if q_group is not None:
        s[:, torch.from_numpy(groups == q_group).to(DEV)] = -torch.inf
    top = s.topk(k, dim=1)
    return top.values.cpu().numpy(), top.indices.cpu().numpy()


def conformal_tau(q: np.ndarray, alpha: float) -> float:
    """The ceil((N + 1)(1 - alpha))-th smallest of the N run scores (inf if that rank exceeds N)."""
    rank = math.ceil((len(q) + 1) * (1 - alpha))
    return float(np.sort(q)[rank - 1]) if rank <= len(q) else math.inf


def finite(x):
    """JSON without Infinity / NaN (an unreachable conformal rank gives tau = inf): those become null."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def score_pool(data: Path, dataset: str, pool: str, embedders: list[str]) -> dict:
    plans = P.plan()
    ref = [p.name for p in plans if pool in p.pools and (data / p.name / "corpus" / "frames.jsonl").exists()]
    tests = [p.name for p in plans if not p.pools and (data / p.name / "corpus" / "frames.jsonl").exists()]
    cfg = json.loads((ROOT / "config" / f"data_{dataset}.json").read_text())
    wins = {r["run_name"]: r.get("windows", [[r["onset_tick"], r["offset_tick"]]]) for r in cfg["anomaly_runs"]}
    kinds = {r["run_name"]: r["anomaly_kind"] for r in cfg["anomaly_runs"]}
    scores = {n: {} for n in ref + tests}   # run -> {"<src>@<emb>/<metric>": per-point scores}
    knn = {n: {s: {} for s in (("gt",) if n in ref else TEST_SOURCES)} for n in ref + tests}
    ticks = {n: [json.loads(l)["t"] for l in (data / n / "corpus" / "frames.jsonl").read_text().splitlines()] for n in ref + tests}
    results = {"pool": pool, "reference_runs": ref, "test_runs": tests, "windows": wins,
               "protocol": {"alpha_frame": ALPHA_FRAME, "alpha_run": ALPHA_RUN, "aesop_pct": AESOP_PCT, "stride": P.STRIDE,
                            "ks": list(KS), "cache_source": "gt", "test_sources": list(TEST_SOURCES)}, "embedders": {}}
    for emb in embedders:
        vecs = [load(data, n, emb, "gt") for n in ref]
        cache = np.concatenate(vecs)
        groups = np.concatenate([np.full(len(v), i) for i, v in enumerate(vecs)])
        index = [(n, t) for n in ref for t in ticks[n]]
        queries = [(n, "gt", v, i) for i, (n, v) in enumerate(zip(ref, vecs))]
        queries += [(n, src, load(data, n, emb, src), None) for src in TEST_SOURCES for n in tests]
        test_q = [q for q in queries if q[3] is None]
        maha_loo, maha_test = mahalanobis_scores(cache, np.concatenate([v for _, _, v, _ in test_q]), groups)
        off = 0
        for n, src, v, gi in queries:
            sim, idx = nearest(cache, v, groups, gi, max(KS))
            key = f"{src}@{emb}"
            for k in KS:
                scores[n][f"{key}/cosine_k{k}"] = 1 - sim[:, :k].mean(1)
            if gi is None:
                scores[n][f"{key}/mahalanobis"] = maha_test[off : off + len(v)]
                off += len(v)
            else:
                scores[n][f"{key}/mahalanobis"] = maha_loo[groups == gi]
            knn[n][src][emb] = {str(t): [[*index[j], round(float(c), 4)] for j, c in zip(idx[i, :KNN_SHOWN], sim[i, :KNN_SHOWN])]
                                for i, t in enumerate(ticks[n])}
        results["embedders"][emb] = {}
        for metric in METRICS:
            q = np.array([np.quantile(scores[n][f"gt@{emb}/{metric}"], 1 - ALPHA_FRAME, method="higher") for n in ref])
            tau = conformal_tau(q, ALPHA_RUN)
            tau_aesop = float(np.percentile(np.concatenate([scores[n][f"gt@{emb}/{metric}"] for n in ref]), AESOP_PCT))
            out = {"Q": dict(zip(ref, q.round(6).tolist())), "tau": tau, "tau_aesop": tau_aesop, "frac_above": {}, "metrics": {}}
            neg = [n for n in tests if n not in wins]
            for src in TEST_SOURCES:
                key = f"{src}@{emb}/{metric}"
                out["frac_above"][src] = {n: float((scores[n][key] > tau).mean()) for n in (tests if src != "gt" else ref + tests)}
                out["metrics"][src] = {}
                for name, group in [("total", tests)] + [(kinds[a], neg + [a]) for a in tests if a in wins]:
                    ys, first, test = [], 0, []
                    for n in group:
                        y, first = window_ids(np.array(ticks[n]), wins.get(n, []), first)
                        ys.append(y)
                        test.append(scores[n][key])
                    y, test = np.concatenate(ys), np.concatenate(test)
                    out["metrics"][src][name] = {"conformal": metrics_at(tau, test, y), "aesop": metrics_at(tau_aesop, test, y)}
            results["embedders"][emb][metric] = out
        m = results["embedders"][emb]["cosine_k5"]
        print(f"{pool} / {emb}: tau(cosine_k5) {m['tau']:.4f} conformal, {m['tau_aesop']:.4f} AESOP; total at the conformal tau: "
              + "; ".join(f"{src} TPR {m['metrics'][src]['total']['conformal']['TPR']:.2f} FPR {m['metrics'][src]['total']['conformal']['FPR']:.3f}"
                          for src in TEST_SOURCES), flush=True)
    for n in ref + tests:
        with open(data / n / "corpus" / f"scores_{pool}.csv", "w", newline="") as f:
            w = csv.writer(f)
            cols = sorted(scores[n])
            w.writerow(["tick"] + cols)
            for i, t in enumerate(ticks[n]):
                w.writerow([t] + [f"{scores[n][c][i]:.6g}" for c in cols])
        (data / n / "corpus" / f"knn_{pool}.json").write_text(json.dumps(knn[n]))
    (data / "_results").mkdir(exist_ok=True)
    (data / "_results" / f"score_{pool}.json").write_text(json.dumps(finite(results), indent=1))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=P.DATASET)
    ap.add_argument("--pools", default="withheld,included")
    ap.add_argument("--embedders", default=",".join(TEXT_EMBEDDERS))
    args = ap.parse_args()
    data = ROOT / "data" / args.dataset
    for pool in args.pools.split(","):
        score_pool(data, args.dataset, pool, args.embedders.split(","))


if __name__ == "__main__":
    main()
