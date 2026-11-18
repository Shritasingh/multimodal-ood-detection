"""The Experiment tab's embedding-space view of the scored semantic24 corpus (build_corpus.py, score.py): where
the points sit, per text model, and how dense the space is.

    .venv/bin/python aesop_recreate/scripts/embed_map.py      # both pools

Points: every pool run's ground-truth prompt points (the cache) and every test run's points twice, from its
ground-truth and from its raw OWL-ViT prompt. Per text model:
  map      2-D coordinates: t-SNE (cosine, perplexity 30, all points together) and PCA (fit on the pool, mean-centred)
  density  cosine distances (1 - cos): RANDOM_PAIRS random pool pairs; each pool point's leave-one-out k5 score and each
           test point's k5 score (score.py's); the conformal tau
  stats    effective dimensionality of the pool (participation ratio of the PCA spectrum), the share of variance in
           the first two components, the mean cosine of random pool pairs (how bunched the space is)
Writes _results/embed_map_<pool>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT.parent), str(ROOT.parent / "scripts")]
from encoders.text_embedders import TEXT_EMBEDDERS  # noqa: E402
from metrics import window_ids  # noqa: E402
from score import load  # noqa: E402
from sim import semantic24 as P  # noqa: E402

RANDOM_PAIRS = 5000
METRIC = "cosine_k5"


def embed_map(data: Path, pool: str, embedders: list[str]) -> dict:
    S = json.loads((data / "_results" / f"score_{pool}.json").read_text())
    ref, tests, wins = S["reference_runs"], S["test_runs"], S["windows"]
    rows = {n: [json.loads(l) for l in (data / n / "corpus" / "frames.jsonl").read_text().splitlines()] for n in ref + tests}
    scores = {n: list(csv.DictReader((data / n / "corpus" / f"scores_{pool}.csv").open())) for n in ref + tests}
    # the points, in this order for every model: pool (gt), then each test run's gt, then each test run's raw
    points = [{"run": n, "t": r["t"], "src": "gt", "group": "pool", "concepts": r["gt"]} for n in ref for r in rows[n]]
    for src, field in (("gt", "gt"), ("owl", "raw")):
        for n in tests:
            y, _ = window_ids(np.array([r["t"] for r in rows[n]]), wins.get(n, []))
            points += [{"run": n, "t": r["t"], "src": src, "concepts": r[field],
                        "group": "anomaly window" if yi >= 0 else "test nominal"} for r, yi in zip(rows[n], y)]
    out = {"pool": pool, "points": points, "embedders": {}}
    rng = np.random.default_rng(0)
    n_pool = sum(len(rows[n]) for n in ref)
    for emb in embedders:
        X = np.concatenate([load(data, n, emb, "gt") for n in ref] + [load(data, n, emb, s) for s in ("gt", "owl") for n in tests])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        P = X[:n_pool]
        pca = PCA().fit(P)
        ev = pca.explained_variance_
        xy = pca.transform(X)[:, :2]
        ts = TSNE(2, metric="cosine", perplexity=30, init="pca", random_state=0).fit_transform(X)
        i, j = rng.integers(0, n_pool, RANDOM_PAIRS), rng.integers(0, n_pool, RANDOM_PAIRS)
        keep = i != j
        rand = 1 - (P[i[keep]] * P[j[keep]]).sum(1)
        score = lambda n, src: [float(r[f"{src}@{emb}/{METRIC}"]) for r in scores[n]]  # noqa: E731
        k5 = [v for n in ref for v in score(n, "gt")] + [v for s in ("gt", "owl") for n in tests for v in score(n, s)]
        out["embedders"][emb] = {
            "tsne": np.round(ts, 3).tolist(), "pca": np.round(xy, 5).tolist(), "k5": np.round(k5, 6).tolist(),
            "random_pairs": np.round(rand, 6).tolist(), "tau": S["embedders"][emb][METRIC]["tau"],
            "stats": {"dim": int(X.shape[1]), "effective_dim": float(ev.sum() ** 2 / (ev ** 2).sum()),
                      "pc12_share": float(pca.explained_variance_ratio_[:2].sum()), "mean_random_cos": float(1 - rand.mean()),
                      "median_random_dist": float(np.median(rand)), "median_pool_k5": float(np.median(k5[:n_pool]))}}
        print(f"{pool} / {emb}: effective dim {out['embedders'][emb]['stats']['effective_dim']:.1f}, "
              f"mean random cos {out['embedders'][emb]['stats']['mean_random_cos']:.4f}", flush=True)
    (data / "_results" / f"embed_map_{pool}.json").write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=P.DATASET)
    ap.add_argument("--pools", default="withheld,included")
    ap.add_argument("--embedders", default=",".join(TEXT_EMBEDDERS))
    args = ap.parse_args()
    for pool in args.pools.split(","):
        embed_map(ROOT / "data" / args.dataset, pool, args.embedders.split(","))


if __name__ == "__main__":
    main()
