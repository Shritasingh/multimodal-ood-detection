"""Frame-level evaluation of the semantic24 encoders for one reference pool (the experiment plan's protocol).

    .venv/bin/python aesop_recreate/scripts/evaluate_semantic24.py aesop_recreate/config/encoders_semantic24.json        # withheld routes
    .venv/bin/python aesop_recreate/scripts/evaluate_semantic24.py aesop_recreate/config/encoders_semantic24_incl.json   # included routes
    ... --series gt@minilm/cosine_k5,clip_image/cosine_k5     # only some series

Per series (encoder / scoring function):
  cache     every nominal frame (ticks 0, STRIDE, ...) of the pool's reference runs
  tau       PCT-th percentile of the cache's run-level leave-one-out scores (each pool run's frames scored against the
            cache of the other pool runs, as an unseen test run is; a frame-level leave-one-out keeps the frame's
            near-duplicate neighbours from its own run in the cache and gave tau too low, FPR ~0.3 for every encoder)
  test      the 4 long test runs (t2_test_nominal + the anomaly runs, P.INJECTIONS injections each); a frame is flagged
            when its score > tau. Scored as Elhafsi et al. 2023 (the CARLA data AESOP reports on): each injection's
            window (the ticks it is on screen) is one anomaly, detected if any of its frames is flagged; every frame
            outside the windows is one nominal observation
  TPR = detected windows / windows, FPR = flagged nominal frames / nominal frames, TNR = 1 - FPR,
  accuracy = (TPR + TNR) / 2 (AESOP's accuracy), AUROC = windows (their highest frame score, the one that decides
  detection) vs nominal frames
Scoring functions: cosine_k<N> (mean cosine distance to the N nearest cache frames), mahalanobis (to the cache mean,
Ledoit-Wolf covariance, re-fitted without each left-out run), patch (vision_patch: mean distance of a frame's worst
TOP_FRAC patches to their nearest cache patch).
Also: the same metrics as tau sweeps from the 50th to the 100th percentile (THRESH_PCTS), and the whole protocol on pools
of SIZES runs (from 2: run-level leave-one-out needs another run; DRAWS random draws each; the included pool draws Town01 and Town02 runs in equal numbers), recording each
draw's distinct concepts. Setup, per concept source, adding the pool's runs one at a time in config order: each run's
concepts per frame and the cumulative distinct concepts in the nominal cache after it (--setup-only recomputes just this
into an existing results file). Corpus sweep: the cache is a random FRACS share of the
pool's nominal frames (DRAWS draws each), the rest of the protocol unchanged; each draw also records the share of the
source's distinct concepts its frames cover (--corpus-only adds just this sweep to an existing results file).
Per anomaly (--per-anomaly-only recomputes it): the same metrics at the full pool's tau with the test set cut to
t2_test_nominal plus that one anomaly run. Cross-source series (--cross-only adds them to an existing results file and the test runs' score files): <source>_gtc@<embedder>
scores a detector's descriptions (Qwen closed / open) against the ground truth's nominal cache; tau comes from the pool
runs' own detector descriptions, each run against the ground-truth cache of the other runs. Writes data/semantic24/_results/<output_stem>_eval.json and,
for a run over every series, each test run's per-frame scores against the whole pool to <out_path>/<output_stem>.csv
(the dashboard's Scores tab).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
MAIN = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(MAIN), str(MAIN / "scripts")]
import run_metrics as rm  # noqa: E402
from run_concepts import concepts_of, load_overrides  # noqa: E402
from sim import semantic24 as P  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PCT = 95.0
SIZES = (2, 5, 10, 15, 20)
DRAWS = 5
TOP_FRAC = 0.05
CROSS = {"qwen_closed_gtc": "qwen_closed", "qwen_open_gtc": "qwen_open"}   # cross-source series -> its query source
FRACS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
THRESH_PCTS = [50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5, 100]
OUT = ROOT / "data" / "semantic24_long" / "_results"


def source_of(name: str) -> str:
    """Concept source of an encoder: its part before '@'; the image encoders count the ground truth's concepts."""
    return name.split("@")[0] if "@" in name else "gt"


class Runs:
    """Every run's embeddings per encoder (loaded once), concepts per source, and the test labels."""

    def __init__(self, cfg: dict):
        self.ref = [r["run_name"] for r in cfg["reference_runs"]]
        self.test = [r["run_name"] for r in cfg["negative_runs"] + cfg["anomaly_runs"]]
        self.path = {r["run_name"]: r["out_path"] for r in cfg["reference_runs"] + cfg["negative_runs"] + cfg["anomaly_runs"]}
        self.window = {r["run_name"]: [tuple(w) for w in r.get("windows", [[r["onset_tick"], r["offset_tick"]]])] for r in cfg["anomaly_runs"]}
        self.n_ticks = {p.name: p.n_ticks for p in P.plan()}
        self.emb, self.concepts = {}, {}

    def embeddings(self, name: str, run: str) -> np.ndarray:
        if (name, run) not in self.emb:
            d = rm.load_npz(self.path[run], name + ("_nominal" if run in self.ref else ""))
            self.emb[name, run] = (d["embeddings"].astype(np.float16 if d["embeddings"].ndim == 3 else np.float32), d["ticks"])
        return self.emb[name, run][0]

    def labels(self, name: str) -> np.ndarray:
        """Per test frame (self.test order), the index of the anomaly window it is in, else -1."""
        out, k = [], 0
        for run in self.test:
            self.embeddings(name, run)
            y, k = window_ids(self.emb[name, run][1], self.window.get(run, []), k)
            out.append(y)
        return np.concatenate(out)

    def concept_lists(self, source: str, run: str) -> list[list[str]]:
        if (source, run) not in self.concepts:
            c = concepts_of(run, source, load_overrides())
            self.concepts[source, run] = [c.get(t, []) for t in range(0, self.n_ticks[run], P.STRIDE)]
        return self.concepts[source, run]


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV)


def cosine_scores(cache: np.ndarray, query: np.ndarray | None, ks: list[int], groups: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """{k: mean cosine distance to the k nearest cache frames}; query None = the cache itself, each frame against the
    cache frames of other runs (groups: run index of every cache frame)."""
    X = torch.nn.functional.normalize(_t(cache).float(), dim=-1)
    Q = X if query is None else torch.nn.functional.normalize(_t(query).float(), dim=-1)
    g = None
    if query is None:
        g = _t(groups)
        kmax = min(max(ks), int(len(X) - np.bincount(groups).max()))
    else:
        kmax = min(max(ks), len(X))
    tops = []
    for i in range(0, len(Q), 2048):
        s = Q[i : i + 2048] @ X.T
        if g is not None:
            s[g[i : i + 2048, None] == g[None, :]] = -torch.inf
        tops.append(s.topk(kmax, dim=1).values)
    top = torch.cat(tops)
    return {k: (1 - top[:, : min(k, kmax)].mean(1)).cpu().numpy() for k in ks}


def _ledoit_wolf(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean, inverse covariance) of a Ledoit-Wolf fit (sklearn's shrinkage formula), float64 on DEV."""
    n, p = X.shape
    m = X.mean(0)
    Xc = X - m
    S = Xc.T @ Xc / n
    X2 = Xc * Xc
    tr = X2.sum(0) / n
    mu = tr.sum() / p
    delta = (S * S).sum()
    beta = ((X2.T @ X2).sum() / n - delta) / (p * n)
    delta = (delta - 2 * mu * tr.sum() + p * mu * mu) / p
    shrink = 0.0 if beta == 0 else float(min(beta, delta) / delta)
    return m, torch.linalg.inv((1 - shrink) * S + shrink * mu * torch.eye(p, dtype=X.dtype, device=X.device))


def _maha(m: torch.Tensor, inv: torch.Tensor, Q: torch.Tensor) -> np.ndarray:
    D = Q - m
    return torch.sqrt(((D @ inv) * D).sum(1).clamp(min=0)).cpu().numpy()


def mahalanobis_scores(cache: np.ndarray, query: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(run-level leave-one-out scores of the cache, scores of the query) under a Ledoit-Wolf Gaussian of the cache."""
    X, Q = _t(cache).double(), _t(query).double()
    test = _maha(*_ledoit_wolf(X), Q)
    loo = np.empty(len(X))
    for gi in np.unique(groups):
        own = groups == gi
        loo[own] = _maha(*_ledoit_wolf(X[_t(~own)]), X[_t(own)])
    return loo, test


def patch_scores(cache: np.ndarray, query: np.ndarray | None, groups: np.ndarray | None = None) -> np.ndarray:
    """vision_patch: per frame, mean distance of its worst TOP_FRAC patches to their nearest cache patch
    (cache (F, N, D)); query None = the cache itself, each frame against the patches of other runs' frames."""
    F, N, D = cache.shape
    bank = torch.nn.functional.normalize(_t(cache.reshape(-1, D)).float(), dim=-1).half()
    Q = cache if query is None else query
    worst = max(1, round(TOP_FRAC * N))
    out = []
    for j in range(len(Q)):
        s = torch.nn.functional.normalize(_t(Q[j]).float(), dim=-1).half() @ bank.T
        if query is None:
            own = np.flatnonzero(groups == groups[j])          # the frame's run is contiguous in the cache
            s[:, own[0] * N : (own[-1] + 1) * N] = -torch.inf
        out.append(float((1 - s.max(1).values.float()).topk(worst).values.mean()))
    del bank
    return np.array(out)


def metrics(loo: np.ndarray, test: np.ndarray, y: np.ndarray, pct: float = PCT) -> dict:
    return metrics_at(float(np.percentile(loo, pct)), test, y)


def window_ids(ticks: np.ndarray, windows: list, first: int = 0) -> tuple[np.ndarray, int]:
    """(per tick: first + the index of the window [on, off) it is in, else -1; first + the number of windows)."""
    y = np.full(len(ticks), -1)
    for j, (on, off) in enumerate(windows):
        y[(ticks >= on) & (ticks < off)] = first + j
    return y, first + len(windows)


def metrics_at(tau: float, test: np.ndarray, y: np.ndarray) -> dict:
    """Elhafsi et al.'s scoring; y: per frame its anomaly window's index, -1 for nominal frames."""
    neg = y < 0
    peak = np.array([test[y == w].max() for w in np.unique(y[~neg])])   # a window is detected iff its highest score is
    tpr, fpr = float((peak > tau).mean()) if len(peak) else None, float((test[neg] > tau).mean())
    auroc = float(roc_auc_score(np.r_[np.ones(len(peak)), np.zeros(neg.sum())], np.r_[peak, test[neg]])) if len(peak) else None
    return {"tau": tau, "TPR": tpr, "FPR": fpr, "TNR": 1 - fpr, "accuracy": (tpr + 1 - fpr) / 2 if tpr is not None else None,
            "AUROC": auroc, "n_anomalous": len(peak), "n_nominal": int(neg.sum())}


def per_run(runs: Runs, name: str, test: np.ndarray, tau: float) -> dict:
    """Per test run: TPR over its windows, FPR over its other frames."""
    out, i = {}, 0
    for run in runs.test:
        t = runs.emb[name, run][1]
        m = metrics_at(tau, test[i : i + len(t)], window_ids(t, runs.window.get(run, []))[0])
        i += len(t)
        out[run] = {"TPR": m["TPR"], "FPR": m["FPR"]}
    return out


def per_anomaly(runs: Runs, res: dict, stem: str) -> dict:
    """{series: {anomaly run: metrics}} at the series' full-pool tau, the test set being t2_test_nominal plus that one
    anomaly run (its window frames anomalous), read from the test runs' score files."""
    import csv as _csv
    cols = {}
    for run in runs.test:
        rows = list(_csv.DictReader(open(rm.resolve_path(runs.path[run]) / f"{stem}.csv")))
        cols[run] = {c: (np.array([int(r["tick"]) for r in rows if r[c] != ""]), np.array([float(r[c]) for r in rows if r[c] != ""]))
                     for c in rows[0] if c != "tick"}
    nominal = [r for r in runs.test if r not in runs.window]
    out = {}
    for key, m in res["full"].items():
        if not all(key in cols[r] for r in runs.test):
            continue
        for run in runs.window:
            parts = [cols[r][key] for r in nominal + [run]]
            y = np.concatenate([window_ids(t, runs.window.get(r, []))[0] for (t, _), r in zip(parts, nominal + [run])])
            out.setdefault(key, {})[run] = metrics_at(m["tau"], np.concatenate([s for _, s in parts]), y)
    return out


def score_series(runs: Runs, name: str, fns: list[str], pool: list[str], keep: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{scoring fn: (leave-one-out cache scores, test scores)} for one encoder and pool; keep: sorted indices of the
    pool's frames that form the cache (default: all)."""
    parts = [runs.embeddings(name, r) for r in pool]
    cache, groups = np.concatenate(parts), np.repeat(np.arange(len(parts)), [len(x) for x in parts])
    if keep is not None:
        cache, groups = cache[keep], groups[keep]
    test = np.concatenate([runs.embeddings(name, r) for r in runs.test])
    out = {}
    ks = [int(f.split("_k")[1]) for f in fns if f.startswith("cosine_k")]
    if ks:
        loo, tst = cosine_scores(cache, None, ks, groups), cosine_scores(cache, test, ks)
        out.update({f"cosine_k{k}": (loo[k], tst[k]) for k in ks})
    if "mahalanobis" in fns:
        out["mahalanobis"] = mahalanobis_scores(cache, test, groups)
    if "patch" in fns:
        out["patch"] = (patch_scores(cache, None, groups), patch_scores(cache, test))
    return out


def corpus_plan(runs: Runs, rng: random.Random) -> list[tuple[float, int, np.ndarray]]:
    """(share, draw, sorted frame indices) of the pool's nominal frames for every FRACS share (one draw at 100%)."""
    n = sum(len(runs.concept_lists("gt", r)) for r in runs.ref)
    return [(f, d, np.sort(np.array(rng.sample(range(n), round(f * n))))) for f in FRACS for d in range(1 if f >= 1 else DRAWS)]


def corpus_sweep(runs: Runs, series: dict, sources: list[str]) -> dict:
    """{series: [{share, draw, coverage, TPR, FPR, accuracy, AUROC, ...}]}: the cache is a random share of the nominal frames."""
    plan = corpus_plan(runs, random.Random("semantic24-corpus"))
    sources = sorted({CROSS.get(s, s) for s in sources})
    frames = {s: [c for r in runs.ref for c in runs.concept_lists(s, r)] for s in sources}
    distinct = {s: len({c for f in frames[s] for c in f}) for s in sources}
    out = {}
    for name, fns in series.items():
        cross = source_of(name) in CROSS
        src = CROSS.get(source_of(name), source_of(name))
        y = runs.labels(f"{src}@{name.split('@')[1]}" if cross else name)
        for share, d, keep in plan:
            cov = len({c for i in keep for c in frames[src][i]}) / distinct[src]
            for fn, (loo, test) in (cross_series(runs, name, fns, runs.ref, keep) if cross else score_series(runs, name, fns, runs.ref, keep)).items():
                out.setdefault(f"{name}/{fn}", []).append({"share": share, "draw": d, "n_frames": len(keep), "coverage": cov, **metrics(loo, test, y)})
        runs.emb = {k: v for k, v in runs.emb.items() if k[0] != name}
        print(f"corpus sweep: {name}", flush=True)
    return out


def cross_cosine(cache: np.ndarray, cgroups: np.ndarray, query: np.ndarray, qgroups: np.ndarray | None, ks: list[int]) -> dict[int, np.ndarray]:
    """{k: mean cosine distance of each query row to its k nearest cache rows}, skipping cache rows of the query's own run
    (qgroups None: no skipping)."""
    X = torch.nn.functional.normalize(_t(cache).float(), dim=-1)
    Q = torch.nn.functional.normalize(_t(query).float(), dim=-1)
    cg, kmax = _t(cgroups), min(max(ks), int(len(X) - (np.bincount(cgroups).max() if qgroups is not None else 0)))
    tops = []
    for i in range(0, len(Q), 2048):
        sim = Q[i : i + 2048] @ X.T
        if qgroups is not None:
            sim[_t(qgroups[i : i + 2048])[:, None] == cg[None, :]] = -torch.inf
        tops.append(sim.topk(kmax, dim=1).values)
    top = torch.cat(tops)
    return {k: (1 - top[:, : min(k, kmax)].mean(1)).cpu().numpy() for k in ks}


def cross_series(runs: Runs, name: str, fns: list[str], pool: list[str], keep: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{scoring fn: (calibration scores, test scores)} of a cross-source series <source>_gtc@<embedder>: the source's
    descriptions against the ground truth's nominal cache of the pool (keep: sorted indices of the pool's frames to use)."""
    src, emb = name.split("@")
    query = f"{CROSS[src]}@{emb}"
    cparts, qparts = [runs.embeddings(f"gt@{emb}", r) for r in pool], [runs.embeddings(query, r) for r in pool]
    cache, cg = np.concatenate(cparts), np.repeat(np.arange(len(cparts)), [len(x) for x in cparts])
    qnom, qg = np.concatenate(qparts), np.repeat(np.arange(len(qparts)), [len(x) for x in qparts])
    if keep is not None:   # the same frames on both sides: every source has the same ticks
        cache, cg, qnom, qg = cache[keep], cg[keep], qnom[keep], qg[keep]
    test = np.concatenate([runs.embeddings(query, r) for r in runs.test])
    out = {}
    ks = [int(f.split("_k")[1]) for f in fns if f.startswith("cosine_k")]
    if ks:
        cal, tst = cross_cosine(cache, cg, qnom, qg, ks), cross_cosine(cache, cg, test, None, ks)
        out.update({f"cosine_k{k}": (cal[k], tst[k]) for k in ks})
    if "mahalanobis" in fns:
        X, Qn = _t(cache).double(), _t(qnom).double()
        tst = _maha(*_ledoit_wolf(X), _t(test).double())
        cal = np.empty(len(qnom))
        for gi in np.unique(qg):
            cal[qg == gi] = _maha(*_ledoit_wolf(X[_t(cg != gi)]), Qn[_t(qg == gi)])
        out["mahalanobis"] = (cal, tst)
    return out


def setup_per_run(runs: Runs, sources: list[str]) -> dict:
    """{source: [{run, i, per_frame, cumulative_distinct}]} with the pool's runs added one at a time in config order."""
    out = {}
    for source in sources:
        seen, rows = set(), []
        for i, r in enumerate(runs.ref, 1):
            frames = runs.concept_lists(source, r)
            seen |= {c for f in frames for c in f}
            rows.append({"run": r, "i": i, "per_frame": float(np.mean([len(f) for f in frames])), "cumulative_distinct": len(seen)})
        out[source] = rows
    return out


def draws(pool: list[str], included: bool, rng: random.Random) -> list[tuple[int, int, list[str]]]:
    """(size, draw, runs) for every pool size; the included pool takes Town01 and Town02 runs in equal numbers."""
    t1, t2 = [r for r in pool if r.startswith("t1_")], [r for r in pool if r.startswith("t2_")]
    out = []
    for n in SIZES:
        if n >= len(pool):
            out.append((len(pool), 0, list(pool)))
            continue
        for d in range(DRAWS):
            if included:
                a = n // 2 + (n % 2) * rng.randint(0, 1)
                out.append((n, d, rng.sample(t1, a) + rng.sample(t2, n - a)))
            else:
                out.append((n, d, rng.sample(pool, n)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--series", help="comma-separated encoder/metric (default: every series in the config)")
    ap.add_argument("--setup-only", action="store_true", help="recompute only the per-run setup of an existing results file")
    ap.add_argument("--corpus-only", action="store_true", help="add only the corpus sweep to an existing results file")
    ap.add_argument("--cross-only", action="store_true", help="add only the cross-source series (Qwen vs the ground-truth cache)")
    ap.add_argument("--corpus-cross-only", action="store_true", help="add the cross-source series' corpus sweep to an existing results file")
    ap.add_argument("--per-anomaly-only", action="store_true", help="recompute only the per-anomaly metrics of an existing results file")
    args = ap.parse_args()
    enc_cfg = rm.load_config(rm.resolve_path(args.config))   # the encoder config merged with its data config's runs
    runs = Runs(enc_cfg)
    included = any(r.startswith("t2_nominal") for r in runs.ref)
    series = {name: p["metrics"] for name, p in enc_cfg["encoders"].items()}
    if args.series:
        want = [s.split("/") for s in args.series.split(",")]
        series = {n: [m for w, m in want if w == n] for n in {w for w, _ in want}}
    plan = draws(runs.ref, included, random.Random(f"semantic24-eval:{included}"))
    sources = sorted({source_of(n) for n in series})
    path = OUT / f"{enc_cfg['output_stem']}_eval.json"
    if args.per_anomaly_only:
        res = json.loads(path.read_text())
        res["per_anomaly"] = per_anomaly(runs, res, enc_cfg["output_stem"])
        path.write_text(json.dumps(res, indent=1, default=float))
        return print(f"wrote the per-anomaly metrics into {path}")
    if args.setup_only:
        res = json.loads(path.read_text())
        res["setup"]["per_run"] = setup_per_run(runs, sources)
        res["setup"].pop("spread", None)
        path.write_text(json.dumps(res, indent=1, default=float))
        return print(f"wrote the per-run setup into {path}")
    if args.corpus_cross_only:
        res = json.loads(path.read_text())
        fns = series.get("qwen_closed@e5_mistral", ["cosine_k1", "cosine_k3", "cosine_k5", "cosine_k10", "mahalanobis"])
        cross = {f"{src}@{e}": fns for src in CROSS for e in sorted({n.split("@")[1] for n in series if "@" in n})}
        res.setdefault("corpus_sweep", {}).update(corpus_sweep(runs, cross, sorted(CROSS)))
        path.write_text(json.dumps(res, indent=1, default=float))
        return print(f"wrote the cross-source corpus sweep into {path}")
    if args.cross_only:
        res = json.loads(path.read_text())
        fns = series.get("qwen_closed@e5_mistral", ["cosine_k1", "cosine_k3", "cosine_k5", "cosine_k10", "mahalanobis"])
        emb = sorted({n.split("@")[1] for n in series if "@" in n})
        added = {run: {} for run in runs.test}
        for src, qsrc in CROSS.items():
            for e in emb:
                name, query = f"{src}@{e}", f"{qsrc}@{e}"
                y = runs.labels(query)
                for size, d, pool in plan:
                    for fn, (cal, test) in cross_series(runs, name, fns, pool).items():
                        key, m = f"{name}/{fn}", metrics(cal, test, y)
                        if size == len(runs.ref):
                            res["full"][key] = m
                            res["per_run"][key] = per_run(runs, query, test, m["tau"])
                            res["threshold_sweep"][key] = [{"pct": p, **metrics(cal, test, y, p)} for p in THRESH_PCTS]
                            i = 0
                            for run in runs.test:
                                t = runs.emb[query, run][1]
                                added[run][key] = (t.astype(int).tolist(), test[i : i + len(t)].tolist())
                                i += len(t)
                        res["pool_sweep"].setdefault(key, [])
                        res["pool_sweep"][key] = [r for r in res["pool_sweep"][key] if (r["n_runs"], r["draw"]) != (size, d)] + [{"n_runs": size, "draw": d, "runs": pool, **m}]
                print(f"cross: {name}  " + "  ".join(f"{fn} acc {res['full'][f'{name}/{fn}']['accuracy']:.3f} AUROC {res['full'][f'{name}/{fn}']['AUROC']:.3f}" for fn in fns), flush=True)
        path.write_text(json.dumps(res, indent=1, default=float))
        import csv as _csv
        for run in runs.test:   # add the new columns to the run's score file (the Scores tab)
            f = rm.resolve_path(runs.path[run]) / f"{enc_cfg['output_stem']}.csv"
            rows = list(_csv.DictReader(open(f)))
            old = [c for c in rows[0] if c != "tick"]
            data = {c: ([int(r["tick"]) for r in rows if r[c] != ""], [float(r[c]) for r in rows if r[c] != ""]) for c in old}
            data.update(added[run])
            rm.write_scores_csv(f, [(k, None, None) for k in data], data)
        res["per_anomaly"] = per_anomaly(runs, res, enc_cfg["output_stem"])
        path.write_text(json.dumps(res, indent=1, default=float))
        return print(f"wrote the cross-source series into {path}")
    if args.corpus_only:
        res = json.loads(path.read_text())
        res["corpus_sweep"] = corpus_sweep(runs, series, sources)
        res["protocol"]["fracs"] = FRACS
        path.write_text(json.dumps(res, indent=1, default=float))
        return print(f"wrote the corpus sweep into {path}")

    res = {"pool": "included" if included else "withheld", "reference_runs": runs.ref, "test_runs": runs.test,
           "windows": runs.window, "protocol": {"scoring": "per injection", "pct": PCT, "stride": P.STRIDE, "sizes": SIZES, "draws": DRAWS, "fracs": FRACS, "top_frac": TOP_FRAC},
           "full": {}, "per_run": {}, "threshold_sweep": {}, "pool_sweep": {}, "setup": {"concepts": {}, "per_run": setup_per_run(runs, sources)}}
    csv = {run: {} for run in runs.test}   # {run: {series: (ticks, scores against the whole pool)}}
    for name, fns in series.items():
        y = runs.labels(name)
        for size, d, pool in plan:
            sc = score_series(runs, name, fns, pool)
            for fn, (loo, test) in sc.items():
                key = f"{name}/{fn}"
                m = metrics(loo, test, y)
                res["pool_sweep"].setdefault(key, []).append({"n_runs": size, "draw": d, "runs": pool, **m})
                if size == len(runs.ref):
                    res["full"][key] = m
                    res["per_run"][key] = per_run(runs, name, test, m["tau"])
                    res["threshold_sweep"][key] = [{"pct": p, **metrics(loo, test, y, p)} for p in THRESH_PCTS]
                    i = 0
                    for run in runs.test:
                        t = runs.emb[name, run][1]
                        csv[run][key] = (t.astype(int).tolist(), test[i : i + len(t)].tolist())
                        i += len(t)
        print(f"{name}: " + "  ".join(f"{fn} acc {res['full'][f'{name}/{fn}']['accuracy']:.3f} AUROC {res['full'][f'{name}/{fn}']['AUROC']:.3f}"
                                      for fn in fns if f"{name}/{fn}" in res["full"]), flush=True)
        runs.emb = {k: v for k, v in runs.emb.items() if k[0] != name}   # free this encoder's embeddings
    for source in sources:
        for size, d, pool in plan:
            frames = [c for r in pool for c in runs.concept_lists(source, r)]
            res["setup"]["concepts"].setdefault(source, []).append(
                {"n_runs": size, "draw": d, "distinct": len({c for f in frames for c in f}), "per_frame": float(np.mean([len(f) for f in frames]))})
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=float))
    if not args.series:
        cols = [(f"{n}/{m}", n, m) for n, fns in series.items() for m in fns]
        for run in runs.test:
            rm.write_scores_csv(rm.resolve_path(runs.path[run]) / f"{enc_cfg['output_stem']}.csv", cols, csv[run])
        res["per_anomaly"] = per_anomaly(runs, res, enc_cfg["output_stem"])
        path.write_text(json.dumps(res, indent=1, default=float))

    w = max(len(k) for k in res["full"])
    print(f"\n{res['pool']} pool ({len(runs.ref)} runs), tau = {PCT:g}th percentile of run-level leave-one-out cache scores; test = {', '.join(runs.test)}")
    print(f"{'series':{w}s}  {'TPR':>5s} {'FPR':>5s} {'TNR':>5s} {'acc':>5s} {'AUROC':>5s}")
    for k, m in res["full"].items():
        print(f"{k:{w}s}  {m['TPR']:5.3f} {m['FPR']:5.3f} {m['TNR']:5.3f} {m['accuracy']:5.3f} {m['AUROC']:5.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
