"""Config-driven scoring step: loads the raw embeddings run_encoders.py
cached (nominal feature pools + each anomaly run's own embeddings) and
computes a per-frame k-nearest-neighbour anomaly score per encoder, using
that encoder's own metric (cosine or Mahalanobis, both oriented so higher
= more anomalous) and k from config/encoders.json.

Cheap and rerunnable: doesn't touch DINOv2/OWL-ViT/BERT/CARLA at all, only
the cached .npz files, so re-scoring with a different k or metric is just
re-running this script -- run_encoders.py doesn't need to be re-run.

Everything comes from the config file (default config/encoders.json).

Usage:
    .venv/bin/python scripts/run_metrics.py [config/encoders_v2.json]

Extra encoders: vision_patch (each patch vs all nominal patches, k=1 cosine, mean of the worst top_frac patches)
and physics_roles (standardised Euclidean kNN). reference_runs pools several nominal runs; output_stem names the csv/png.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# optional first CLI arg: a different config file, e.g. config/encoders_v2.json
CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "config" / "encoders.json"
sys.path.insert(0, str(REPO_ROOT))


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_npz(out_path: str, name: str) -> dict:
    path = resolve_path(out_path) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_encoders.py first")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def load_reference(out_paths: list[str], name: str) -> dict:
    """Load <name>.npz from every reference run and concatenate them along the first axis."""
    parts = [load_npz(p, name) for p in out_paths]
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


def dedupe_by_label(ref: dict) -> dict:
    """Keep the first row of each label so a label already in the pool is not added again."""
    _, first = np.unique(ref["labels"], return_index=True)
    return {k: v[first] if len(v) == len(ref["labels"]) else v for k, v in ref.items()}


# --- distance metrics: both oriented so higher = more anomalous ---

def cosine_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query)
    r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    return 1.0 - (r @ q)


def mahalanobis_distance(query: np.ndarray, reference: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Distance from query to each individual reference row (not to the
    reference set's mean), using one global inverse covariance."""
    diff = reference - query
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(np.clip(d2, 0, None))


def fit_inv_cov(reference: np.ndarray) -> np.ndarray:
    cov = np.cov(reference, rowvar=False)
    # pinv, not inv: a near-constant feature (e.g. an always-empty
    # tracked-agent slot) has ~zero nominal variance, making cov singular.
    return np.linalg.pinv(cov)


def knn_score(query: np.ndarray, reference: np.ndarray, k: int, metric: str, inv_cov: np.ndarray | None) -> float:
    """Mean distance to the k nearest neighbours of query in reference."""
    if metric == "cosine":
        d = cosine_distance(query, reference)
    elif metric == "mahalanobis":
        d = mahalanobis_distance(query, reference, inv_cov)
    else:
        raise ValueError(f"unknown metric: {metric!r} (expected 'cosine' or 'mahalanobis')")
    k = min(k, len(d))
    nearest = np.partition(d, k - 1)[:k]
    return float(nearest.mean())


def score_per_frame(nominal: dict, query: dict, k: int, metric: str, inv_cov: np.ndarray | None) -> tuple[list[int], list[float]]:
    """vision/physics: one embedding per tick, in query['ticks']/['embeddings']."""
    ticks = query["ticks"].tolist()
    scores = [
        knn_score(query["embeddings"][i], nominal["embeddings"], k, metric, inv_cov)
        for i in range(len(ticks))
    ]
    return ticks, scores


def score_semantic_per_frame(nominal: dict, query: dict, k: int, metric: str, inv_cov: np.ndarray | None) -> tuple[list[int], list[float]]:
    """A frame's score is the worst (highest-distance) of its individually
    detected labels; frames with zero detections get NaN."""
    frame_idx = query["frame_idx"]
    n_frames = int(frame_idx.max()) + 1 if len(frame_idx) else 0
    scores = [float("nan")] * n_frames
    per_frame_best: dict[int, float] = {}
    for row_i, t in enumerate(frame_idx):
        t = int(t)
        d = knn_score(query["embeddings"][row_i], nominal["embeddings"], k, metric, inv_cov)
        per_frame_best[t] = max(d, per_frame_best.get(t, float("-inf")))
    for t, d in per_frame_best.items():
        scores[t] = d
    return list(range(n_frames)), scores


def score_raw(query: dict) -> tuple[list[int], list[float]]:
    """A score the encoder produced itself (qwen_judge_score: the VLM's 0-10 rating / 10)."""
    return query["ticks"].tolist(), [float(v) for v in query["embeddings"][:, 0]]


def hidden_features(d: dict, p: dict) -> np.ndarray:
    """qwen_hidden: (T, D) float32 features at the configured pooling ('last_prompt' | 'answer_mean') and layer."""
    layer = list(d["layers"][: d[p.get("pool", "last_prompt")].shape[1]]).index(p.get("layer", 18))
    return d[p.get("pool", "last_prompt")][:, layer].astype(np.float32)


def score_hidden(nominal: dict, query: dict, p: dict, metric: str) -> tuple[list[int], list[float]]:
    """kNN on decoder hidden states. pca_whiten: centre on the nominal mean, project onto the top
    n_components (at most N/10) nominal principal axes and divide by their std, so the Euclidean distance is a
    Mahalanobis distance in the nominal subspace (raw LLM states are dominated by a few huge-activation
    dimensions). cosine: cosine distance after centring."""
    ref, q = hidden_features(nominal, p), hidden_features(query, p)
    mu = ref.mean(0)
    ref, q = ref - mu, q - mu
    if metric == "pca_whiten":
        _, sv, vt = np.linalg.svd(ref, full_matrices=False)
        n = min(p.get("n_components", 64), len(ref) // 10)  # >= 10 nominal samples per whitened axis
        w = vt[:n].T / (sv[:n] / np.sqrt(len(ref) - 1))
        ref, q = ref @ w, q @ w
        d = np.sqrt(((q[:, None, :] - ref[None]) ** 2).sum(-1))
    elif metric == "cosine":
        rn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        d = 1.0 - qn @ rn.T
    else:
        raise ValueError(f"unknown qwen_hidden metric: {metric!r} (expected 'pca_whiten' or 'cosine')")
    k = min(p.get("k", 5), d.shape[1])
    return query["ticks"].tolist(), np.partition(d, k - 1, axis=1)[:, :k].mean(1).tolist()


def load_patch_bank(out_paths: list[str], name: str):
    """Stream every reference run's patch tokens into one L2-normalised fp16 bank (GPU if available), one run at a time."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    bank, row = None, 0
    for p in out_paths:
        e = np.load(resolve_path(p) / f"{name}.npz")["embeddings"]
        flat = e.reshape(-1, e.shape[-1])
        if bank is None:
            bank = torch.empty((len(out_paths) * len(flat), flat.shape[1]), dtype=dt, device=dev)
        assert row + len(flat) <= len(bank), f"{p} has a different size from the first reference run"
        for i in range(0, len(flat), 65536):
            block = torch.from_numpy(flat[i : i + 65536]).to(dev).float()
            bank[row + i : row + i + len(block)] = torch.nn.functional.normalize(block, dim=-1).to(dt)
        row += len(flat)
    return bank[:row]


def score_vision_patch(nominal: dict, query: dict, top_frac: float, k: int = 1, keep_maps: bool = False):
    """Score each frame by the mean distance of its worst top_frac patches to their nearest nominal patch (any frame, any position)."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32

    def prep(x):
        return torch.nn.functional.normalize(torch.from_numpy(x).to(dev).float(), dim=-1).to(dt)

    bank = nominal["bank"] if "bank" in nominal else prep(nominal["embeddings"].reshape(-1, nominal["embeddings"].shape[-1]))
    N = query["embeddings"].shape[1]
    n_worst = max(1, int(round(top_frac * N)))
    scores, maps = [], []
    for i in range(len(query["ticks"])):
        sim = prep(query["embeddings"][i]) @ bank.T                       # (N, Tn*N)
        nn = sim.max(dim=1).values if k == 1 else sim.topk(k, dim=1).values.mean(dim=1)
        d = 1.0 - nn.float()
        scores.append(float(d.topk(n_worst).values.mean()))
        if keep_maps:
            maps.append(d.cpu().numpy())
    out = (query["ticks"].tolist(), scores)
    return out + (np.stack(maps),) if keep_maps else out


def score_std_euclidean(nominal: dict, query: dict, k: int) -> tuple[list[int], list[float]]:
    """kNN mean distance in nominal-standardised feature space (no covariance inverse)."""
    ref = nominal["embeddings"].astype(np.float64)
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-6] = 1.0
    R, Q = (ref - mu) / sd, (query["embeddings"].astype(np.float64) - mu) / sd
    scores = []
    for i in range(len(Q)):
        d = np.linalg.norm(R - Q[i], axis=1)
        kk = min(k, len(d))
        scores.append(float(np.partition(d, kk - 1)[:kk].mean()))
    return query["ticks"].tolist(), scores


def metrics_of(p: dict) -> list:
    """The metrics configured for an encoder: `metrics` (a list) or the single `metric`."""
    return p.get("metrics") or [p.get("metric")]


def score_encoder(name, nominal, query, p, metric, inv_cov):
    """Per-frame scores of one encoder under one metric."""
    if name == "vision_patch":
        return score_vision_patch(nominal, query, p.get("top_frac", 0.05), p.get("k", 1))
    if name == "physics_roles" and metric in (None, "std_euclidean"):
        return score_std_euclidean(nominal, query, p.get("k", 5))
    if name in ("semantic", "qwen_labels", "qwen_judge_labels"):
        return score_semantic_per_frame(nominal, query, p["k"], metric, inv_cov)
    if name == "qwen_judge_score":
        return score_raw(query)
    if name == "qwen_hidden":
        return score_hidden(nominal, query, p, metric)
    return score_per_frame(nominal, query, p["k"], metric, inv_cov)


def plot_run(
    run_name: str,
    out_path: Path,
    scores: dict[str, tuple[list[int], list[float]]],
    show: bool,
    onset_tick: int | None = None,
    offset_tick: int | None = None,
    stem: str = "anomaly_scores",
) -> None:
    """One row per encoder (vision and semantic share a cosine row); an encoder with several metrics plots them together on one axis."""
    import matplotlib.pyplot as plt

    def enc_of(key):
        return key.split("/")[0]

    groups: list[dict] = []
    cosine = {k: v for k, v in scores.items() if k in ("vision", "semantic")}
    if cosine:
        groups.append(cosine)
    for enc in dict.fromkeys(enc_of(k) for k in scores if k not in ("vision", "semantic")):
        groups.append({k: v for k, v in scores.items() if enc_of(k) == enc})
    if not groups:
        return

    colours = {"vision": "tab:blue", "semantic": "tab:orange", "physics": "tab:green", "vision_patch": "tab:blue", "physics_roles": "tab:green"}
    ylabels = {"physics": "Mahalanobis distance (kNN)", "vision_patch": "patch cosine distance", "physics_roles": "std. Euclidean (kNN)"}
    styles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 4.5 * len(groups)), squeeze=False)
    for ax, grp in zip(axes[:, 0], groups):
        multi = len(grp) > 1 and len({enc_of(k) for k in grp}) == 1
        if onset_tick is not None:
            ax.axvspan(onset_tick, offset_tick if offset_tick is not None else onset_tick, color="red", alpha=0.12, label="anomaly window", zorder=0)
        for n, (key, (ticks, vals)) in enumerate(grp.items()):
            v = np.array(vals, dtype=float)
            label = f"{key} embedding"
            ax.plot(ticks, v, linewidth=1.5, color=colours.get(enc_of(key)), linestyle=styles[n % len(styles)], label=label)
        ax.set_ylabel("distance (kNN)" if multi else ylabels.get(next(iter(grp)), "cosine distance (kNN)"))
        if set(grp) == {"semantic"}:
            ax.set_ylim(-0.4, 0.4)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("tick")
        ax.grid(True, linewidth=0.5, alpha=0.4)
    axes[0, 0].set_title(run_name)
    fig.tight_layout()

    plot_path = out_path / f"{stem}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  wrote {plot_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def params_from_config(config: dict) -> dict:
    """Per-encoder scoring parameters, with defaults for anything the config omits."""
    return {
        "vision": config.get("vision_params", {"k": 5, "metric": "cosine"}),
        "semantic": config.get("semantic_params", {"k": 5, "metric": "cosine"}),
        "physics": config.get("physics_params", {"k": 5, "metric": "mahalanobis"}),
        "vision_patch": config.get("vision_patch_params", {"k": 1, "top_frac": 0.05}),
        "physics_roles": config.get("physics_roles_params", {"k": 5, "metric": "std_euclidean"}),
        "qwen_labels": config.get("qwen_labels_params", {"k": 1, "metric": "cosine", "dedupe_reference": True}),
        "qwen_judge_labels": config.get("qwen_judge_labels_params", {"k": 1, "metric": "cosine", "dedupe_reference": True}),
        "qwen_judge_score": config.get("qwen_judge_score_params", {"k": 1, "metric": "raw"}),
        "qwen_hidden": config.get("qwen_hidden_params", {"k": 5, "metric": "pca_whiten", "layer": 18, "pool": "last_prompt", "n_components": 64}),
    }


def build_reference(ref_paths: list[str], active_encoders: list[str], params: dict):
    """Pooled nominal data per encoder (patch bank on the GPU, one row per semantic label if configured) plus Mahalanobis inverse covariances."""
    nominal = {
        name: {"bank": load_patch_bank(ref_paths, f"{name}_nominal")} if name == "vision_patch" else load_reference(ref_paths, f"{name}_nominal")
        for name in active_encoders
    }
    for name in nominal:
        if params[name].get("dedupe_reference") and "labels" in nominal[name]:
            nominal[name] = dedupe_by_label(nominal[name])
    inv_cov = {
        name: fit_inv_cov(nominal[name]["embeddings"])
        for name in active_encoders
        if "mahalanobis" in metrics_of(params[name])
    }
    return nominal, inv_cov


def series_of(active_encoders: list[str], params: dict) -> list[tuple[str, str, str]]:
    """(column key, encoder, metric) for every scored series."""
    return [(name if len(metrics_of(params[name])) == 1 else f"{name}/{m}", name, m) for name in active_encoders for m in metrics_of(params[name])]


def main():
    config = json.loads(CONFIG_PATH.read_text())
    active_encoders = config["encoders"]
    stem = config.get("output_stem", "anomaly_scores")
    params = params_from_config(config)
    refs = config.get("reference_runs") or [config["reference_run"]]
    nominal, inv_cov = build_reference([r["out_path"] for r in refs], active_encoders, params)
    series = series_of(active_encoders, params)

    for run in config["anomaly_runs"]:
        print(f"=== {run['run_name']} ===")
        out_path = resolve_path(run["out_path"])
        scores: dict[str, tuple[list[int], list[float]]] = {}

        queries = {name: load_npz(run["out_path"], name) for name in active_encoders}
        for key, name, metric in series:
            ticks, vals = score_encoder(name, nominal[name], queries[name], params[name], metric, inv_cov.get(name))
            scores[key] = (ticks, vals)
            print(f"  {key}: {len(ticks)} scored ticks ({metric or 'patch-cosine'}, k={params[name]['k']})")

        # union of all ticks any encoder scored, so the CSV has one row per
        # tick even though encoders can cover different tick ranges/subsets
        # (physics skips tick 0, physics_roles ticks 0-1, semantic skips zero-detection ticks)
        all_ticks = sorted(set().union(*(set(t) for t, _ in scores.values())))
        by_tick = {
            name: dict(zip(ticks, vals))
            for name, (ticks, vals) in scores.items()
        }

        csv_path = out_path / f"{stem}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick"] + [key for key, _, _ in series])
            for t in all_ticks:
                row = [t]
                for key, _, _ in series:
                    v = by_tick[key].get(t, float("nan"))
                    row.append("" if np.isnan(v) else round(v, 6))
                writer.writerow(row)
        print(f"  wrote {csv_path}")

        if config.get("plot", False) or config.get("show", False):
            plot_run(
                run["run_name"], out_path, scores, config.get("show", False),
                onset_tick=run.get("onset_tick"), offset_tick=run.get("offset_tick"),
                stem=stem,
            )


if __name__ == "__main__":
    main()
