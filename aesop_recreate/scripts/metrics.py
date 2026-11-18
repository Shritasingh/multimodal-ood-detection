"""Shared scoring helpers of score.py, experiment.py and embed_map.py: anomaly windows, Elhafsi et al.'s metrics at a
threshold, and the Mahalanobis score under a Ledoit-Wolf Gaussian."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV)


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
