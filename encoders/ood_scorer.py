"""Q1 (measuring OOD): a single, representation-agnostic novelty scorer.

TODO: implement. Should work on any encoder's fixed-size embedding vector
(k-NN or Mahalanobis distance on standardized embeddings, calibrated on a
nominal embedding bank) so the same scorer applies to vision, semantic, and
physics embeddings unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoveltyScorer:
    method: str = "knn"  # "knn" or "mahalanobis"
    k: int = 5

    def fit(self, nominal_embeddings: np.ndarray) -> "NoveltyScorer":
        raise NotImplementedError

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher = more anomalous. embeddings: (N, D) or (D,)."""
        raise NotImplementedError

    def calibrate_threshold(self, nominal_embeddings: np.ndarray, fpr: float = 0.05) -> float:
        raise NotImplementedError
