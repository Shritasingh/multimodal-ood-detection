"""
TODO: implement simple OOD detection scoring methods for embeddings, 
    e.g. cosine similarity, kNN etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoveltyScorer:
    method: str = "knn"  # tbd
    k: int = 5

    def fit(self, nominal_embeddings: np.ndarray) -> "NoveltyScorer":
        raise NotImplementedError

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher = more anomalous. embeddings: (N, D) or (D,)."""
        raise NotImplementedError

    def calibrate_threshold(self, nominal_embeddings: np.ndarray, fpr: float = 0.05) -> float:
        raise NotImplementedError
