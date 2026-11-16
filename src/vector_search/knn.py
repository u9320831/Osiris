"""k-NN utilities, grouping and outlier scores."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from src.vector_search.index import VectorIndex


def nearest_neighbors(vectors: np.ndarray, k: int = 5, metric: str = "cosine") -> dict[str, np.ndarray]:
    index = VectorIndex(metric).fit(vectors)
    indices, distances = index.search(vectors, k=min(k + 1, len(vectors)))
    return {"indices": indices[:, 1:], "distances": distances[:, 1:]}


def identify_groups(vectors: np.ndarray, eps: float = 0.15, min_samples: int = 2) -> np.ndarray:
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(normalized)


def outlier_scores(vectors: np.ndarray, k: int = 5) -> np.ndarray:
    neighbors = nearest_neighbors(vectors, k=k)
    return neighbors["distances"].mean(axis=1)
