"""Numpy/sklearn vector index with cosine-distance search."""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, metric: str = "cosine") -> None:
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric doit être 'cosine' ou 'euclidean'")
        self.metric = metric
        self.vectors: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> "VectorIndex":
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("vectors doit être une matrice non vide")
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        self.vectors = vectors
        return self

    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.vectors is None:
            raise RuntimeError("L'index doit être entraîné avec fit()")
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        k = min(k, len(self.vectors))
        if self.metric == "cosine":
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / np.maximum(norms, 1e-12)
            scores = queries @ self.vectors.T
            indices = np.argsort(-scores, axis=1)[:, :k]
            return indices, 1.0 - np.take_along_axis(scores, indices, axis=1)
        distances = np.linalg.norm(queries[:, None, :] - self.vectors[None, :, :], axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        return indices, np.take_along_axis(distances, indices, axis=1)
