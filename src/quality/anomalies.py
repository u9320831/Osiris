"""Semantic anomaly scoring."""

from __future__ import annotations

import numpy as np

from src.vector_search.knn import outlier_scores


def semantic_anomalies(vectors: np.ndarray, k: int = 5, percentile: float = 95.0) -> dict[str, object]:
    scores = outlier_scores(vectors, k=k)
    threshold = float(np.percentile(scores, percentile)) if len(scores) else 0.0
    return {
        "scores": scores,
        "threshold": threshold,
        "indices": np.flatnonzero(scores >= threshold),
    }
