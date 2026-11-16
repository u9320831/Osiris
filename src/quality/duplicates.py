"""Exact and approximate duplicate checks."""

from __future__ import annotations

from typing import Any

import polars as pl

from src.similarity.candidate_generation import generate_candidates
from src.similarity.text_similarity import similarity


def exact_duplicates(frame: pl.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    columns = columns or frame.columns
    grouped = frame.group_by(columns).agg(pl.len().alias("count")).filter(pl.col("count") > 1)
    duplicate_rows = int(grouped.select(pl.col("count").sum()).item() or 0)
    return {
        "duplicate_groups": grouped.height,
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": round(duplicate_rows / frame.height, 6) if frame.height else 0.0,
    }


def approximate_duplicates(
    frame: pl.DataFrame,
    fields: list[str],
    threshold: float = 0.90,
    max_pairs: int = 100_000,
) -> list[dict[str, Any]]:
    records = frame.select(fields).to_dicts()
    pairs = generate_candidates(records, fields, max_pairs=max_pairs)
    results = []
    for left, right in pairs:
        metrics = [similarity(records[left].get(field), records[right].get(field))["score"] for field in fields]
        score = sum(metrics) / len(metrics) if metrics else 0.0
        if score >= threshold:
            results.append({"left_index": left, "right_index": right, "score": score})
    return sorted(results, key=lambda item: item["score"], reverse=True)
