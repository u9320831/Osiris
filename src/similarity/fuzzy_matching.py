"""Approximate duplicate detection built on text similarity."""

from __future__ import annotations

from typing import Any, Iterable

from rapidfuzz import process

from src.similarity.text_similarity import normalize_text, similarity


def best_matches(value: object, candidates: Iterable[object], limit: int = 5, threshold: float = 0.80) -> list[dict[str, Any]]:
    values = list(candidates)
    normalized = [normalize_text(candidate) for candidate in values]
    results = process.extract(normalize_text(value), normalized, limit=limit, score_cutoff=threshold * 100)
    matches = []
    for _, _, index in results:
        metrics = similarity(value, values[index])
        matches.append({"index": index, "value": values[index], **metrics})
    return sorted(matches, key=lambda item: item["score"], reverse=True)


def find_approximate_duplicates(values: list[object], threshold: float = 0.90, max_pairs: int = 100_000) -> list[dict[str, Any]]:
    """Return high-confidence duplicate pairs without an all-pairs explosion."""

    normalized: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        key = normalize_text(value)
        if key:
            normalized.setdefault(key, []).append(index)
    pairs: list[dict[str, Any]] = []
    unique_values = list(normalized)
    for left_index, left in enumerate(unique_values):
        matches = process.extract(left, unique_values[left_index + 1 :], limit=10, score_cutoff=threshold * 100)
        for _, _, relative in matches:
            right = unique_values[left_index + 1 + relative]
            metrics = similarity(left, right)
            for i in normalized[left]:
                for j in normalized[right]:
                    pairs.append({"left_index": i, "right_index": j, **metrics})
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs
