"""Blocking strategies that reduce fuzzy-comparison cost."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from src.similarity.text_similarity import ngrams, normalize_text


def generate_candidates(records: Iterable[dict[str, Any]], fields: list[str], max_pairs: int = 100_000) -> list[tuple[int, int]]:
    """Generate record pairs sharing at least one normalized word/trigram."""

    buckets: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(records):
        tokens: set[str] = set()
        for field in fields:
            value = normalize_text(record.get(field))
            tokens.update(value.split())
            tokens.update(ngrams(value, 3))
        for token in tokens:
            if token:
                buckets[token].add(index)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        ordered = sorted(members)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pairs.add((left, right))
                if len(pairs) >= max_pairs:
                    return sorted(pairs)
    return sorted(pairs)
