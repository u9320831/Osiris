"""Deterministic text similarity primitives."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import JaroWinkler, Levenshtein


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(value: object, *, remove_punctuation: bool = True) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().strip()
    if remove_punctuation:
        text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def ngrams(value: object, n: int = 3) -> set[str]:
    normalized = normalize_text(value, remove_punctuation=False)
    padded = f"  {normalized}  "
    return {padded[index : index + n] for index in range(max(len(padded) - n + 1, 0))}


def ngram_similarity(left: object, right: object, n: int = 3) -> float:
    left_grams, right_grams = ngrams(left, n), ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    if not left_grams or not right_grams:
        return 0.0
    return 2 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def levenshtein(left: object, right: object) -> dict[str, float | int]:
    a, b = normalize_text(left), normalize_text(right)
    return {"distance": Levenshtein.distance(a, b), "ratio": Levenshtein.normalized_similarity(a, b)}


def jaro_winkler(left: object, right: object) -> float:
    return JaroWinkler.normalized_similarity(normalize_text(left), normalize_text(right))


def similarity(left: object, right: object) -> dict[str, float | int]:
    lev = levenshtein(left, right)
    jw = jaro_winkler(left, right)
    ng = ngram_similarity(left, right)
    return {
        "levenshtein_distance": int(lev["distance"]),
        "levenshtein_ratio": float(lev["ratio"]),
        "jaro_winkler": float(jw),
        "ngram_similarity": float(ng),
        "score": float(0.45 * lev["ratio"] + 0.35 * jw + 0.20 * ng),
    }
