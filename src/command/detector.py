"""Reliable detection of the text format used by CSV files.

``csv.Sniffer`` is useful as a last resort, but it is deliberately heuristic:
with a short sample it can easily confuse a comma in an address with the real
separator. This module evaluates several delimiter/quote pairs on multiple
records and keeps the most coherent one.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path
from typing import Any

from charset_normalizer import from_bytes


DEFAULT_SAMPLE_SIZE = 256 * 1024
SUPPORTED_DELIMITERS = (",", ";", "\t", "|")
SUPPORTED_QUOTE_CHARS = ('"', "'")


def _read_sample(path: Path, sample_size: int) -> bytes:
    """Read a bounded, record-aligned sample from *path*."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    if not path.is_file():
        raise ValueError(f"Le chemin ne désigne pas un fichier : {path}")
    if sample_size < 1024:
        raise ValueError("sample_size doit être supérieur ou égal à 1024")

    with path.open("rb") as stream:
        sample = stream.read(sample_size)

    if not sample:
        raise ValueError("Le fichier est vide")

    # Avoid scoring a partially-read last row whenever possible. Do not trim
    # UTF-16/32 at the byte level: the newline is two/four bytes wide and a
    # byte-level cut could make the decoder reject an otherwise valid sample.
    has_wide_bom = sample.startswith(
        (b"\xff\xfe", b"\xfe\xff", b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")
    )
    if not has_wide_bom:
        last_newline = max(sample.rfind(b"\n"), sample.rfind(b"\r"))
        if last_newline > 0:
            sample = sample[: last_newline + 1]
    return sample


def _encoding_from_bom(sample: bytes) -> str | None:
    """Return the Python codec implied by a BOM, if one is present."""

    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith(b"\xff\xfe\x00\x00") or sample.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32"
    if sample.startswith(b"\xff\xfe") or sample.startswith(b"\xfe\xff"):
        return "utf-16"
    return None


def _decode_sample(sample: bytes) -> tuple[str, str]:
    """Decode a bounded sample without reading the whole source file."""

    bom_encoding = _encoding_from_bom(sample)
    if bom_encoding is not None:
        return sample.decode(bom_encoding), bom_encoding

    matches = list(from_bytes(sample))
    detected = matches[0] if matches else None

    candidates: list[str] = []
    if detected is not None and detected.encoding:
        detected_name = detected.encoding.lower().replace("-", "_")
        # Single-byte Central/Baltic European encodings can be byte-for-byte
        # valid for a French export. Prefer the interoperable Windows-1252
        # interpretation in that ambiguous case. UTF encodings stay first.
        if detected_name in {
            "cp1250",
            "cp1252",
            "cp1257",
            "cp775",
            "iso8859_1",
            "iso8859_14",
            "iso8859_16",
            "mac_latin2",
        }:
            candidates.append("cp1252")
        candidates.append(detected.encoding)
    candidates.extend(("utf-8", "cp1252", "latin-1"))

    seen: set[str] = set()
    for encoding in candidates:
        normalized = encoding.lower().replace("-", "_")
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            text = sample.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

        # Wrongly decoding UTF-16/32 as a single-byte codec produces NULs.
        if text and text.count("\x00") / len(text) > 0.01:
            continue
        return text, encoding

    raise ValueError("Encodage indéterminé ou fichier binaire")


def _parse_rows(text: str, delimiter: str, quote_char: str) -> list[list[str]]:
    """Parse a bounded number of rows, tolerating a malformed final row."""

    reader = csv.reader(
        io.StringIO(text, newline=""),
        delimiter=delimiter,
        quotechar=quote_char,
        doublequote=True,
        strict=False,
    )
    rows: list[list[str]] = []
    try:
        for row in reader:
            if row and any(cell.strip() for cell in row):
                rows.append(row)
            if len(rows) >= 250:
                break
    except csv.Error:
        # One broken quote must not prevent detection from using other rows.
        pass
    return rows


def _score_candidate(text: str, delimiter: str, quote_char: str) -> dict[str, Any] | None:
    rows = _parse_rows(text, delimiter, quote_char)
    if len(rows) < 2:
        return None

    widths = Counter(len(row) for row in rows)
    width, consistent_rows = widths.most_common(1)[0]
    if width < 2:
        return None

    consistency = consistent_rows / len(rows)
    if text.count(delimiter) == 0:
        return None

    # Consistency is the strongest signal. Number of columns breaks ties in
    # favour of the real separator over punctuation occurring in every row.
    score = (consistency * 100.0) + min(width, 100) * 1.5
    score += min(consistent_rows, 100) * 0.05

    quote_count = text.count(quote_char)
    if quote_count >= 2 and quote_count % 2 == 0:
        score += 2.0

    return {
        "score": score,
        "rows": len(rows),
        "columns": width,
        "consistency": consistency,
        "delimiter": delimiter,
        "quote_char": quote_char,
    }


def detect_csv(path: str | Path, *, sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict[str, Any]:
    """Detect encoding, separator and quote character for a CSV file.

    The original keys are preserved for :mod:`src.command.loader`; metadata
    such as ``confidence`` makes the decision inspectable by other callers.
    """

    file_path = Path(path)
    sample = _read_sample(file_path, sample_size)
    text, encoding = _decode_sample(sample)

    candidates = [
        candidate
        for delimiter in SUPPORTED_DELIMITERS
        for quote_char in SUPPORTED_QUOTE_CHARS
        if (candidate := _score_candidate(text, delimiter, quote_char)) is not None
    ]
    if not candidates:
        raise ValueError("Séparateur CSV indéterminé : aucune structure tabulaire cohérente")

    best = max(candidates, key=lambda candidate: candidate["score"])
    ranked = sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0

    confidence = min(
        1.0,
        (best["consistency"] * 0.85)
        + (min(best["rows"], 100) / 100 * 0.10)
        + (min(max(best["score"] - second_score, 0.0), 20.0) / 20 * 0.05),
    )
    line_terminator = "\r\n" if "\r\n" in text else ("\r" if "\r" in text else "\n")

    return {
        "encoding": encoding,
        "separator": best["delimiter"],
        "quote_char": best["quote_char"],
        "columns": best["columns"],
        "sample_rows": best["rows"],
        "consistency": round(best["consistency"], 4),
        "confidence": round(confidence, 4),
        "line_terminator": line_terminator,
    }
