"""Accessors for the parquet cache used by the profiling pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl


ACCENT_REPLACEMENTS = {
    "à": "a", "á": "a", "â": "a", "ä": "a", "ã": "a",
    "ç": "c", "è": "e", "é": "e", "ê": "e", "ë": "e",
    "ì": "i", "í": "i", "î": "i", "ï": "i",
    "ñ": "n", "ò": "o", "ó": "o", "ô": "o", "ö": "o", "õ": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u", "ý": "y", "ÿ": "y",
}
SEARCH_TEXT_METADATA_COLUMNS = frozenset({"_row_id", "_source", "_source_file", "_search_text"})


def search_text_expression(
    column_names: Iterable[str],
    *,
    normalize: bool = True,
) -> pl.Expr:
    """Build search text, optionally normalizing it before use."""

    names = list(column_names)
    data_columns = [name for name in names if name not in SEARCH_TEXT_METADATA_COLUMNS]
    if data_columns:
        search_text = pl.concat_str(
            [pl.col(name).fill_null("") for name in data_columns], separator=" | "
        )
        if normalize:
            search_text = search_text.str.to_lowercase()
            for accented, plain in ACCENT_REPLACEMENTS.items():
                search_text = search_text.str.replace_all(accented, plain)
    else:
        search_text = pl.lit("")

    if "_search_text" in names:
        return pl.coalesce([pl.col("_search_text"), search_text])
    return search_text


def scan_cache(path: str | Path = "cache/profile.parquet") -> pl.LazyFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cache parquet introuvable : {path}")
    return pl.scan_parquet(path)


def scan_ndjson_as_strings(
    path: str | Path,
    infer_schema_length: int | None = 1_000,
) -> pl.LazyFrame:
    """Scan JSON Lines with stable string columns, including initially-null fields."""

    source = Path(path)
    decoder = json.JSONDecoder()
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            position = 0
            while position < len(line) and line[position].isspace():
                position += 1
            if position == len(line):
                continue
            try:
                record, _ = decoder.raw_decode(line, position)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalide dans {source}, ligne {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Enregistrement JSON non objet dans {source}, ligne {line_number}"
                )
            overrides = {name: pl.String for name in record}
            return pl.scan_ndjson(
                source,
                infer_schema_length=infer_schema_length,
                schema_overrides=overrides,
            )
    raise ValueError(f"Aucun enregistrement JSON trouvé dans {source}")


def read_cache_sample(path: str | Path, n_rows: int = 10_000) -> pl.DataFrame:
    if n_rows <= 0:
        raise ValueError("n_rows doit être strictement positif")
    return scan_cache(path).head(n_rows).collect()
