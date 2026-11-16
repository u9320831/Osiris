"""Generate vector embeddings from the parquet cache through TEI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import tempfile
import time
from typing import Sequence

import polars as pl

from src.embeddings.batch_processor import BatchEmbeddingProcessor
from src.embeddings.embedding_cache import SQLiteEmbeddingCache
from src.embeddings.model import DEFAULT_MODEL, DEFAULT_TEI_ENDPOINT, EmbeddingModel
from src.utils.logging_config import configure_logging, log_stage


DEFAULT_OUTPUT = Path("cache/embeddings.parquet")
SENSITIVE_NAME_PARTS = ("phone", "tel", "mobile", "email", "mail", "iban", "bic")
WHITESPACE_RE = re.compile(r"\s+")
logger = logging.getLogger(__name__)


def _default_columns(schema: pl.Schema) -> list[str]:
    """Choose useful text fields while avoiding direct contact identifiers."""

    text_columns = [name for name, dtype in schema.items() if dtype == pl.String]
    safe_columns = [
        name
        for name in text_columns
        if not any(part in name.casefold() for part in SENSITIVE_NAME_PARTS)
    ]
    return safe_columns or text_columns


def _build_text(row: dict[str, object], columns: Sequence[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        cleaned = WHITESPACE_RE.sub(" ", str(value)).strip()
        if cleaned:
            parts.append(f"{column}: {cleaned}")
    return " | ".join(parts)


@dataclass
class EmbeddingEngine:
    """Embed a bounded set of cache rows through Text Embeddings Inference."""

    cache: Path | str = Path("cache/profile.parquet")
    output: Path | str = DEFAULT_OUTPUT
    model_name: str = DEFAULT_MODEL
    embedding_cache: Path | str | None = None
    endpoint: str = DEFAULT_TEI_ENDPOINT
    columns: Sequence[str] | None = None
    max_rows: int | None = None
    batch_size: int = 16
    read_batch_size: int = 128
    normalize: bool = True

    def __post_init__(self) -> None:
        self.cache = Path(self.cache)
        self.output = Path(self.output)
        if not self.cache.exists():
            raise FileNotFoundError(f"Cache parquet introuvable : {self.cache}")
        if not self.cache.is_file():
            raise ValueError(f"Le cache ne désigne pas un fichier : {self.cache}")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows doit être strictement positif")
        if self.batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if self.read_batch_size <= 0:
            raise ValueError("read_batch_size doit être strictement positif")

    def embed(self) -> Path:
        """Generate embeddings and write them to a parquet file."""

        started = time.perf_counter()
        log_stage(
            logger,
            "EMBED_START",
            cache=self.cache,
            model=self.model_name,
            max_rows=self.max_rows if self.max_rows is not None else "all",
        )

        query = pl.scan_parquet(self.cache)
        schema = query.collect_schema()
        columns = list(self.columns) if self.columns is not None else _default_columns(schema)
        missing = [column for column in columns if column not in schema]
        if missing:
            raise ValueError(f"Colonnes absentes du cache : {', '.join(missing)}")
        if not columns:
            raise ValueError("Aucune colonne textuelle à vectoriser")

        model = EmbeddingModel(self.model_name, self.endpoint)
        embedding_cache = SQLiteEmbeddingCache(self.embedding_cache) if self.embedding_cache else None
        query = query.select(columns)
        if self.max_rows is not None:
            query = query.head(self.max_rows)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0
        dimensions = 0
        try:
            processor = BatchEmbeddingProcessor(
                model,
                embedding_cache,
                batch_size=self.batch_size,
            )
            with tempfile.TemporaryDirectory(prefix=".osiris-embeddings-", dir=self.output.parent) as temp_dir:
                parts = Path(temp_dir)
                row_offset = 0
                for part_number, frame in enumerate(
                    query.collect_batches(chunk_size=self.read_batch_size)
                ):
                    texts = [_build_text(row, columns) for row in frame.iter_rows(named=True)]
                    non_empty = [
                        (row_offset + index, text)
                        for index, text in enumerate(texts)
                        if text
                    ]
                    row_offset += frame.height
                    if not non_empty:
                        continue

                    vectors = processor.encode(
                        [text for _, text in non_empty],
                        normalize=self.normalize,
                    )
                    dimensions = int(vectors.shape[1])
                    part = pl.DataFrame(
                        {
                            "row_id": [index for index, _ in non_empty],
                            "text": [text for _, text in non_empty],
                            "embedding": vectors.tolist(),
                        }
                    )
                    part.write_parquet(parts / f"part-{part_number:08d}.parquet", compression="zstd")
                    rows_written += part.height

                if not rows_written:
                    raise ValueError("Aucune ligne exploitable pour produire un embedding")

                pl.scan_parquet(str(parts / "part-*.parquet")).sink_parquet(
                    self.output,
                    compression="zstd",
                    row_group_size=2_048,
                    engine="streaming",
                )
        finally:
            if embedding_cache:
                embedding_cache.close()
            model.close()

        log_stage(
            logger,
            "EMBED_DONE",
            rows=rows_written,
            dimensions=dimensions,
            output=self.output,
            seconds=f"{time.perf_counter() - started:.2f}",
        )
        return self.output


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère des embeddings depuis le cache OSIRIS")
    parser.add_argument("--cache", type=Path, default=Path("cache/profile.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_TEI_ENDPOINT, help="URL de Text Embeddings Inference")
    parser.add_argument("--embedding-cache", type=Path, help="Cache SQLite optionnel de vecteurs")
    parser.add_argument("--columns", help="Colonnes séparées par des virgules")
    parser.add_argument("--max-rows", type=int, help="Limite optionnelle; par défaut, encode toutes les lignes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--read-batch-size", type=int, default=128, help="Nombre de lignes chargé en mémoire à la fois")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Ne normalise pas les vecteurs (la normalisation est active par défaut)",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    configure_logging(level=args.log_level)

    columns = [value.strip() for value in args.columns.split(",") if value.strip()] if args.columns else None
    engine = EmbeddingEngine(
        cache=args.cache,
        output=args.output,
        model_name=args.model,
        embedding_cache=args.embedding_cache,
        endpoint=args.endpoint,
        columns=columns,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        read_batch_size=args.read_batch_size,
        normalize=not args.no_normalize,
    )
    print(f"Embeddings écrits dans : {engine.embed()}")


if __name__ == "__main__":
    main()
