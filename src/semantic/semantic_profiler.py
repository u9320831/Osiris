"""Orchestrate classical profiling, fuzzy matching and semantic analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from src.embeddings.batch_processor import BatchEmbeddingProcessor
from src.embeddings.embedding_cache import DEFAULT_EMBEDDING_CACHE_PATH, SQLiteEmbeddingCache
from src.embeddings.model import DEFAULT_MODEL, DEFAULT_TEI_ENDPOINT, EmbeddingModel
from src.ingestion.cache import scan_cache
from src.profiling.column_profiler import ColumnProfiler
from src.profiling.type_inference import is_sensitive_name
from src.quality.consistency import check_consistency
from src.quality.duplicates import approximate_duplicates, exact_duplicates
from src.similarity.text_similarity import normalize_text
from src.vector_search.knn import identify_groups, nearest_neighbors, outlier_scores


class SemanticProfiler:
    def __init__(
        self,
        cache: str | Path,
        sample_size: int = 10_000,
        embedding_model: str | None = None,
        embedding_cache: str | Path | None = DEFAULT_EMBEDDING_CACHE_PATH,
        endpoint: str = DEFAULT_TEI_ENDPOINT,
    ) -> None:
        self.cache = Path(cache)
        self.sample_size = sample_size
        self.embedding_model = embedding_model
        self.embedding_cache = Path(embedding_cache) if embedding_cache else None
        self.endpoint = endpoint

    def prepare_text(self, frame: pl.DataFrame, columns: list[str] | None = None) -> tuple[list[str], list[str]]:
        schema = frame.schema
        selected = columns or [
            name for name, dtype in schema.items()
            if dtype == pl.String and not is_sensitive_name(name)
        ]
        selected = [name for name in selected if name in schema]
        texts = []
        for row in frame.select(selected).to_dicts():
            parts = []
            for name in selected:
                value = row.get(name)
                normalized = normalize_text(value)
                if normalized:
                    parts.append(f"{name}: {normalized}")
            texts.append(" | ".join(parts))
        return selected, texts

    def profile(
        self,
        *,
        text_columns: list[str] | None = None,
        generate_embeddings: bool = False,
        fuzzy_fields: list[str] | None = None,
        knn_k: int = 5,
    ) -> dict[str, Any]:
        frame = scan_cache(self.cache).head(self.sample_size).collect()
        schema = scan_cache(self.cache).collect_schema()
        selected, texts = self.prepare_text(frame, text_columns)
        report: dict[str, Any] = {
            "classical": ColumnProfiler(self.cache, self.sample_size).profile(),
            "semantic_columns": selected,
            "sensitive_columns": [name for name in schema.names() if is_sensitive_name(name)],
            "exact_duplicates_sample": exact_duplicates(frame),
            "consistency": check_consistency(frame),
            "fuzzy_duplicates": approximate_duplicates(frame, fuzzy_fields or selected[:3]) if selected else [],
        }
        if generate_embeddings:
            model = EmbeddingModel(self.embedding_model or DEFAULT_MODEL, self.endpoint)
            cache = SQLiteEmbeddingCache(self.embedding_cache) if self.embedding_cache else None
            try:
                vectors = BatchEmbeddingProcessor(model, cache).encode(texts)
            finally:
                if cache:
                    cache.close()
                model.close()
            neighbors = nearest_neighbors(vectors, k=knn_k)
            report["embeddings"] = {
                "rows": len(vectors),
                "dimensions": int(vectors.shape[1]),
                "endpoint": model.endpoint,
                "neighbors": {"indices": neighbors["indices"].tolist(), "distances": neighbors["distances"].tolist()},
                "groups": identify_groups(vectors).tolist(),
                "outlier_scores": outlier_scores(vectors, knn_k).tolist(),
            }
        return report
