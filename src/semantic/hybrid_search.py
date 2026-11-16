"""Fast hybrid retrieval: lexical candidates followed by semantic reranking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.batch_processor import BatchEmbeddingProcessor
from src.embeddings.embedding_cache import (
    DEFAULT_EMBEDDING_CACHE_PATH,
    EmbeddingCache,
    SQLiteEmbeddingCache,
)
from src.embeddings.model import DEFAULT_MODEL, DEFAULT_TEI_ENDPOINT, EmbeddingModel
from src.utils.logging_config import log_stage
from src.vector_search.lexical_index import LexicalIndex


logger = logging.getLogger(__name__)


class HybridSearch:
    def __init__(
        self,
        lexical_index: str | Path = "cache/lexical.sqlite",
        model_name: str = DEFAULT_MODEL,
        embedding_cache: str | Path | None = DEFAULT_EMBEDDING_CACHE_PATH,
        endpoint: str = DEFAULT_TEI_ENDPOINT,
        embedding_batch_size: int = 16,
        embedding_concurrency: int = 4,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size doit être strictement positif")
        if embedding_concurrency <= 0:
            raise ValueError("embedding_concurrency doit être strictement positif")
        self.lexical = LexicalIndex(lexical_index)
        self.model = EmbeddingModel(model_name, endpoint)
        self.embedding_batch_size = embedding_batch_size
        self.embedding_concurrency = embedding_concurrency
        if embedding_cache:
            cache_path = Path(embedding_cache)
            self.cache = (
                SQLiteEmbeddingCache(cache_path)
                if cache_path.suffix.casefold() in {".db", ".sqlite", ".sqlite3"}
                else EmbeddingCache(cache_path)
            )
        else:
            self.cache = None

    def close(self) -> None:
        self.model.close()
        self.lexical.close()
        if isinstance(self.cache, SQLiteEmbeddingCache):
            self.cache.close()

    def search(self, text: str, limit: int = 10, candidates: int = 200) -> list[dict[str, Any]]:
        lexical_hits = self.lexical.search(text, candidates, match_any=True)
        if not lexical_hits:
            return []
        processor = BatchEmbeddingProcessor(
            self.model,
            self.cache,
            batch_size=self.embedding_batch_size,
            concurrency=self.embedding_concurrency,
        )
        vectors = processor.encode(
            [text] + [hit["text"] for hit in lexical_hits],
            persist_cache=False,
        )
        query_vector = vectors[0]
        candidate_vectors = vectors[1:]
        norms = np.linalg.norm(candidate_vectors, axis=1) * max(float(np.linalg.norm(query_vector)), 1e-12)
        semantic_scores = (candidate_vectors @ query_vector) / np.maximum(norms, 1e-12)
        results = []
        for hit, score in zip(lexical_hits, semantic_scores, strict=True):
            results.append({**hit, "semantic_score": float(score)})
        results.sort(key=lambda item: item["semantic_score"], reverse=True)
        log_stage(logger, "HYBRID_SEARCH", candidates=len(lexical_hits), results=min(limit, len(results)))
        if self.cache:
            self.cache.save()
        return results[:limit]
