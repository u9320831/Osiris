"""Batch encoding with deduplication and persistent cache reuse."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from src.embeddings.embedding_cache import EmbeddingCache, SQLiteEmbeddingCache
from src.embeddings.model import EmbeddingBackend


class BatchEmbeddingProcessor:
    def __init__(
        self,
        model: EmbeddingBackend,
        cache: EmbeddingCache | SQLiteEmbeddingCache | None = None,
        batch_size: int = 16,
        concurrency: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if concurrency <= 0:
            raise ValueError("concurrency doit être strictement positif")
        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self.concurrency = concurrency

    def encode(self, texts: Sequence[str], normalize: bool = True, persist_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        result: list[np.ndarray | None] = [None] * len(texts)
        missing: OrderedDict[str, list[int]] = OrderedDict()
        for index, text in enumerate(texts):
            cached = self.cache.get(text, self.model.model_name) if self.cache else None
            if cached is not None:
                result[index] = cached
            else:
                missing.setdefault(text, []).append(index)

        new_texts = list(missing)
        if new_texts:
            new_vectors = self.model.encode(
                new_texts,
                self.batch_size,
                normalize,
                self.concurrency,
            )
            if self.cache:
                self.cache.put_many(new_texts, self.model.model_name, new_vectors)
            for text, vector in zip(new_texts, new_vectors, strict=True):
                for index in missing[text]:
                    result[index] = vector
        if self.cache and persist_cache:
            self.cache.save()
        return np.stack([vector for vector in result if vector is not None])
