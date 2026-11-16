"""Small persistent content-addressed cache for embedding vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np


DEFAULT_EMBEDDING_CACHE_PATH = Path("cache/embeddings.db")


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(self.path.suffix + ".json")
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, str] = {}
        self._load()

    @staticmethod
    def key(text: str, model_name: str) -> str:
        return hashlib.sha256(f"{model_name}\0{text}".encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            keys = data["keys"].tolist()
            vectors = data["vectors"]
            self._vectors = {key: vectors[index] for index, key in enumerate(keys)}
        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def get(self, text: str, model_name: str) -> np.ndarray | None:
        return self._vectors.get(self.key(text, model_name))

    def put_many(self, texts: Iterable[str], model_name: str, vectors: np.ndarray) -> None:
        for text, vector in zip(texts, vectors, strict=True):
            key = self.key(text, model_name)
            self._vectors[key] = np.asarray(vector, dtype=np.float32)
        self._metadata["model_name"] = model_name

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        vectors = np.stack([self._vectors[key] for key in keys]) if keys else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(self.path, keys=np.asarray(keys), vectors=vectors)
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")


class SQLiteEmbeddingCache:
    """Incremental embedding cache that does not keep millions of vectors in RAM."""

    COMMIT_EVERY = 10_000

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self._pending_writes = 0
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "cache_key TEXT PRIMARY KEY, model_name TEXT NOT NULL, "
            "dimension INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.connection.commit()

    def get(self, text: str, model_name: str) -> np.ndarray | None:
        key = EmbeddingCache.key(text, model_name)
        row = self.connection.execute(
            "SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[1], dtype=np.float32, count=row[0]).copy()

    def put_many(self, texts: Iterable[str], model_name: str, vectors: np.ndarray) -> None:
        rows = []
        for text, vector in zip(texts, vectors, strict=True):
            vector = np.asarray(vector, dtype=np.float32)
            rows.append((EmbeddingCache.key(text, model_name), model_name, vector.size, vector.tobytes()))
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings(cache_key, model_name, dimension, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._pending_writes += len(rows)
        if self._pending_writes >= self.COMMIT_EVERY:
            self.connection.commit()
            self._pending_writes = 0

    def save(self) -> None:
        self.connection.commit()
        self._pending_writes = 0

    def close(self) -> None:
        self.connection.close()
