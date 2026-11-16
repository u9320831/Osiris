"""Index and search normalized profiles in Qdrant."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import time
import uuid
import zlib
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.embeddings.batch_processor import BatchEmbeddingProcessor
from src.embeddings.embedding_cache import (
    EmbeddingCache,
    SQLiteEmbeddingCache,
)
from src.embeddings.model import (
    DEFAULT_MODEL,
    DEFAULT_TEI_ENDPOINT,
    EmbeddingBackend,
    EmbeddingModel,
    HashingEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from src.ingestion.cache import search_text_expression
from src.semantic.identity_linking import (
    DOCUMENT_LOOKUP_FIELD,
    IDENTITY_TOKEN_FIELD,
    combine_profiles,
    document_lookup_token,
    identity_tokens,
    profile_embedding_chunks,
    related_profile_groups,
)
from src.similarity.text_similarity import normalize_text
from src.utils.logging_config import configure_logging, log_stage
from src.vector_search.lexical_index import LexicalIndex
from src.vector_search.qdrant_store import QdrantStore


logger = logging.getLogger(__name__)
DEFAULT_LEXICAL_INDEX = Path("cache/lexical.sqlite")
RRF_CONSTANT = 60
QUERY_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _document_key(
    source_file: object,
    row_id: object,
    shard: object = None,
) -> tuple[str, str, str | None]:
    shard_name = Path(str(shard)).name if shard else None
    return str(source_file or ""), str(row_id), shard_name


def _normalized_vector_score(score: object, *, hashing: bool) -> float | None:
    if score is None:
        return None
    numeric_score = float(score)
    if hashing:
        return max(0.0, min(1.0, numeric_score))
    return (max(-1.0, min(1.0, numeric_score)) + 1.0) / 2.0


def _fast_point_id(shard_index: int, row_id: object) -> int:
    if not 0 <= shard_index < 2**16:
        raise ValueError("Le backend hashing accepte au plus 65 536 shards")
    numeric_row_id = int(row_id)
    if not 0 <= numeric_row_id < 2**48:
        raise ValueError(f"_row_id hors plage pour l'identifiant Qdrant : {row_id!r}")
    return (shard_index << 48) | numeric_row_id


def _pack_profile(row: dict[str, Any], columns: list[str]) -> str:
    values = [_safe_value(row.get(column)) for column in columns]
    serialized = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(serialized, level=1)).decode("ascii")


def _encode_fast_batch(
    model: HashingEmbeddingModel,
    texts: list[str],
    batch_size: int,
) -> tuple[csr_matrix, float]:
    started = time.perf_counter()
    vectors = model.encode(texts, batch_size=batch_size)
    return vectors, time.perf_counter() - started


def _build_fast_payloads(
    rows: list[dict[str, Any]],
    columns: list[str],
    shard_name: str,
    used: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    payloads = [
        {
            "_source_file": _safe_value(row.get("_source_file", shard_name)),
            "_row_id": _safe_value(row.get("_row_id", used + offset)),
            "_source": _safe_value(row.get("_source")),
            "_osiris_shard": shard_name,
            "_profile_blob": _pack_profile(row, columns),
            IDENTITY_TOKEN_FIELD: identity_tokens(row),
        }
        for offset, row in enumerate(rows)
    ]
    return payloads, time.perf_counter() - started


def _merge_ranked_results(
    vector_hits: list[dict[str, Any]],
    lexical_hits: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Fuse independent Qdrant and FTS rankings with reciprocal rank fusion."""

    merged: dict[tuple[str, str, str | None], dict[str, Any]] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        payload = hit.get("payload") or {}
        source_file = payload.get("_source_file", payload.get("source_file", ""))
        row_id = payload.get("_row_id", payload.get("row_id", hit["id"]))
        shard = payload.get("_osiris_shard")
        key = _document_key(source_file, row_id, shard)
        item = merged.setdefault(
            key,
            {
                "id": hit["id"],
                "payload": payload,
                "score": 0.0,
                "vector_score": hit.get("score"),
                "lexical_score": None,
                "vector_rank": None,
                "lexical_rank": None,
                "matched_by": [],
            },
        )
        item["score"] += 1.0 / (RRF_CONSTANT + rank)
        item["vector_rank"] = rank
        item["matched_by"].append("vector")

    for rank, hit in enumerate(lexical_hits, start=1):
        key = _document_key(hit.get("source_file", ""), hit.get("row_id", ""), hit.get("shard"))
        item = merged.get(key)
        if item is None:
            item = merged.get(_document_key(hit.get("source_file", ""), hit.get("row_id", "")))
        if item is None:
            item = {
                "id": None,
                "payload": {
                    "_source_file": hit.get("source_file"),
                    "_row_id": hit.get("row_id"),
                    "_osiris_shard": Path(str(hit["shard"])).name if hit.get("shard") else None,
                },
                "score": 0.0,
                "vector_score": None,
                "lexical_score": None,
                "vector_rank": None,
                "lexical_rank": None,
                "matched_by": [],
            }
            merged[key] = item
        item["score"] += 1.0 / (RRF_CONSTANT + rank)
        item["lexical_score"] = hit.get("lexical_score")
        item["lexical_rank"] = rank
        item["text"] = hit.get("text", "")
        item["matched_by"].append("lexical")

    results = sorted(
        merged.values(),
        key=lambda item: (
            -item["score"],
            item["vector_rank"] or float("inf"),
            item["lexical_rank"] or float("inf"),
        ),
    )
    return results[:limit]


class QdrantProfilePipeline:
    def __init__(
        self,
        shards_dir: Path | str = Path("cache/ingested"),
        collection: str = "osiris_profiles",
        url: str | None = "http://localhost:6333",
        local_path: Path | str | None = None,
        model_name: str = DEFAULT_MODEL,
        embedding_cache: Path | str | None = None,
        endpoint: str = DEFAULT_TEI_ENDPOINT,
        embedding_backend: str = "tei",
        device: str = "cuda",
        precision: str = "auto",
        prefer_grpc: bool = False,
        identity_threshold: float = 0.72,
        max_linked_profiles: int = 1_000,
    ) -> None:
        if not 0.0 < identity_threshold <= 1.0:
            raise ValueError("identity_threshold doit être supérieur à 0 et inférieur ou égal à 1")
        if max_linked_profiles <= 0:
            raise ValueError("max_linked_profiles doit être strictement positif")
        if embedding_backend == "hashing" and embedding_cache:
            raise ValueError("Le backend hashing ne prend pas en charge --embedding-cache")
        self.shards_dir = Path(shards_dir)
        self.store = QdrantStore(collection, url, local_path, prefer_grpc)
        self.model: EmbeddingBackend | HashingEmbeddingModel
        if embedding_backend == "tei":
            self.model = EmbeddingModel(model_name, endpoint)
        elif embedding_backend == "sentence-transformers":
            self.model = SentenceTransformerEmbeddingModel(model_name, device, precision)
        elif embedding_backend == "hashing":
            self.model = HashingEmbeddingModel()
        else:
            raise ValueError(f"Backend d'embeddings inconnu : {embedding_backend}")
        self.embedding_backend = embedding_backend
        self._preprocessing_executor = (
            ThreadPoolExecutor(max_workers=2)
            if embedding_backend == "hashing"
            else None
        )
        self._upload_executor = (
            ThreadPoolExecutor(max_workers=2)
            if embedding_backend == "hashing"
            else None
        )
        self._profile_columns_cache: dict[str, list[str]] = {}
        self.identity_threshold = identity_threshold
        self.max_linked_profiles = max_linked_profiles
        if embedding_cache:
            self.embedding_cache = (
                SQLiteEmbeddingCache(embedding_cache)
                if Path(embedding_cache).suffix.casefold() in {".db", ".sqlite", ".sqlite3"}
                else EmbeddingCache(embedding_cache)
            )
        else:
            self.embedding_cache = None

    def close(self) -> None:
        if self._preprocessing_executor is not None:
            self._preprocessing_executor.shutdown(wait=True)
        if self._upload_executor is not None:
            self._upload_executor.shutdown(wait=True)
        self.model.close()
        if isinstance(self.embedding_cache, SQLiteEmbeddingCache):
            self.embedding_cache.close()
        self.store.close()

    def _restore_fast_profiles(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for hit in hits:
            payload = hit.get("payload") or {}
            packed = payload.pop("_profile_blob", None)
            if packed is None:
                continue
            shard_name = Path(str(payload.get("_osiris_shard", ""))).name
            columns = self._profile_columns_cache.get(shard_name)
            if columns is None:
                shard = self.shards_dir / shard_name
                if not shard.is_file():
                    raise FileNotFoundError(
                        f"Parquet requis pour restaurer le profil introuvable : {shard}"
                    )
                schema_names = pl.scan_parquet(shard).collect_schema().names()
                columns = [
                    name
                    for name in schema_names
                    if name not in {"_search_text", "_row_id", "_source", "_source_file"}
                ]
                self._profile_columns_cache[shard_name] = columns
            try:
                values = json.loads(zlib.decompress(base64.b64decode(packed)))
            except (ValueError, zlib.error) as error:
                raise RuntimeError(
                    f"Payload profil compressé invalide pour le shard {shard_name}"
                ) from error
            if not isinstance(values, list) or len(values) != len(columns):
                raise RuntimeError(
                    f"Schéma du profil compressé incohérent pour le shard {shard_name}"
                )
            restored = dict(zip(columns, values, strict=True))
            restored.update(payload)
            hit["payload"] = restored
        return hits

    def index(
        self,
        batch_size: int = 64,
        read_batch_size: int = 2_048,
        embedding_concurrency: int = 4,
        upsert_batch_size: int | None = None,
    ) -> dict[str, int]:
        effective_upsert_batch_size = (
            1_024 if self.embedding_backend == "hashing" else 128
        ) if upsert_batch_size is None else upsert_batch_size
        if effective_upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size doit être strictement positif")
        shards = sorted(self.shards_dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"Aucun shard parquet dans {self.shards_dir}")
        processor = None
        if self.embedding_backend != "hashing":
            processor = BatchEmbeddingProcessor(
                self.model,
                self.embedding_cache,
                batch_size,
                concurrency=(
                    embedding_concurrency
                    if self.embedding_backend == "tei"
                    else 1
                ),
            )
        indexed = 0

        def finish_fast_upload(
            pending: list[tuple[Any, int, str, int, float, float, float]],
        ) -> None:
            future, row_count, shard_name, total_rows, batch_started, embedding_seconds, payload_seconds = pending.pop(0)
            future.result()
            elapsed = time.perf_counter() - batch_started
            log_stage(
                logger,
                "QDRANT_BATCH",
                shard=shard_name,
                rows=row_count,
                total=total_rows,
                embeddings_seconds=f"{embedding_seconds:.2f}",
                payload_seconds=f"{payload_seconds:.2f}",
                upsert_seconds=f"{elapsed:.2f}",
                rows_per_second=f"{row_count / elapsed:.0f}",
            )

        for shard_index, shard in enumerate(shards):
            log_stage(logger, "QDRANT_SHARD_START", shard=shard.name)
            query = pl.scan_parquet(shard)
            schema = query.collect_schema()
            schema_names = schema.names()
            text_expression = search_text_expression(schema_names)
            profile_columns = [
                name
                for name in schema_names
                if name not in {"_search_text", "_row_id", "_source", "_source_file"}
            ]
            self._profile_columns_cache[shard.name] = profile_columns
            used = 0
            pending_uploads: list[tuple[Any, int, str, int, float, float, float]] = []
            for frame in query.collect_batches(chunk_size=read_batch_size):
                batch_started = time.perf_counter()
                rows = frame.to_dicts()
                texts = frame.select(text_expression.alias("_search_text")).to_series().to_list()
                if self.embedding_backend == "hashing":
                    if not isinstance(self.model, HashingEmbeddingModel):
                        raise RuntimeError("Backend hashing incohérent")
                    if self._preprocessing_executor is None:
                        raise RuntimeError("Le pool de préparation hashing n'est pas initialisé")
                    embedding_future = self._preprocessing_executor.submit(
                        _encode_fast_batch,
                        self.model,
                        texts,
                        batch_size,
                    )
                    payload_future = self._preprocessing_executor.submit(
                        _build_fast_payloads,
                        rows,
                        profile_columns,
                        shard.name,
                        used,
                    )
                    vectors, embedding_seconds = embedding_future.result()
                    payloads, payload_seconds = payload_future.result()
                else:
                    embedding_started = time.perf_counter()
                    if processor is None:
                        raise RuntimeError("Le processeur d'embeddings n'est pas initialisé")
                    vectors = processor.encode(texts, persist_cache=False)
                    embedding_seconds = time.perf_counter() - embedding_started
                    payload_started = time.perf_counter()
                    ids = [
                        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{shard.name}:{indexed + offset}"))
                        for offset in range(len(rows))
                    ]
                    payloads = [
                        {
                            **{
                                key: _safe_value(value)
                                for key, value in row.items()
                                if key != "_search_text"
                            },
                            "_source_file": _safe_value(row.get("_source_file", shard.name)),
                            "_row_id": _safe_value(row.get("_row_id", used + offset)),
                            "_osiris_shard": shard.name,
                            IDENTITY_TOKEN_FIELD: identity_tokens(row),
                            DOCUMENT_LOOKUP_FIELD: document_lookup_token(
                                row.get("_source_file", shard.name),
                                row.get("_row_id", used + offset),
                                shard.name,
                            ),
                        }
                        for offset, row in enumerate(rows)
                    ]
                    payload_seconds = time.perf_counter() - payload_started
                if self.embedding_backend == "hashing":
                    ids = [
                        _fast_point_id(
                            shard_index,
                            row.get("_row_id", used + offset),
                        )
                        for offset, row in enumerate(rows)
                    ]
                upsert_started = time.perf_counter()
                if self.embedding_backend == "hashing":
                    if not isinstance(vectors, csr_matrix):
                        raise RuntimeError("Le backend hashing doit produire une matrice creuse")
                    if self._upload_executor is None:
                        raise RuntimeError("Le pool d'upload hashing n'est pas initialisé")
                    indexed += len(rows)
                    used += len(rows)
                    future = self._upload_executor.submit(
                        self.store.upsert_sparse,
                        ids,
                        vectors,
                        payloads,
                        effective_upsert_batch_size,
                        True,
                    )
                    pending_uploads.append(
                        (
                            future,
                            len(rows),
                            shard.name,
                            indexed,
                            batch_started,
                            embedding_seconds,
                            payload_seconds,
                        )
                    )
                    if len(pending_uploads) >= 2:
                        finish_fast_upload(pending_uploads)
                else:
                    self.store.upsert(
                        ids,
                        vectors,
                        payloads,
                        batch_size=effective_upsert_batch_size,
                        wait=False,
                    )
                    upsert_seconds = time.perf_counter() - upsert_started
                    indexed += len(rows)
                    used += len(rows)
                    batch_seconds = time.perf_counter() - batch_started
                    log_stage(
                        logger,
                        "QDRANT_BATCH",
                        shard=shard.name,
                        rows=len(rows),
                        total=indexed,
                        embeddings_seconds=f"{embedding_seconds:.2f}",
                        payload_seconds=f"{payload_seconds:.2f}",
                        upsert_seconds=f"{upsert_seconds:.2f}",
                        rows_per_second=f"{len(rows) / batch_seconds:.0f}",
                    )
            while pending_uploads:
                finish_fast_upload(pending_uploads)
            log_stage(logger, "QDRANT_SHARD_DONE", shard=shard.name, rows=used)
        if self.embedding_cache:
            self.embedding_cache.save()
        return {"indexed": indexed, "shards": len(shards)}

    def search(
        self,
        text: str,
        limit: int = 10,
        candidates: int = 200,
        lexical_index: Path | str = DEFAULT_LEXICAL_INDEX,
    ) -> list[dict[str, Any]]:
        if not text.strip():
            raise ValueError("La requête ne peut pas être vide")
        if limit <= 0 or candidates <= 0:
            raise ValueError("limit et candidates doivent être strictement positifs")
        processor = (
            None
            if self.embedding_backend == "hashing"
            else BatchEmbeddingProcessor(self.model, self.embedding_cache)
        )
        normalized_query = normalize_text(text, remove_punctuation=False)
        terms = list(dict.fromkeys(QUERY_TOKEN_RE.findall(normalized_query)))
        query_texts = [text, *(term for term in terms if term.casefold() != text.casefold())]
        if self.embedding_backend == "hashing":
            if not isinstance(self.model, HashingEmbeddingModel):
                raise RuntimeError("Backend hashing incohérent")
            query_vectors = self.model.encode(query_texts)
            vector = query_vectors.getrow(0)
        else:
            if processor is None:
                raise RuntimeError("Le processeur d'embeddings n'est pas initialisé")
            query_vectors = processor.encode(query_texts, persist_cache=False)
            vector = query_vectors[0]
        lexical = LexicalIndex(lexical_index)
        try:
            if self.embedding_backend == "hashing":
                vector_rankings = [
                    hit
                    for query_index in range(query_vectors.shape[0])
                    for hit in self.store.search_sparse(
                        query_vectors.getrow(query_index),
                        candidates,
                    )
                ]
            else:
                vector_rankings = [
                    hit
                    for query_vector in query_vectors
                    for hit in self.store.search(query_vector, candidates)
                ]
            vector_hits = _merge_ranked_results(vector_rankings, [], candidates)
            lexical_hits = lexical.search(text, candidates, match_any=True)
        finally:
            lexical.close()

        if self.embedding_backend == "hashing":
            shard_indexes = {
                shard.name: index
                for index, shard in enumerate(sorted(self.shards_dir.glob("*.parquet")))
            }
            lexical_point_ids = []
            for hit in lexical_hits:
                shard_name = Path(str(hit.get("shard", ""))).name
                if shard_name not in shard_indexes:
                    continue
                lexical_point_ids.append(
                    _fast_point_id(shard_indexes[shard_name], hit.get("row_id", 0))
                )
            lexical_profile_hits = self._restore_fast_profiles(
                self.store.retrieve(dict.fromkeys(lexical_point_ids))
            )
            vector_hits = self._restore_fast_profiles(vector_hits)
            lexical_profiles_truncated = False
        else:
            lexical_document_keys = {
                document_lookup_token(
                    hit.get("source_file", ""),
                    hit.get("row_id", ""),
                    hit.get("shard"),
                )
                for hit in lexical_hits
            }
            lexical_profile_hits, lexical_profiles_truncated = (
                self.store.search_profiles_by_document_keys(
                    lexical_document_keys,
                    limit=max(candidates, 1),
                )
            )
        seed_hits = list(vector_hits)
        known_seed_keys = {
            _document_key(
                (hit.get("payload") or {}).get("_source_file", ""),
                (hit.get("payload") or {}).get("_row_id", hit.get("id", "")),
                (hit.get("payload") or {}).get("_osiris_shard"),
            )
            for hit in vector_hits
        }
        for hit in lexical_profile_hits:
            payload = hit.get("payload") or {}
            key = _document_key(
                payload.get("_source_file", ""),
                payload.get("_row_id", hit.get("id", "")),
                payload.get("_osiris_shard"),
            )
            if key not in known_seed_keys:
                seed_hits.append(hit)
                known_seed_keys.add(key)

        identity_query_tokens = {
            token
            for hit in seed_hits
            for token in identity_tokens(hit.get("payload") or {})
        }
        related_hits, links_truncated = self.store.search_related_profiles(
            identity_query_tokens,
            limit=self.max_linked_profiles,
        )
        links_truncated = links_truncated or lexical_profiles_truncated
        groups = related_profile_groups(
            seed_hits,
            related_hits,
            self.identity_threshold,
        )
        fused_results = _merge_ranked_results(
            vector_hits,
            lexical_hits,
            max(limit, len(vector_hits) + len(lexical_hits)),
        )
        fused_by_key = {
            _document_key(
                (result.get("payload") or {}).get("_source_file", ""),
                (result.get("payload") or {}).get("_row_id", ""),
                (result.get("payload") or {}).get("_osiris_shard"),
            ): result
            for result in fused_results
        }
        for hit in lexical_profile_hits:
            payload = hit.get("payload") or {}
            key = _document_key(
                payload.get("_source_file", ""),
                payload.get("_row_id", hit.get("id", "")),
                payload.get("_osiris_shard"),
            )
            result = fused_by_key.get(key)
            if result is not None:
                result["id"] = hit.get("id")
                result["payload"] = payload

        entity_results = []
        grouped_keys = set()
        group_chunks: list[str] = []
        group_chunk_ranges: list[tuple[int, int, int]] = []
        for seed, linked in groups:
            seed_payload = seed.get("payload") or {}
            seed_key = _document_key(
                seed_payload.get("_source_file", ""),
                seed_payload.get("_row_id", seed.get("id", "")),
                seed_payload.get("_osiris_shard"),
            )
            result = fused_by_key.get(seed_key)
            if result is None:
                continue
            profiles = [seed_payload, *(hit.get("payload") or {} for hit, _ in linked)]
            related_profiles = [
                {
                    "id": hit.get("id"),
                    "payload": {
                        key: value
                        for key, value in (hit.get("payload") or {}).items()
                        if key not in {IDENTITY_TOKEN_FIELD, DOCUMENT_LOOKUP_FIELD}
                    },
                    "identity_confidence": confidence,
                }
                for hit, confidence in linked
            ]
            result = dict(result)
            result["payload"] = {
                key: value
                for key, value in seed_payload.items()
                if key not in {IDENTITY_TOKEN_FIELD, DOCUMENT_LOOKUP_FIELD}
            }
            result["related_profiles"] = related_profiles
            result["combined_profile"] = combine_profiles(profiles)
            result["identity_confidence"] = (
                min(confidence for _, confidence in linked) if linked else None
            )
            result["identity_links_truncated"] = links_truncated
            result["identity_link_count"] = len(linked)
            result["matched_by"] = list(result.get("matched_by", []))
            if linked and "identity" not in result["matched_by"]:
                result["matched_by"].append("identity")
            grouped_keys.add(seed_key)
            for hit, _ in linked:
                payload = hit.get("payload") or {}
                grouped_keys.add(
                    _document_key(
                        payload.get("_source_file", ""),
                        payload.get("_row_id", hit.get("id", "")),
                        payload.get("_osiris_shard"),
                    )
                )
            if linked and self.embedding_backend != "hashing":
                entity_index = len(entity_results)
                start = len(group_chunks)
                group_chunks.extend(profile_embedding_chunks(profiles))
                group_chunk_ranges.append((entity_index, start, len(group_chunks)))
            result["identity_links_scope"] = "global"
            entity_results.append(result)

        if group_chunks and processor is not None:
            group_vectors = processor.encode(group_chunks, persist_cache=False)
            normalized_query = vector / max(float(np.linalg.norm(vector)), 1e-12)
            normalized_vectors = group_vectors / np.maximum(
                np.linalg.norm(group_vectors, axis=1, keepdims=True),
                1e-12,
            )
            chunk_scores = normalized_vectors @ normalized_query
            for entity_index, start, end in group_chunk_ranges:
                entity_results[entity_index]["entity_vector_score"] = float(
                    np.max(chunk_scores[start:end])
                )

        max_fused_score = max((item["score"] for item in fused_results), default=0.0)
        for result in entity_results:
            result.setdefault("entity_vector_score", result.get("vector_score"))
            semantic_normalized = _normalized_vector_score(
                result.get("entity_vector_score"),
                hashing=self.embedding_backend == "hashing",
            )
            lexical_normalized = (
                float(result["score"]) / max_fused_score if max_fused_score > 0 else 0.0
            )
            result["hybrid_score"] = (
                0.75 * semantic_normalized + 0.25 * lexical_normalized
                if semantic_normalized is not None
                else lexical_normalized
            )

        entity_results.extend(
            {
                **item,
                "related_profiles": [],
                "combined_profile": None,
                "identity_confidence": None,
                "identity_link_count": 0,
                "identity_links_truncated": links_truncated,
                "identity_links_scope": "global",
                "entity_vector_score": item.get("vector_score"),
                "hybrid_score": (
                    0.75
                    * (
                        _normalized_vector_score(
                            item.get("vector_score"),
                            hashing=self.embedding_backend == "hashing",
                        )
                        or 0.0
                    )
                    + 0.25 * (item["score"] / max_fused_score)
                    if item.get("vector_score") is not None and max_fused_score > 0
                    else item["score"] / max_fused_score if max_fused_score > 0 else 0.0
                ),
            }
            for item in fused_results
            if _document_key(
                (item.get("payload") or {}).get("_source_file", ""),
                (item.get("payload") or {}).get("_row_id", ""),
                (item.get("payload") or {}).get("_osiris_shard"),
            )
            not in grouped_keys
        )
        results = sorted(
            entity_results,
            key=lambda item: (
                -item["hybrid_score"],
                -item["score"],
                item.get("vector_rank") or float("inf"),
            ),
        )[:limit]
        if self.embedding_cache:
            self.embedding_cache.save()
        log_stage(
            logger,
            "QDRANT_SEARCH",
            query=text,
            vector_candidates=len(vector_hits),
            lexical_candidates=len(lexical_hits),
            query_terms=len(terms),
            linked_profiles=sum(item["identity_link_count"] for item in entity_results),
            identity_links_truncated=links_truncated,
            identity_links_scope="global",
            identity_threshold=self.identity_threshold,
            results=len(results),
        )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexation et recherche des profils dans Qdrant")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--collection",
        help="Collection Qdrant (par défaut: osiris_profiles_fast pour hashing, sinon osiris_profiles)",
    )
    common.add_argument("--url", default="http://localhost:6333")
    common.add_argument("--local-path", type=Path)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument(
        "--embedding-backend",
        choices=("tei", "sentence-transformers", "hashing"),
        default="tei",
        help="TEI HTTP, Sentence Transformers/PyTorch ou recherche sparse hashing rapide",
    )
    common.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Périphérique du backend sentence-transformers",
    )
    common.add_argument(
        "--precision",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Précision Sentence Transformers (auto utilise float16 avec CUDA)",
    )
    common.add_argument(
        "--prefer-grpc",
        action="store_true",
        help="Utiliser gRPC pour accélérer les échanges avec un Qdrant distant",
    )
    common.add_argument(
        "--embedding-cache",
        type=Path,
        help="Cache optionnel de vecteurs (non disponible avec le backend hashing)",
    )
    common.add_argument("--endpoint", default=DEFAULT_TEI_ENDPOINT, help="URL de Text Embeddings Inference")
    common.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    index = sub.add_parser("index", parents=[common], help="Indexe les shards normalisés")
    index.add_argument("--shards-dir", type=Path, default=Path("cache/ingested"))
    index.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Nombre de textes encodés par micro-lot d'embedding",
    )
    index.add_argument("--read-batch-size", type=int, default=2_048, help="Taille des lots lus depuis Parquet")
    index.add_argument(
        "--embedding-concurrency",
        type=int,
        default=4,
        help="Nombre maximal de requêtes TEI simultanées (ignoré par le backend PyTorch)",
    )
    index.add_argument(
        "--upsert-batch-size",
        type=int,
        default=None,
        help="Points par requête Qdrant (défaut: 1 024 en hashing, 128 sinon)",
    )
    search = sub.add_parser("search", parents=[common], help="Recherche les profils proches")
    search.add_argument("text")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--candidates", type=int, default=200)
    search.add_argument("--lexical-index", type=Path, default=DEFAULT_LEXICAL_INDEX)
    search.add_argument(
        "--identity-threshold",
        type=float,
        default=0.72,
        help="Confiance minimale après correspondance exacte des champs (> 0 et <= 1)",
    )
    search.add_argument(
        "--max-linked-profiles",
        type=int,
        default=1_000,
        help="Nombre maximal de fiches récupérées pour l’expansion d’identité",
    )
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    pipeline = QdrantProfilePipeline(
        collection=args.collection
        or (
            "osiris_profiles_fast"
            if args.embedding_backend == "hashing"
            else "osiris_profiles"
        ),
        url=None if args.local_path else args.url,
        local_path=args.local_path,
        model_name=args.model,
        embedding_cache=args.embedding_cache,
        endpoint=args.endpoint,
        embedding_backend=args.embedding_backend,
        device=args.device,
        precision=args.precision,
        prefer_grpc=args.prefer_grpc,
        identity_threshold=args.identity_threshold if args.command == "search" else 0.72,
        max_linked_profiles=args.max_linked_profiles if args.command == "search" else 1_000,
    )
    try:
        if args.command == "index":
            print(
                pipeline.index(
                    args.batch_size,
                    args.read_batch_size,
                    args.embedding_concurrency,
                    args.upsert_batch_size,
                )
            )
        else:
            import json
            print(
                json.dumps(
                    pipeline.search(args.text, args.limit, args.candidates, args.lexical_index),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
