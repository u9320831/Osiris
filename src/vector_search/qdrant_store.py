"""Qdrant-backed persistent vector store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix

from src.semantic.identity_linking import DOCUMENT_LOOKUP_FIELD, IDENTITY_TOKEN_FIELD
from src.utils.logging_config import log_stage


logger = logging.getLogger(__name__)


class QdrantStore:
    """Small compatibility wrapper around the Qdrant Python client."""

    def __init__(
        self,
        collection: str = "osiris_profiles",
        url: str | None = "http://localhost:6333",
        local_path: str | Path | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise RuntimeError(
                "qdrant-client est requis : pip install -r requirements.txt"
            ) from error
        self.models = models
        self.collection = collection
        self._local = local_path is not None
        if local_path is not None:
            self.client = QdrantClient(path=str(local_path))
            self.connection = f"local:{local_path}"
        elif url:
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc)
            self.connection = url
        else:
            raise ValueError("url ou local_path doit être défini")
        self._collection_ready = False
        self._sparse_collection_ready = False
        self._identity_index_ready = False

    def close(self) -> None:
        self.client.close()

    def _ensure_identity_payload_index(self) -> None:
        if self._identity_index_ready:
            return
        if self._local:
            logger.warning(
                "Local Qdrant does not index payload fields; identity lookups may be slower"
            )
            self._identity_index_ready = True
            return
        info = self.client.get_collection(self.collection)
        for field_name in (IDENTITY_TOKEN_FIELD, DOCUMENT_LOOKUP_FIELD):
            if field_name not in info.payload_schema:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=self.models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        self._identity_index_ready = True

    def ensure_collection(self, vector_size: int, distance: str = "cosine") -> None:
        if self._collection_ready:
            self._ensure_identity_payload_index()
            return
        distance_map = {
            "cosine": self.models.Distance.COSINE,
            "dot": self.models.Distance.DOT,
            "euclidean": self.models.Distance.EUCLID,
        }
        if distance not in distance_map:
            raise ValueError(f"Distance inconnue : {distance}")
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            current_size = info.config.params.vectors.size
            if current_size != vector_size:
                raise ValueError(
                    f"La collection {self.collection} contient des vecteurs de taille "
                    f"{current_size}, taille reçue : {vector_size}"
                )
            self._collection_ready = True
            self._ensure_identity_payload_index()
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=self.models.VectorParams(size=vector_size, distance=distance_map[distance]),
        )
        self._collection_ready = True
        self._ensure_identity_payload_index()
        log_stage(logger, "QDRANT_COLLECTION_CREATED", collection=self.collection, size=vector_size)

    def ensure_sparse_collection(self) -> None:
        if self._sparse_collection_ready:
            return
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            sparse_vectors = info.config.params.sparse_vectors or {}
            if "profile_terms" not in sparse_vectors:
                raise ValueError(
                    f"La collection {self.collection} n'est pas un index sparse; "
                    "utilisez un nom de collection distinct pour le backend hashing."
                )
            self._sparse_collection_ready = True
            self._ensure_identity_payload_index()
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={},
            sparse_vectors_config={
                "profile_terms": self.models.SparseVectorParams(
                    index=self.models.SparseIndexParams(on_disk=True)
                )
            },
            on_disk_payload=True,
        )
        self._sparse_collection_ready = True
        self._ensure_identity_payload_index()
        log_stage(logger, "QDRANT_SPARSE_COLLECTION_CREATED", collection=self.collection)

    def upsert_sparse(
        self,
        ids: Iterable[int],
        vectors: csr_matrix,
        payloads: Iterable[dict[str, Any]],
        batch_size: int = 1_024,
        wait: bool = False,
    ) -> int:
        ids = list(ids)
        payloads = list(payloads)
        vectors = vectors.tocsr()
        if len(ids) != vectors.shape[0] or len(ids) != len(payloads):
            raise ValueError("ids, vecteurs creux et payloads doivent avoir la même longueur")
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if not len(ids):
            return 0
        self.ensure_sparse_collection()

        def points() -> Iterable[Any]:
            for index, (point_id, payload) in enumerate(zip(ids, payloads, strict=True)):
                start = vectors.indptr[index]
                end = vectors.indptr[index + 1]
                yield self.models.PointStruct(
                    id=point_id,
                    vector={
                        "profile_terms": self.models.SparseVector(
                            indices=vectors.indices[start:end].tolist(),
                            values=vectors.data[start:end].tolist(),
                        )
                    },
                    payload=payload,
                )

        self.client.upload_points(
            collection_name=self.collection,
            points=points(),
            batch_size=batch_size,
            parallel=1,
            wait=wait,
        )
        return len(ids)

    def search_sparse(self, vector: csr_matrix, limit: int = 10) -> list[dict[str, Any]]:
        vector = vector.tocsr()
        if vector.shape[0] != 1:
            raise ValueError("La recherche sparse attend un seul vecteur")
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.models.SparseVector(
                indices=vector.indices.tolist(),
                values=vector.data.tolist(),
            ),
            using="profile_terms",
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload or {}}
            for point in response.points
        ]

    def retrieve(self, ids: Iterable[int]) -> list[dict[str, Any]]:
        ids = list(ids)
        if not ids:
            return []
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": point.id, "payload": point.payload or {}}
            for point in points
        ]

    def upsert(
        self,
        ids: Iterable[str | int],
        vectors: np.ndarray,
        payloads: Iterable[dict[str, Any]],
        batch_size: int = 256,
        wait: bool = False,
    ) -> int:
        ids = list(ids)
        payloads = list(payloads)
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors et payloads doivent avoir la même longueur")
        if not len(vectors):
            return 0
        self.ensure_collection(vectors.shape[1])
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            points = [
                self.models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
                for point_id, vector, payload in zip(ids[start:end], vectors[start:end], payloads[start:end], strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=wait)
        return len(ids)

    def search(self, vector: np.ndarray, limit: int = 10, query_filter: Any = None) -> list[dict[str, Any]]:
        vector = np.asarray(vector, dtype=np.float32).tolist()
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload or {}}
            for point in points
        ]

    def search_related_profiles(
        self,
        tokens: Iterable[str],
        limit: int = 1_000,
        page_size: int = 256,
    ) -> tuple[list[dict[str, Any]], bool]:
        return self._scroll_by_tokens(IDENTITY_TOKEN_FIELD, tokens, limit, page_size)

    def search_profiles_by_document_keys(
        self,
        keys: Iterable[str],
        limit: int = 1_000,
        page_size: int = 256,
    ) -> tuple[list[dict[str, Any]], bool]:
        return self._scroll_by_tokens(DOCUMENT_LOOKUP_FIELD, keys, limit, page_size)

    def _scroll_by_tokens(
        self,
        field_name: str,
        tokens: Iterable[str],
        limit: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        if limit <= 0 or page_size <= 0:
            raise ValueError("limit et page_size doivent être strictement positifs")
        token_values = sorted(set(tokens))
        if not token_values:
            return [], False
        if not self.client.collection_exists(self.collection):
            return [], False

        query_filter = self.models.Filter(
            must=[
                self.models.FieldCondition(
                    key=field_name,
                    match=self.models.MatchAny(any=token_values),
                )
            ]
        )
        profiles: list[dict[str, Any]] = []
        offset: Any = None
        truncated = False
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=min(page_size, limit - len(profiles)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            profiles.extend(
                {"id": point.id, "payload": point.payload or {}}
                for point in points
            )
            if next_offset is None:
                break
            if len(profiles) >= limit:
                truncated = True
                break
            offset = next_offset
        return profiles, truncated
