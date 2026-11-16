"""HTTP client for a Text Embeddings Inference service."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Protocol, Sequence

import httpx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_MODEL = os.environ.get(
    "OSIRIS_TEI_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_TEI_ENDPOINT = os.environ.get("OSIRIS_TEI_URL", "http://localhost:8080")


class EmbeddingBackend(Protocol):
    model_name: str

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        normalize: bool = True,
        concurrency: int = 1,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_TEI_ENDPOINT,
        timeout: float = 300.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout doit être strictement positif")
        self.model_name = model_name
        self.endpoint = endpoint.strip().rstrip("/")
        if not self.endpoint:
            raise ValueError("endpoint TEI ne peut pas être vide")
        self._client = httpx.Client(timeout=timeout)
        self._model_checked = False

    def _check_model(self) -> None:
        if self._model_checked:
            return
        try:
            response = self._client.get(f"{self.endpoint}/info")
            response.raise_for_status()
            info = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Impossible de joindre TEI à {self.endpoint}. "
                "Démarrez-le avec `docker compose --profile embeddings up --detach tei` "
                "ou vérifiez --endpoint / OSIRIS_TEI_URL."
            ) from error
        if not isinstance(info, dict):
            raise RuntimeError(f"Réponse /info TEI invalide depuis {self.endpoint}")
        served_model = info.get("model_id")
        if served_model and served_model != self.model_name:
            raise RuntimeError(
                f"Le modèle TEI actif est {served_model!r}, mais "
                f"{self.model_name!r} est configuré. Alignez --model et OSIRIS_TEI_MODEL."
            )
        self._model_checked = True

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        normalize: bool = True,
        concurrency: int = 1,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if concurrency <= 0:
            raise ValueError("concurrency doit être strictement positif")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._check_model()
        starts = range(0, len(texts), batch_size)
        inputs = (list(texts[start : start + batch_size]) for start in starts)
        if concurrency == 1:
            batches = [
                self._encode_batch(batch, normalize, start)
                for batch, start in zip(inputs, starts, strict=True)
            ]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                batches = list(
                    executor.map(
                        self._encode_batch,
                        inputs,
                        repeat(normalize),
                        starts,
                    )
                )

        dimensions = {vectors.shape[1] for vectors in batches}
        if len(dimensions) != 1:
            raise RuntimeError("TEI a retourné des dimensions de vecteurs incohérentes")
        return np.concatenate(batches, axis=0)

    def _encode_batch(
        self,
        inputs: Sequence[str],
        normalize: bool,
        start: int,
    ) -> np.ndarray:
        try:
            response = self._client.post(
                f"{self.endpoint}/embed",
                json={"inputs": list(inputs)},
            )
            response.raise_for_status()
            vectors = np.asarray(response.json(), dtype=np.float32)
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"La requête TEI a échoué pour le lot commençant à la ligne {start}."
            ) from error
        if vectors.ndim != 2 or vectors.shape[0] != len(inputs):
            raise RuntimeError(
                "Réponse TEI invalide : le nombre de vecteurs ne correspond pas "
                f"au lot ({vectors.shape!r}, attendu {len(inputs)} lignes)."
            )
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, np.finfo(np.float32).tiny)
        return vectors

    def close(self) -> None:
        self._client.close()


class SentenceTransformerEmbeddingModel:
    """Run sentence-transformers directly, using a local CUDA device when selected."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        precision: str = "auto",
        cache_folder: str | Path = Path("cache/tei"),
    ) -> None:
        if device not in {"cuda", "cpu"}:
            raise ValueError("device doit être 'cuda' ou 'cpu'")
        if precision not in {"auto", "float16", "float32"}:
            raise ValueError("precision doit être 'auto', 'float16' ou 'float32'")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Le backend PyTorch nécessite sentence-transformers et PyTorch "
                "avec CUDA; installez une version PyTorch correspondant à votre pilote."
            ) from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA n'est pas disponible dans cet environnement Python. "
                "Vérifiez que PyTorch avec CUDA et le pilote NVIDIA sont installés."
            )
        if precision == "float16" and device != "cuda":
            raise ValueError("La précision float16 nécessite le périphérique CUDA")
        use_float16 = precision == "float16" or (precision == "auto" and device == "cuda")
        self.model_name = model_name
        self.device = device
        self.precision = "float16" if use_float16 else "float32"
        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_folder),
            model_kwargs={"torch_dtype": torch.float16} if use_float16 else None,
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        normalize: bool = True,
        concurrency: int = 1,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if concurrency <= 0:
            raise ValueError("concurrency doit être strictement positif")
        if concurrency != 1:
            raise ValueError(
                "Le backend sentence-transformers exécute ses lots directement "
                "sur le périphérique; concurrency doit être égal à 1."
            )
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise RuntimeError(
                "Réponse sentence-transformers invalide : "
                f"{embeddings.shape!r}, attendu {len(texts)} lignes."
            )
        return embeddings

    def close(self) -> None:
        del self._model


class HashingEmbeddingModel:
    """Fast sparse lexical vectors for approximate Qdrant retrieval."""

    model_name = "osiris-hashing-unigram-1048576-v1"

    def __init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=1_048_576,
            norm="l2",
            alternate_sign=False,
            ngram_range=(1, 1),
            strip_accents="unicode",
            dtype=np.float32,
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        normalize: bool = True,
        concurrency: int = 1,
    ) -> csr_matrix:
        if batch_size <= 0:
            raise ValueError("batch_size doit être strictement positif")
        if concurrency <= 0:
            raise ValueError("concurrency doit être strictement positif")
        if not texts:
            return csr_matrix((0, self._vectorizer.n_features), dtype=np.float32)
        if not normalize:
            raise ValueError("Les vecteurs creux hashing sont toujours normalisés")
        vectors = self._vectorizer.transform(texts).tocsr()
        vectors.sort_indices()
        return vectors

    def close(self) -> None:
        return None
