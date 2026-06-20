"""
Embedding generation using BAAI/bge-small-en-v1.5.

Loads the model once (via sentence-transformers) and provides:
    - ``encode()``     — batch-encode document chunks for ingestion.
    - ``query_encode()`` — encode a search query with the BGE query prefix.

Both methods return L2-normalized vectors so that FAISS inner-product
search (``IndexFlatIP``) computes exact cosine similarity.

The model loads in ~2 seconds on first use and uses ~130 MB of RAM.
It is cached by HuggingFace after the first download (~130 MB disk).
"""

from __future__ import annotations

import logging

import numpy as np

from rag.config import Settings, settings as default_settings
from rag.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# BGE query instruction prefix — improves retrieval quality.
# Applied to queries (not documents) before encoding.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# The embedding dimension of bge-small-en-v1.5.
_EMBEDDING_DIM = 384


class Embedder:
    """Embedding model wrapper for bge-small-en-v1.5.

    Parameters
    ----------
    config:
        Application settings (provides ``embedding_model`` and
        ``embedding_batch_size``). Defaults to the global singleton.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self._model = None
        self._dimension = _EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the sentence-transformers model (lazy, once per process)."""
        if self._model is not None:
            return

        model_name = self.config.embedding_model
        logger.info("Loading embedding model: %s", model_name)

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name,
                device="cpu",
            )
            self._dimension = self._model.get_embedding_dimension()
            logger.info(
                "Embedding model loaded: dim=%d", self._dimension
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "Failed to load embedding model '{}': {}".format(model_name, exc)
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for bge-small-en-v1.5)."""
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of document texts into embeddings.

        Used during ingestion to embed chunks. Vectors are L2-normalized
        so that FAISS inner-product search yields cosine similarity.

        Parameters
        ----------
        texts:
            List of text strings (chunks) to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), 384)``, dtype ``float32``, L2-normalized.

        Raises
        ------
        EmbeddingError
            If encoding fails.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        self._load_model()

        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.config.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vectors.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "Failed to encode {} texts: {}".format(len(texts), exc)
            ) from exc

    def query_encode(self, query: str) -> np.ndarray:
        """Encode a search query into an embedding.

        Prepends the BGE query instruction prefix before encoding to
        improve retrieval quality. Returns a single L2-normalized vector.

        Parameters
        ----------
        query:
            User's search question.

        Returns
        -------
        np.ndarray
            Shape ``(1, 384)``, dtype ``float32``, L2-normalized.

        Raises
        ------
        EmbeddingError
            If encoding fails.
        """
        if not query or not query.strip():
            raise EmbeddingError("Cannot encode an empty query")

        self._load_model()

        prefixed_query = _BGE_QUERY_PREFIX + query.strip()
        try:
            vector = self._model.encode(
                prefixed_query,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vector.reshape(1, -1).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                "Failed to encode query: {}".format(exc)
            ) from exc
