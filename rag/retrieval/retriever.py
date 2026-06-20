"""
Retrieval orchestration: query → embed → search → filter → rank.

The ``Retriever`` ties together the embedding model and FAISS vector
store into a single ``retrieve()`` call that:

1. Encodes the user's query with the BGE query prefix.
2. Searches the FAISS index for the top-K most similar vectors.
3. Filters out results below ``score_threshold`` (cosine similarity).
4. Returns a ``RetrievalResult`` with the surviving chunks, their
   scores, and provenance metadata (source_file, page) for citations.

This is the single entry point used by the generation layer and the
chat API — neither of those should touch ``Embedder`` or ``FAISSStore``
directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rag.config import Settings, settings as default_settings
from rag.exceptions import RetrievalError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk with its similarity score.

    Attributes
    ----------
    text:
        The chunk text (already cleaned/normalized during ingestion).
    source_file:
        Original PDF filename (basename, no directory).
    page:
        1-based page number in the source PDF.
    score:
        Cosine similarity score (0.0–1.0, higher is better).
    chunk_id:
        Unique chunk identifier from ingestion.
    metadata:
        Full metadata dict from the FAISS sidecar.
    """

    text: str
    source_file: str
    page: int
    score: float
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of a retrieval operation.

    Attributes
    ----------
    query:
        The original user query.
    chunks:
        Filtered, ranked list of retrieved chunks.
    total_found:
        Number of raw results from FAISS (before threshold filtering).
    total_returned:
        Number of chunks returned (after filtering).
    latency_ms:
        Wall-clock time for embed + search in milliseconds.
    """

    query: str
    chunks: list[RetrievedChunk]
    total_found: int
    total_returned: int
    latency_ms: float


class Retriever:
    """Orchestrates query encoding, FAISS search, and score filtering.

    Parameters
    ----------
    embedder:
        An ``Embedder`` instance (provides ``query_encode``).
    store:
        A ``FAISSStore`` instance (provides ``search``). Must be
        initialized and (optionally) loaded from disk beforehand.
    config:
        Application settings (provides ``top_k`` and ``score_threshold``).
        Defaults to the global singleton.
    """

    def __init__(
        self,
        embedder: Any,
        store: Any,
        config: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or default_settings

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve the top-K most relevant chunks for a query.

        Parameters
        ----------
        query:
            The user's question.
        top_k:
            Override for the number of chunks to retrieve. Defaults to
            ``config.top_k`` (5).

        Returns
        -------
        RetrievalResult
            Filtered and ranked chunks with scores.

        Raises
        ------
        RetrievalError
            If query encoding or FAISS search fails.
        """
        if not query or not query.strip():
            raise RetrievalError("Query must not be empty")

        effective_k = top_k if top_k is not None else self.config.top_k
        threshold = self.config.score_threshold

        logger.info(
            "Retrieving chunks for query (%d chars, top_k=%d, threshold=%.2f)",
            len(query),
            effective_k,
            threshold,
        )

        t0 = time.perf_counter()

        # Step 1: encode the query.
        try:
            query_vector = self.embedder.query_encode(query)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                "Failed to encode query: {}".format(exc)
            ) from exc

        # Step 2: search FAISS.
        try:
            raw_results = self.store.search(query_vector, top_k=effective_k)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                "FAISS search failed: {}".format(exc)
            ) from exc

        total_found = len(raw_results)

        # Step 3: filter by score threshold.
        filtered = [
            r for r in raw_results if r.score >= threshold
        ]

        # Step 4: build RetrievedChunk objects.
        chunks: list[RetrievedChunk] = []
        for r in filtered:
            meta = r.metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=meta.get("text", ""),
                    source_file=meta.get("source_file", "unknown"),
                    page=meta.get("page", 0),
                    score=round(r.score, 4),
                    chunk_id=r.chunk_id,
                    metadata=meta,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "Retrieved %d/%d chunks above threshold %.2f in %.1f ms",
            len(chunks),
            total_found,
            threshold,
            latency_ms,
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_found=total_found,
            total_returned=len(chunks),
            latency_ms=latency_ms,
        )
