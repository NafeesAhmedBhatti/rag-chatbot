"""
FAISS vector store for persistent embedding storage.

Manages a ``faiss.IndexFlatIP`` (exact inner-product search on
L2-normalized vectors = cosine similarity) paired with a JSON metadata
sidecar. Supports:

    - ``add()``            — insert vectors + metadata.
    - ``search()``         — top-K cosine similarity search.
    - ``remove_by_source()`` — delete all chunks belonging to a document.
    - ``save()`` / ``load()`` — persist/read the index + metadata.

Design notes
------------
- We use ``IndexIDMap2(IndexFlatIP)`` so that each vector has a
  stable integer ID. ``IDMap2`` supports ``remove_ids`` (for
  re-ingestion), unlike a bare ``IndexFlatIP``.
- The metadata sidecar is a JSON list indexed by the same integer IDs.
  When a vector is removed, we rebuild both the index and the metadata
  list so IDs stay contiguous.
- At ~6,000 vectors (10 PDFs × 200 pages × ~3 chunks/page) the flat
  index searches in <1 ms — no need for approximate search (IVF/HNSW).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.config import Settings, settings as default_settings
from rag.exceptions import RetrievalError

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval result from the FAISS store.

    Attributes
    ----------
    chunk_id:
        Unique identifier from the chunk metadata.
    score:
        Cosine similarity score (0.0–1.0, higher is better).
    metadata:
        Full chunk metadata dict (source_file, page, etc.).
    """

    chunk_id: str
    score: float
    metadata: dict[str, Any]


class FAISSStore:
    """FAISS-backed vector store with JSON metadata sidecar.

    Parameters
    ----------
    config:
        Application settings (provides ``index_dir`` paths). Defaults
        to the global singleton.
    dimension:
        Embedding dimension. Defaults to 384 (bge-small-en-v1.5).
    """

    def __init__(
        self,
        config: Settings | None = None,
        dimension: int = 384,
    ) -> None:
        self.config = config or default_settings
        self.dimension = dimension

        # The underlying FAISS index (IDMap-wrapped flat IP index).
        self._index: faiss.IndexIDMap2 | None = None
        # Metadata list, keyed by FAISS vector ID.
        # Stored as dict[int, dict] for O(1) lookup and easy removal.
        self._metadata: dict[int, dict[str, Any]] = {}

        # The next available integer ID for new vectors.
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create an empty FAISS index in memory."""
        base = faiss.IndexFlatIP(self.dimension)
        self._index = faiss.IndexIDMap2(base)
        self._metadata = {}
        self._next_id = 0
        logger.info(
            "Initialized empty FAISS index (dim=%d)", self.dimension
        )

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_ready(self) -> bool:
        """True if the index is initialized and ready for operations."""
        return self._index is not None

    # ------------------------------------------------------------------
    # Add vectors
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add vectors and their metadata to the index.

        Parameters
        ----------
        vectors:
            Shape ``(n, dimension)``, dtype ``float32``, L2-normalized.
        metadata:
            List of ``n`` metadata dicts (one per vector).

        Raises
        ------
        RetrievalError
            If the index is not initialized, or the input shapes mismatch.
        """
        if self._index is None:
            raise RetrievalError("FAISS index not initialized; call initialize() first")

        if len(vectors) == 0:
            return

        if vectors.shape[1] != self.dimension:
            raise RetrievalError(
                "Vector dimension mismatch: expected {}, got {}".format(
                    self.dimension, vectors.shape[1]
                )
            )

        if len(metadata) != len(vectors):
            raise RetrievalError(
                "Metadata count ({}) does not match vector count ({})".format(
                    len(metadata), len(vectors)
                )
            )

        # Assign sequential IDs.
        ids = np.arange(
            self._next_id, self._next_id + len(vectors), dtype=np.int64
        )

        self._index.add_with_ids(vectors, ids)

        # Store metadata keyed by ID.
        for id_val, meta in zip(ids, metadata, strict=True):
            self._metadata[int(id_val)] = meta

        self._next_id += len(vectors)
        logger.debug(
            "Added %d vectors (total: %d)", len(vectors), self.size
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[SearchResult]:
        """Search for the top-K most similar vectors.

        Parameters
        ----------
        query_vector:
            Shape ``(1, dimension)``, dtype ``float32``, L2-normalized.
        top_k:
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Results sorted by descending cosine similarity. Fewer than
            ``top_k`` results if the index has fewer entries.

        Raises
        ------
        RetrievalError
            If the index is not initialized or empty.
        """
        if self._index is None:
            raise RetrievalError("FAISS index not initialized; call initialize() first")

        if self.size == 0:
            return []

        if query_vector.shape[1] != self.dimension:
            raise RetrievalError(
                "Query dimension mismatch: expected {}, got {}".format(
                    self.dimension, query_vector.shape[1]
                )
            )

        # Search up to top_k (but not more than what's in the index).
        k = min(top_k, self.size)
        scores, ids = self._index.search(query_vector, k)

        results: list[SearchResult] = []
        for score, faiss_id in zip(scores[0], ids[0], strict=True):
            if faiss_id == -1:
                continue  # FAISS returns -1 for missing slots

            meta = self._metadata.get(int(faiss_id), {})
            results.append(
                SearchResult(
                    chunk_id=meta.get("chunk_id", ""),
                    score=float(score),
                    metadata=meta,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Remove by source document
    # ------------------------------------------------------------------

    def remove_by_source(self, source_file: str) -> int:
        """Remove all vectors belonging to a source document.

        Used for idempotent re-ingestion: before adding new chunks for
        a file, remove the old ones.

        Parameters
        ----------
        source_file:
            The ``source_file`` value from chunk metadata.

        Returns
        -------
        int
            Number of vectors removed.
        """
        if self._index is None or self.size == 0:
            return 0

        # Find all IDs whose metadata matches the source_file.
        ids_to_remove = [
            id_val
            for id_val, meta in self._metadata.items()
            if meta.get("source_file") == source_file
        ]

        if not ids_to_remove:
            return 0

        selector = faiss.IDSelectorBatch(
            np.array(ids_to_remove, dtype=np.int64)
        )
        removed = self._index.remove_ids(selector)

        # Rebuild metadata dict without the removed IDs.
        for id_val in ids_to_remove:
            del self._metadata[id_val]

        logger.info(
            "Removed %d vectors for source '%s' (remaining: %d)",
            removed,
            source_file,
            self.size,
        )
        return removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk.

        Writes two files:
            - ``faiss.index``   — FAISS binary index.
            - ``metadata.json`` — JSON list of metadata + ID mapping.

        Raises
        ------
        RetrievalError
            If the index is not initialized.
        """
        if self._index is None:
            raise RetrievalError("Cannot save an uninitialized index")

        index_path = self.config.faiss_index_path
        metadata_path = self.config.metadata_path

        # Ensure the directory exists.
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))

        # Serialize metadata as a list of {id, metadata} pairs.
        metadata_list = [
            {"id": id_val, "metadata": meta}
            for id_val, meta in sorted(self._metadata.items())
        ]
        metadata_path.write_text(
            json.dumps(
                {
                    "dimension": self.dimension,
                    "next_id": self._next_id,
                    "entries": metadata_list,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Saved FAISS index (%d vectors) to %s",
            self.size,
            index_path.name,
        )

    def load(self) -> bool:
        """Load the FAISS index and metadata from disk.

        Returns
        -------
        bool
            ``True`` if loaded successfully, ``False`` if no index file
            exists (caller should call ``initialize()`` for a fresh index).

        Raises
        ------
        RetrievalError
            If the index file exists but is corrupted or incompatible.
        """
        index_path = self.config.faiss_index_path
        metadata_path = self.config.metadata_path

        if not index_path.exists():
            logger.info("No existing FAISS index found at %s", index_path)
            return False

        try:
            self._index = faiss.read_index(str(index_path))
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(
                "Failed to read FAISS index '{}': {}".format(index_path, exc)
            ) from exc

        # Load metadata sidecar.
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.dimension = data.get("dimension", self.dimension)
            self._next_id = data.get("next_id", 0)
            self._metadata = {
                entry["id"]: entry["metadata"]
                for entry in data.get("entries", [])
            }
        else:
            logger.warning(
                "Metadata sidecar not found at %s; index loaded without metadata",
                metadata_path,
            )
            self._metadata = {}
            self._next_id = self._index.ntotal

        logger.info(
            "Loaded FAISS index: %d vectors (dim=%d)",
            self.size,
            self.dimension,
        )
        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the index."""
        index_size_mb = 0.0
        if self.config.faiss_index_path.exists():
            index_size_mb = round(
                self.config.faiss_index_path.stat().st_size / (1024 * 1024), 4
            )

        # Count distinct source documents.
        sources = {meta.get("source_file", "") for meta in self._metadata.values()}

        return {
            "total_documents": len(sources),
            "total_chunks": self.size,
            "index_size_mb": index_size_mb,
            "dimension": self.dimension,
        }
