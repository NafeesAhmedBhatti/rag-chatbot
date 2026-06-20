"""
Tests for rag.vectorstore.faiss_store.FAISSStore.

Covers:
    - Initialization
    - Add vectors + metadata
    - Search (top-K, scores, correct ordering)
    - Remove by source document (re-ingestion)
    - Persistence (save/load roundtrip)
    - Empty index handling
    - Error handling (uninitialized, dimension mismatch)
    - Stats
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest

from rag.config import Settings
from rag.exceptions import RetrievalError
from rag.vectorstore.faiss_store import FAISSStore, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normalized_vectors(n: int, dim: int = 384) -> np.ndarray:
    """Generate n random L2-normalized vectors."""
    vectors = np.random.randn(n, dim).astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def _make_metadata(n: int, source: str = "test.pdf", page: int = 1) -> list[dict]:
    return [
        {
            "chunk_id": f"{source}_p{page}_c{i}",
            "source_file": source,
            "page": page,
            "text": f"chunk text {i}",
            "chunk_index": i,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> FAISSStore:
    """A fresh FAISSStore initialized with a temp directory."""
    config = Settings(index_dir=tmp_path / "index")
    s = FAISSStore(config=config, dimension=384)
    s.initialize()
    return s


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """FAISSStore initialization."""

    def test_initialize_creates_index(self, store: FAISSStore) -> None:
        assert store.is_ready
        assert store.size == 0

    def test_dimension_stored(self, store: FAISSStore) -> None:
        assert store.dimension == 384


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

class TestAdd:
    """Adding vectors and metadata."""

    def test_add_single_batch(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(5)
        store.add(vectors, _make_metadata(5))
        assert store.size == 5

    def test_add_multiple_batches(self, store: FAISSStore) -> None:
        # Add in two batches
        store.add(_make_normalized_vectors(3), _make_metadata(3))
        store.add(_make_normalized_vectors(4), _make_metadata(4))
        assert store.size == 7

    def test_add_empty_does_nothing(self, store: FAISSStore) -> None:
        store.add(
            np.zeros((0, 384), dtype=np.float32),
            [],
        )
        assert store.size == 0

    def test_add_dimension_mismatch_raises(self, store: FAISSStore) -> None:
        bad_vectors = np.zeros((2, 128), dtype=np.float32)  # wrong dim
        with pytest.raises(RetrievalError, match="dimension mismatch"):
            store.add(bad_vectors, _make_metadata(2))

    def test_add_metadata_count_mismatch_raises(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(5)
        with pytest.raises(RetrievalError, match="does not match"):
            store.add(vectors, _make_metadata(3))

    def test_add_uninitialized_raises(self, tmp_path: Path) -> None:
        config = Settings(index_dir=tmp_path / "index")
        s = FAISSStore(config=config)
        with pytest.raises(RetrievalError, match="not initialized"):
            s.add(_make_normalized_vectors(1), _make_metadata(1))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch:
    """Top-K search with similarity scores."""

    def test_returns_correct_count(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(10)
        store.add(vectors, _make_metadata(10))
        query = vectors[0:1].copy()
        results = store.search(query, top_k=3)
        assert len(results) == 3

    def test_top_result_is_query_itself(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(10)
        store.add(vectors, _make_metadata(10))
        query = vectors[3:4].copy()  # search for vector at index 3
        results = store.search(query, top_k=1)
        assert results[0].chunk_id == "test.pdf_p1_c3"
        assert results[0].score > 0.99  # self-similarity ≈ 1.0

    def test_results_sorted_by_score_descending(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(20)
        store.add(vectors, _make_metadata(20))
        query = vectors[0:1].copy()
        results = store.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_index(self, store: FAISSStore) -> None:
        """Asking for more results than exist returns what's available."""
        vectors = _make_normalized_vectors(3)
        store.add(vectors, _make_metadata(3))
        query = vectors[0:1].copy()
        results = store.search(query, top_k=10)
        assert len(results) == 3

    def test_empty_index_returns_empty_list(self, store: FAISSStore) -> None:
        query = _make_normalized_vectors(1)
        results = store.search(query, top_k=5)
        assert results == []

    def test_search_uninitialized_raises(self, tmp_path: Path) -> None:
        config = Settings(index_dir=tmp_path / "index")
        s = FAISSStore(config=config)
        with pytest.raises(RetrievalError, match="not initialized"):
            s.search(_make_normalized_vectors(1), top_k=1)

    def test_search_result_has_metadata(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(5)
        store.add(vectors, _make_metadata(5))
        query = vectors[0:1].copy()
        results = store.search(query, top_k=1)
        assert "source_file" in results[0].metadata
        assert "page" in results[0].metadata
        assert "text" in results[0].metadata


# ---------------------------------------------------------------------------
# Remove by source
# ---------------------------------------------------------------------------

class TestRemoveBySource:
    """Removing vectors by source document."""

    def test_remove_all_from_source(self, store: FAISSStore) -> None:
        store.add(_make_normalized_vectors(5), _make_metadata(5, "a.pdf"))
        store.add(_make_normalized_vectors(3), _make_metadata(3, "b.pdf"))
        removed = store.remove_by_source("a.pdf")
        assert removed == 5
        assert store.size == 3

    def test_remove_nonexistent_source(self, store: FAISSStore) -> None:
        store.add(_make_normalized_vectors(5), _make_metadata(5))
        removed = store.remove_by_source("nonexistent.pdf")
        assert removed == 0
        assert store.size == 5

    def test_remove_from_empty_index(self, store: FAISSStore) -> None:
        removed = store.remove_by_source("any.pdf")
        assert removed == 0

    def test_search_works_after_remove(self, store: FAISSStore) -> None:
        vectors = _make_normalized_vectors(10)
        store.add(vectors[:5], _make_metadata(5, "a.pdf"))
        store.add(vectors[5:], _make_metadata(5, "b.pdf"))
        store.remove_by_source("a.pdf")
        # Search for a vector from b.pdf
        query = vectors[7:8].copy()
        results = store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0].metadata["source_file"] == "b.pdf"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """Save and load roundtrip."""

    def test_save_creates_files(self, store: FAISSStore, tmp_path: Path) -> None:
        store.add(_make_normalized_vectors(5), _make_metadata(5))
        store.save()
        assert store.config.faiss_index_path.exists()
        assert store.config.metadata_path.exists()

    def test_load_rebuilds_index(self, tmp_path: Path) -> None:
        config = Settings(index_dir=tmp_path / "index")

        # Create and save
        store1 = FAISSStore(config=config, dimension=384)
        store1.initialize()
        vectors = _make_normalized_vectors(10)
        store1.add(vectors, _make_metadata(10))
        store1.save()

        # Load into a new instance
        store2 = FAISSStore(config=config, dimension=384)
        loaded = store2.load()
        assert loaded is True
        assert store2.size == 10

    def test_search_after_reload(self, tmp_path: Path) -> None:
        config = Settings(index_dir=tmp_path / "index")

        store1 = FAISSStore(config=config, dimension=384)
        store1.initialize()
        vectors = _make_normalized_vectors(10)
        store1.add(vectors, _make_metadata(10))
        store1.save()

        store2 = FAISSStore(config=config, dimension=384)
        store2.load()
        query = vectors[5:6].copy()
        results = store2.search(query, top_k=1)
        assert results[0].chunk_id == "test.pdf_p1_c5"
        assert results[0].score > 0.99

    def test_load_nonexistent_returns_false(self, tmp_path: Path) -> None:
        config = Settings(index_dir=tmp_path / "index")
        store = FAISSStore(config=config, dimension=384)
        result = store.load()
        assert result is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    """Index statistics."""

    def test_stats_empty_index(self, store: FAISSStore) -> None:
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0

    def test_stats_with_data(self, store: FAISSStore) -> None:
        store.add(_make_normalized_vectors(5), _make_metadata(5, "a.pdf"))
        store.add(_make_normalized_vectors(3), _make_metadata(3, "b.pdf"))
        store.save()
        stats = store.get_stats()
        assert stats["total_chunks"] == 8
        assert stats["total_documents"] == 2

    def test_stats_includes_dimension(self, store: FAISSStore) -> None:
        stats = store.get_stats()
        assert stats["dimension"] == 384


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

class TestSearchResult:
    """SearchResult dataclass behavior."""

    def test_fields(self) -> None:
        result = SearchResult(
            chunk_id="test_p1_c0",
            score=0.95,
            metadata={"source_file": "test.pdf", "page": 1},
        )
        assert result.chunk_id == "test_p1_c0"
        assert result.score == 0.95
        assert result.metadata["source_file"] == "test.pdf"
