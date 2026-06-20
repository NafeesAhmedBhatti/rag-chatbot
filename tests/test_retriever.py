"""
Unit tests for rag.retrieval.retriever.Retriever.

Uses lightweight stubs for Embedder and FAISSStore so tests run fast
without loading the real BGE model or FAISS index.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.exceptions import RetrievalError
from rag.retrieval.retriever import (
    RetrievalResult,
    RetrievedChunk,
    Retriever,
)
from rag.vectorstore.faiss_store import SearchResult


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Minimal embedder that returns a fixed vector for any query."""

    def __init__(self) -> None:
        self.calls = 0

    def query_encode(self, query: str) -> np.ndarray:
        self.calls += 1
        return np.ones((1, 384), dtype=np.float32)


class StubStore:
    """Minimal store holding a configurable list of SearchResult."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []

    def search(self, query_vector, top_k=5):
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_result(score: float, source: str = "doc.pdf", page: int = 1) -> SearchResult:
    return SearchResult(
        chunk_id="{}_p{}_c0".format(source, page),
        score=score,
        metadata={
            "text": "Sample text from {} page {}.".format(source, page),
            "source_file": source,
            "page": page,
            "chunk_index": 0,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass."""

    def test_fields_populated(self):
        chunk = RetrievedChunk(
            text="hello",
            source_file="doc.pdf",
            page=3,
            score=0.85,
            chunk_id="doc.pdf_p3_c0",
            metadata={"source_file": "doc.pdf"},
        )
        assert chunk.text == "hello"
        assert chunk.source_file == "doc.pdf"
        assert chunk.page == 3
        assert chunk.score == 0.85
        assert chunk.chunk_id == "doc.pdf_p3_c0"

    def test_frozen(self):
        chunk = RetrievedChunk(
            text="hello", source_file="d", page=1, score=1.0,
            chunk_id="c", metadata={},
        )
        with pytest.raises(AttributeError):
            chunk.text = "changed"

    def test_default_metadata(self):
        chunk = RetrievedChunk(
            text="x", source_file="d", page=1, score=1.0, chunk_id="c",
        )
        assert chunk.metadata == {}


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_fields(self):
        result = RetrievalResult(
            query="test",
            chunks=[],
            total_found=5,
            total_returned=0,
            latency_ms=12.3,
        )
        assert result.query == "test"
        assert result.chunks == []
        assert result.total_found == 5
        assert result.total_returned == 0
        assert result.latency_ms == 12.3

    def test_frozen(self):
        result = RetrievalResult(
            query="t", chunks=[], total_found=0, total_returned=0,
            latency_ms=0.0,
        )
        with pytest.raises(AttributeError):
            result.query = "x"


class TestRetrieve:
    """Tests for Retriever.retrieve()."""

    def test_retrieves_above_threshold(self):
        results = [
            _make_result(0.9),
            _make_result(0.8),
            _make_result(0.5),
        ]
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query", top_k=3)

        assert out.total_found == 3
        assert out.total_returned == 3
        assert len(out.chunks) == 3
        # Scores should be sorted descending.
        assert out.chunks[0].score >= out.chunks[1].score >= out.chunks[2].score

    def test_filters_below_threshold(self):
        """Results below SCORE_THRESHOLD (0.3) are filtered out."""
        results = [
            _make_result(0.9),
            _make_result(0.2),  # below threshold
            _make_result(0.1),  # below threshold
        ]
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        assert out.total_found == 3
        assert out.total_returned == 1  # only 0.9 passes
        assert out.chunks[0].score == 0.9

    def test_empty_query_raises(self):
        retriever = Retriever(StubEmbedder(), StubStore())
        with pytest.raises(RetrievalError, match="empty"):
            retriever.retrieve("")

    def test_whitespace_query_raises(self):
        retriever = Retriever(StubEmbedder(), StubStore())
        with pytest.raises(RetrievalError, match="empty"):
            retriever.retrieve("   ")

    def test_no_results_returns_empty(self):
        store = StubStore([])
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        assert out.total_found == 0
        assert out.total_returned == 0
        assert out.chunks == []

    def test_custom_top_k(self):
        results = [_make_result(0.5 + i * 0.1) for i in range(10)]
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query", top_k=3)

        assert out.total_found <= 3  # StubStore slices to top_k

    def test_metadata_propagated(self):
        results = [
            _make_result(0.9, source="report.pdf", page=42),
        ]
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        assert out.chunks[0].source_file == "report.pdf"
        assert out.chunks[0].page == 42
        assert out.chunks[0].chunk_id == "report.pdf_p42_c0"
        assert "report.pdf" in out.chunks[0].text

    def test_latency_recorded(self):
        store = StubStore([_make_result(0.9)])
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        assert out.latency_ms > 0
        assert out.latency_ms < 1000  # should be fast with stubs

    def test_scores_rounded(self):
        """Scores are rounded to 4 decimal places."""
        results = [_make_result(0.87654321)]
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        # 0.87654321 should be rounded to 0.8765.
        assert out.chunks[0].score == 0.8765

    def test_exact_threshold_included(self):
        """Score equal to threshold is included (>= comparison)."""
        results = [_make_result(0.3)]  # exactly at threshold
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        assert out.total_returned == 1

    def test_chunks_contain_text(self):
        results = [_make_result(0.9, source="test.pdf", page=5)]
        store = StubStore(results)
        retriever = Retriever(StubEmbedder(), store)

        out = retriever.retrieve("query")

        assert out.chunks[0].text == "Sample text from test.pdf page 5."
