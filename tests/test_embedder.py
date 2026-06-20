"""
Tests for rag.embeddings.embedder.Embedder.

Covers:
    - Model loading and dimension
    - Batch encoding output shape and dtype
    - L2 normalization of document embeddings
    - L2 normalization of query embeddings
    - Semantic similarity (related texts score higher)
    - BGE query prefix application
    - Empty input handling
    - Error handling (empty query)
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings.embedder import Embedder
from rag.exceptions import EmbeddingError


# Module-level embedder — loaded once for all tests in this module
# (model load takes ~2s, so we amortize across tests).
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


# ---------------------------------------------------------------------------
# Model loading and dimensions
# ---------------------------------------------------------------------------

class TestModelLoading:
    """Model loads and reports correct dimension."""

    def test_dimension_is_384(self) -> None:
        emb = get_embedder()
        assert emb.dimension == 384

    def test_model_loads_without_error(self) -> None:
        emb = get_embedder()
        # Encoding triggers lazy load.
        vectors = emb.encode(["test"])
        assert vectors is not None


# ---------------------------------------------------------------------------
# Document encoding
# ---------------------------------------------------------------------------

class TestEncode:
    """Batch encoding of document texts."""

    def test_output_shape(self) -> None:
        emb = get_embedder()
        texts = ["hello world", "machine learning", "data science"]
        vectors = emb.encode(texts)
        assert vectors.shape == (3, 384)

    def test_output_dtype(self) -> None:
        emb = get_embedder()
        vectors = emb.encode(["test text"])
        assert vectors.dtype == np.float32

    def test_l2_normalized(self) -> None:
        emb = get_embedder()
        texts = ["The quick brown fox.", "A separate sentence entirely."]
        vectors = emb.encode(texts)
        for i, v in enumerate(vectors):
            norm = float(np.linalg.norm(v))
            assert abs(norm - 1.0) < 0.01, (
                f"Vector {i} not L2-normalized: norm={norm}"
            )

    def test_single_text(self) -> None:
        emb = get_embedder()
        vectors = emb.encode(["one text"])
        assert vectors.shape == (1, 384)

    def test_empty_list_returns_empty_array(self) -> None:
        emb = get_embedder()
        vectors = emb.encode([])
        assert vectors.shape == (0, 384)

    def test_batch_consistency(self) -> None:
        """Encoding texts individually vs in a batch should give the same result."""
        emb = get_embedder()
        texts = ["first text", "second text", "third text"]
        batch_result = emb.encode(texts)
        individual_results = np.stack([emb.encode([t])[0] for t in texts])
        np.testing.assert_allclose(
            batch_result, individual_results, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

class TestQueryEncode:
    """Query encoding with BGE prefix."""

    def test_output_shape(self) -> None:
        emb = get_embedder()
        qvec = emb.query_encode("What is machine learning?")
        assert qvec.shape == (1, 384)

    def test_output_dtype(self) -> None:
        emb = get_embedder()
        qvec = emb.query_encode("test query")
        assert qvec.dtype == np.float32

    def test_l2_normalized(self) -> None:
        emb = get_embedder()
        qvec = emb.query_encode("normalized query")
        norm = float(np.linalg.norm(qvec[0]))
        assert abs(norm - 1.0) < 0.01

    def test_empty_query_raises_error(self) -> None:
        emb = get_embedder()
        with pytest.raises(EmbeddingError, match="empty query"):
            emb.query_encode("")

    def test_whitespace_query_raises_error(self) -> None:
        emb = get_embedder()
        with pytest.raises(EmbeddingError, match="empty query"):
            emb.query_encode("   ")


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    """Related texts should have higher cosine similarity than unrelated ones."""

    def test_related_text_scores_higher(self) -> None:
        emb = get_embedder()
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Climate change affects global agriculture significantly.",
        ]
        vectors = emb.encode(texts)
        query_vec = emb.query_encode("What is machine learning?")

        similarities = vectors @ query_vec[0]
        # The ML text (index 1) should score highest for an ML query.
        assert similarities[1] > similarities[0]
        assert similarities[1] > similarities[2]

    def test_identical_text_high_similarity(self) -> None:
        emb = get_embedder()
        text = "Artificial intelligence transforms document processing."
        doc_vec = emb.encode([text])[0]
        query_vec = emb.query_encode(text)[0]
        sim = float(doc_vec @ query_vec)
        assert sim > 0.8  # identical content should be highly similar

    def test_unrelated_text_low_similarity(self) -> None:
        emb = get_embedder()
        doc_vec = emb.encode(["Quantum physics involves subatomic particles."])[0]
        query_vec = emb.query_encode("How to bake chocolate chip cookies?")[0]
        sim = float(doc_vec @ query_vec)
        assert sim < 0.5  # unrelated content should have low similarity
