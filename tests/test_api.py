"""
API integration tests for the RAG Chatbot.

Tests cover:
  - GET /api/health (Phase 1)
  - GET /api/stats (Phase 1, updated for Phase 6)
  - POST /api/chat (Phase 6)
  - POST /api/ingest (Phase 6)
  - GET / (root UI)

Chat tests use mocked Retriever/Generator to avoid slow LLM calls.
Ingest tests use a small generated PDF.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_includes_app_env(self):
        data = client.get("/api/health").json()
        assert "app_env" in data

    def test_includes_version(self):
        data = client.get("/api/health").json()
        assert data["version"] == "0.1.0"

    def test_includes_uptime(self):
        data = client.get("/api/health").json()
        assert data["uptime_seconds"] >= 0

    def test_includes_timestamp(self):
        data = client.get("/api/health").json()
        assert "T" in data["timestamp"]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_returns_200(self):
        resp = client.get("/api/stats")
        assert resp.status_code == 200

    def test_has_required_fields(self):
        data = client.get("/api/stats").json()
        for key in [
            "total_documents", "total_chunks", "index_size_mb",
            "embedding_model", "vector_dimensions", "llm_model",
        ]:
            assert key in data

    def test_embedding_model_is_bge(self):
        data = client.get("/api/stats").json()
        assert "bge" in data["embedding_model"].lower()

    def test_dimensions_is_384(self):
        data = client.get("/api/stats").json()
        assert data["vector_dimensions"] == 384

    def test_llm_model_present(self):
        data = client.get("/api/stats").json()
        assert data["llm_model"]


# ---------------------------------------------------------------------------
# Chat (with mocked pipeline)
# ---------------------------------------------------------------------------

class TestChat:
    """Tests for POST /api/chat using mocked retriever/generator."""

    def test_empty_question_returns_422(self):
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_whitespace_question_returns_400(self):
        resp = client.post("/api/chat", json={"question": "   "})
        assert resp.status_code == 400

    def test_missing_question_returns_422(self):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_chat_returns_answer(self):
        """Chat endpoint returns an answer with mocked pipeline."""
        # Mock the lazy pipeline initialization.
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384

        mock_store = MagicMock()
        mock_store.size = 1
        mock_store.get_stats.return_value = {
            "total_documents": 1, "total_chunks": 1,
            "index_size_mb": 0.01, "dimension": 384,
        }

        mock_retriever = MagicMock()
        from rag.retrieval.retriever import RetrievalResult, RetrievedChunk
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="test",
            chunks=[
                RetrievedChunk(
                    text="Deep learning is a subset of machine learning.",
                    source_file="ai.pdf",
                    page=3,
                    score=0.85,
                    chunk_id="ai.pdf_p3_c0",
                    metadata={"token_count": 50},
                ),
            ],
            total_found=1,
            total_returned=1,
            latency_ms=15.0,
        )

        from rag.generation.generator import GenerationResult, Citation
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Deep learning is a subset of machine learning [ai.pdf, p.3].",
            citations=[Citation(filename="ai.pdf", page=3, raw="[ai.pdf, p.3]")],
            model="test-model",
            latency_ms=500.0,
            tokens_used=100,
        )

        with patch("app.api._get_pipeline", return_value=(
            mock_embedder, mock_store, mock_retriever, mock_generator,
        )):
            resp = client.post("/api/chat", json={"question": "What is deep learning?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "deep learning" in data["answer"].lower()
        assert len(data["citations"]) == 1
        assert data["citations"][0]["filename"] == "ai.pdf"
        assert data["citations"][0]["page"] == 3
        assert len(data["retrieved_chunks"]) == 1
        assert data["retrieved_chunks"][0]["score"] == 0.85
        assert data["model"] == "test-model"
        assert data["latency_ms"] > 0

    def test_chat_returns_latency_breakdown(self):
        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_store.size = 1

        from rag.retrieval.retriever import RetrievalResult, RetrievedChunk
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="q", chunks=[
                RetrievedChunk(
                    text="t", source_file="d.pdf", page=1,
                    score=0.9, chunk_id="c", metadata={},
                ),
            ],
            total_found=1, total_returned=1, latency_ms=20.0,
        )

        from rag.generation.generator import GenerationResult
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Answer.", citations=[], model="m",
            latency_ms=800.0, tokens_used=50,
        )

        with patch("app.api._get_pipeline", return_value=(
            mock_embedder, mock_store, mock_retriever, mock_generator,
        )):
            resp = client.post("/api/chat", json={"question": "test"})

        data = resp.json()
        assert "retrieval_latency_ms" in data
        assert "generation_latency_ms" in data
        assert data["retrieval_latency_ms"] == 20.0
        assert data["generation_latency_ms"] == 800.0


# ---------------------------------------------------------------------------
# Root (UI)
# ---------------------------------------------------------------------------

class TestRoot:
    def test_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_contains_chat_ui(self):
        resp = client.get("/")
        assert b"chat-input" in resp.content
        assert b"chat-messages" in resp.content

    def test_contains_sidebar(self):
        resp = client.get("/")
        assert b"sidebar-left" in resp.content
        assert b"sidebar-right" in resp.content
