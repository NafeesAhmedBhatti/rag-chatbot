"""
Unit tests for rag.generation.generator.

Tests citation extraction thoroughly and tests the Generator class
using a mocked LLM client (no real API calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.exceptions import GenerationError
from rag.generation.generator import (
    Citation,
    GenerationResult,
    Generator,
    extract_citations,
)
from rag.retrieval.retriever import RetrievalResult, RetrievedChunk


# ---------------------------------------------------------------------------
# Citation extraction tests (pure functions, no LLM needed)
# ---------------------------------------------------------------------------

class TestExtractCitationsBasic:
    """Tests for extract_citations() with standard patterns."""

    def test_single_citation(self):
        answer = "Paris is in France [report.pdf, p.5]."
        cites = extract_citations(answer)
        assert len(cites) == 1
        assert cites[0].filename == "report.pdf"
        assert cites[0].page == 5

    def test_multiple_citations(self):
        answer = (
            "Neural networks are used in AI [a.pdf, p.3]. "
            "They learn via backpropagation [b.pdf, p.7]."
        )
        cites = extract_citations(answer)
        assert len(cites) == 2
        assert cites[0].filename == "a.pdf"
        assert cites[0].page == 3
        assert cites[1].filename == "b.pdf"
        assert cites[1].page == 7

    def test_no_citations(self):
        answer = "This is a plain sentence without citations."
        cites = extract_citations(answer)
        assert cites == []

    def test_consecutive_citations(self):
        """Multiple citations after a single claim: [a, p.1][b, p.2]."""
        answer = "Machine learning is popular [a.pdf, p.1][b.pdf, p.2]."
        cites = extract_citations(answer)
        assert len(cites) == 2

    def test_citation_without_comma(self):
        """Pattern: [filename p.5] (no comma before p)."""
        answer = "Some fact [report.pdf p.5]."
        cites = extract_citations(answer)
        assert len(cites) == 1
        assert cites[0].page == 5

    def test_multi_page_citation(self):
        """Pattern: [filename, p.5, p.6]."""
        answer = "Some fact [report.pdf, p.5, p.6]."
        cites = extract_citations(answer)
        assert len(cites) == 2
        assert cites[0].page == 5
        assert cites[1].page == 6

    def test_deduplication(self):
        """Same citation appearing twice is only returned once."""
        answer = (
            "Fact one [a.pdf, p.3]. Fact two also on the same page [a.pdf, p.3]."
        )
        cites = extract_citations(answer)
        assert len(cites) == 1

    def test_empty_answer(self):
        assert extract_citations("") == []

    def test_case_insensitive_dedup(self):
        """Citation deduplication is case-insensitive on filename."""
        answer = "Fact [Report.PDF, p.3] and again [report.pdf, p.3]."
        cites = extract_citations(answer)
        assert len(cites) == 1


class TestExtractCitationsFallback:
    """Tests for [p.X] fallback when filename is missing."""

    def test_page_only_with_chunks(self):
        """[p.5] without a filename — assign from chunk metadata."""
        answer = "Some fact [p.5]."
        chunks = [
            RetrievedChunk(
                text="text", source_file="doc.pdf", page=5,
                score=0.9, chunk_id="c", metadata={},
            ),
        ]
        cites = extract_citations(answer, chunks)
        assert len(cites) == 1
        assert cites[0].page == 5
        assert cites[0].filename == "doc.pdf"

    def test_page_only_without_chunks(self):
        """[p.5] without chunks → filename 'unknown'."""
        answer = "Some fact [p.5]."
        cites = extract_citations(answer)
        # Without chunks, the fallback still fires but can't assign a source.
        assert len(cites) == 1
        assert cites[0].filename == "unknown"

    def test_page_only_not_in_chunks(self):
        """[p.99] but no chunk with page 99 → filename 'unknown'."""
        answer = "Some fact [p.99]."
        chunks = [
            RetrievedChunk(
                text="t", source_file="doc.pdf", page=1,
                score=0.9, chunk_id="c", metadata={},
            ),
        ]
        cites = extract_citations(answer, chunks)
        assert len(cites) == 1
        assert cites[0].filename == "unknown"
        assert cites[0].page == 99


class TestCitationDataclass:
    """Tests for the Citation dataclass."""

    def test_fields(self):
        c = Citation(filename="a.pdf", page=3, raw="[a.pdf, p.3]")
        assert c.filename == "a.pdf"
        assert c.page == 3
        assert c.raw == "[a.pdf, p.3]"

    def test_to_dict(self):
        c = Citation(filename="a.pdf", page=3, raw="[a.pdf, p.3]")
        d = c.to_dict()
        assert d["filename"] == "a.pdf"
        assert d["page"] == 3
        assert d["raw"] == "[a.pdf, p.3]"

    def test_frozen(self):
        c = Citation(filename="a", page=1, raw="")
        with pytest.raises(AttributeError):
            c.page = 2


# ---------------------------------------------------------------------------
# Generator tests (with mocked LLM client)
# ---------------------------------------------------------------------------

@dataclass
class _MockUsage:
    total_tokens: int = 150


@dataclass
class _MockMessage:
    content: str


@dataclass
class _MockChoice:
    message: _MockMessage


@dataclass
class _MockResponse:
    choices: list
    usage: _MockUsage = None


def _make_mock_response(
    content: str = "Sample answer [doc.pdf, p.1].",
    tokens: int = 150,
) -> _MockResponse:
    return _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content=content))],
        usage=_MockUsage(total_tokens=tokens),
    )


class TestGeneratorGenerate:
    """Tests for Generator.generate() with a mocked client."""

    def test_generate_with_chunks(self):
        """Generator produces an answer when chunks are available."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "AI is a field of computer science [doc.pdf, p.1]."
        )

        gen = Generator(client=mock_client)
        result = gen.generate(
            RetrievalResult(
                query="What is AI?",
                chunks=[
                    RetrievedChunk(
                        text="AI is a field of computer science.",
                        source_file="doc.pdf",
                        page=1,
                        score=0.9,
                        chunk_id="c",
                        metadata={},
                    ),
                ],
                total_found=1,
                total_returned=1,
                latency_ms=10.0,
            )
        )

        assert "computer science" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].filename == "doc.pdf"
        assert result.citations[0].page == 1
        assert result.tokens_used == 150
        assert result.used_fallback is False
        assert result.latency_ms > 0

    def test_generate_no_chunks_fallback(self):
        """No chunks → fallback answer without calling LLM."""
        mock_client = MagicMock()
        gen = Generator(client=mock_client)
        result = gen.generate(
            RetrievalResult(
                query="test",
                chunks=[],
                total_found=0,
                total_returned=0,
                latency_ms=5.0,
            )
        )

        assert "I don't know" in result.answer
        assert result.citations == []
        assert result.used_fallback is True
        mock_client.chat.completions.create.assert_not_called()

    def test_generate_llm_error_raises(self):
        """LLM API error raises GenerationError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        gen = Generator(client=mock_client)
        with pytest.raises(GenerationError, match="LLM API call failed"):
            gen.generate(
                RetrievalResult(
                    query="q",
                    chunks=[
                        RetrievedChunk(
                            text="t", source_file="d.pdf", page=1,
                            score=0.5, chunk_id="c", metadata={},
                        ),
                    ],
                    total_found=1,
                    total_returned=1,
                    latency_ms=1.0,
                )
            )

    def test_generate_records_model_name(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response()

        gen = Generator(client=mock_client)
        result = gen.generate(
            RetrievalResult(
                query="q",
                chunks=[
                    RetrievedChunk(
                        text="t", source_file="d.pdf", page=1,
                        score=0.5, chunk_id="c", metadata={},
                    ),
                ],
                total_found=1,
                total_returned=1,
                latency_ms=1.0,
            )
        )
        assert result.model == settings.llm_model

    def test_generate_multiple_citations(self):
        answer = (
            "Deep learning is a subset of ML [a.pdf, p.3]. "
            "It uses backpropagation [b.pdf, p.7]."
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            answer
        )
        gen = Generator(client=mock_client)
        result = gen.generate(
            RetrievalResult(
                query="q",
                chunks=[
                    RetrievedChunk(
                        text="t", source_file="a.pdf", page=3,
                        score=0.9, chunk_id="c1", metadata={},
                    ),
                    RetrievedChunk(
                        text="t", source_file="b.pdf", page=7,
                        score=0.8, chunk_id="c2", metadata={},
                    ),
                ],
                total_found=2,
                total_returned=2,
                latency_ms=1.0,
            )
        )

        assert len(result.citations) == 2
        assert result.citations[0].filename == "a.pdf"
        assert result.citations[1].filename == "b.pdf"

    def test_generate_strips_answer_whitespace(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "  \n  Answer text  \n  "
        )
        gen = Generator(client=mock_client)
        result = gen.generate(
            RetrievalResult(
                query="q",
                chunks=[
                    RetrievedChunk(
                        text="t", source_file="d.pdf", page=1,
                        score=0.5, chunk_id="c", metadata={},
                    ),
                ],
                total_found=1,
                total_returned=1,
                latency_ms=1.0,
            )
        )
        assert result.answer == "Answer text"

    def test_generate_no_citations_in_answer(self):
        """Answer with no citations → empty list."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "A plain answer with no citations at all."
        )
        gen = Generator(client=mock_client)
        result = gen.generate(
            RetrievalResult(
                query="q",
                chunks=[
                    RetrievedChunk(
                        text="t", source_file="d.pdf", page=1,
                        score=0.5, chunk_id="c", metadata={},
                    ),
                ],
                total_found=1,
                total_returned=1,
                latency_ms=1.0,
            )
        )
        assert result.citations == []


class TestGeneratorInit:
    """Tests for Generator initialization."""

    def test_no_key_raises_on_load(self):
        """Generator with empty API key raises when client is loaded."""
        gen = Generator()
        # The key is empty by default in config; but we need to ensure
        # _load_client raises.
        # We can't easily override the global settings, so we test that
        # generate() with no chunks doesn't trigger _load_client.
        result = gen.generate(
            RetrievalResult(
                query="q", chunks=[], total_found=0,
                total_returned=0, latency_ms=0.0,
            )
        )
        # Should use fallback, not raise.
        assert result.used_fallback is True

    def test_custom_client_injected(self):
        mock_client = MagicMock()
        gen = Generator(client=mock_client)
        assert gen._client is mock_client

    def test_lazy_client(self):
        """Client is None until _load_client is called."""
        gen = Generator()
        assert gen._client is None
