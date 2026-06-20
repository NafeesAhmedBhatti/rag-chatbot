"""
Unit tests for rag.generation.prompt.

Tests the prompt builder without needing an LLM or network access.
"""

from __future__ import annotations

import pytest

from rag.generation.prompt import build_chat_messages, build_context_block
from rag.retrieval.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str = "Sample passage text.",
    source_file: str = "doc.pdf",
    page: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_file=source_file,
        page=page,
        score=0.9,
        chunk_id="doc.pdf_p1_c0",
        metadata={},
    )


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock:
    """Tests for build_context_block()."""

    def test_empty_chunks_returns_empty_string(self):
        assert build_context_block([]) == ""

    def test_single_chunk(self):
        chunk = _make_chunk("Hello world.", "a.pdf", 3)
        block = build_context_block([chunk])
        assert "[1]" in block
        assert "a.pdf" in block
        assert "Page 3" in block
        assert "Hello world." in block

    def test_multiple_chunks_separated(self):
        chunks = [
            _make_chunk("First passage.", "a.pdf", 1),
            _make_chunk("Second passage.", "b.pdf", 2),
        ]
        block = build_context_block(chunks)
        assert "[1]" in block
        assert "[2]" in block
        assert "---" in block  # separator
        assert "First passage." in block
        assert "Second passage." in block

    def test_numbering_is_sequential(self):
        chunks = [_make_chunk("text", "d.pdf", i) for i in range(1, 6)]
        block = build_context_block(chunks)
        for i in range(1, 6):
            assert "[{}]".format(i) in block

    def test_truncation_at_max_chars(self):
        """Context block truncates when total chars exceed the limit."""
        # Each chunk is ~1000 chars.
        big_text = "A" * 1000
        chunks = [_make_chunk(big_text, "d.pdf", 1) for _ in range(20)]
        block = build_context_block(chunks)
        # Should be truncated — far fewer than 20*1000 = 20,000 chars.
        assert len(block) <= 15_000  # rough bound

    def test_chunk_text_preserved(self):
        chunk = _make_chunk("The Eiffel Tower is in Paris.", "report.pdf", 5)
        block = build_context_block([chunk])
        assert "The Eiffel Tower is in Paris." in block


# ---------------------------------------------------------------------------
# build_chat_messages
# ---------------------------------------------------------------------------

class TestBuildChatMessages:
    """Tests for build_chat_messages()."""

    def test_returns_two_messages(self):
        messages = build_chat_messages("What is AI?", [_make_chunk()])
        assert len(messages) == 2

    def test_first_message_is_system(self):
        messages = build_chat_messages("query", [_make_chunk()])
        assert messages[0]["role"] == "system"

    def test_second_message_is_user(self):
        messages = build_chat_messages("query", [_make_chunk()])
        assert messages[1]["role"] == "user"

    def test_system_message_contains_rules(self):
        messages = build_chat_messages("query", [_make_chunk()])
        system_content = messages[0]["content"]
        assert "ONLY" in system_content or "only" in system_content
        assert "citation" in system_content.lower()
        assert "don't know" in system_content.lower()

    def test_user_message_contains_query(self):
        messages = build_chat_messages("What is deep learning?", [_make_chunk()])
        assert "What is deep learning?" in messages[1]["content"]

    def test_user_message_contains_context(self):
        chunk = _make_chunk("Deep learning is a subset of ML.", "ai.pdf", 7)
        messages = build_chat_messages("What is deep learning?", [chunk])
        assert "Deep learning is a subset of ML." in messages[1]["content"]

    def test_user_message_contains_citation_instruction(self):
        messages = build_chat_messages("query", [_make_chunk()])
        assert "[filename, p.X]" in messages[1]["content"]

    def test_no_chunks_fallback_message(self):
        """When no chunks are provided, the prompt handles it gracefully."""
        messages = build_chat_messages("test query", [])
        assert "No relevant context" in messages[1]["content"]
        assert "I don't know" in messages[1]["content"]

    def test_source_and_page_in_context(self):
        chunk = _make_chunk("text", "manual.pdf", 12)
        messages = build_chat_messages("query", [chunk])
        assert "manual.pdf" in messages[1]["content"]
        assert "Page 12" in messages[1]["content"]
