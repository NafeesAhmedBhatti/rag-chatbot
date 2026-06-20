"""
Tests for rag.processing.chunker.Chunker.

Covers:
    - Token limit enforcement (chunks ≤ 512 tokens + overlap variance)
    - Overlap between consecutive chunks
    - Metadata integrity (chunk_id, source_file, page, language, offsets)
    - Paragraph / sentence boundary respect
    - Empty and short text handling
    - Multi-page chunking (page numbers, chunk indices reset per page)
    - Configurable chunk_size / overlap
"""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.models import Chunk
from rag.processing.chunker import Chunker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker() -> Chunker:
    """Default Chunker with chunk_size=512, overlap=64."""
    return Chunker()


def _generate_long_text(repeats: int = 200) -> str:
    """Generate text that produces multiple chunks (~2000 tokens)."""
    sentence = "The quick brown fox jumps over the lazy dog near the riverbank."
    return " ".join([sentence] * repeats)


# ---------------------------------------------------------------------------
# Basic chunking
# ---------------------------------------------------------------------------

class TestBasicChunking:
    """Basic chunking behavior."""

    def test_short_text_single_chunk(self, chunker: Chunker) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        assert len(chunks) == 1
        assert chunks[0].token_count <= 512

    def test_long_text_multiple_chunks(self, chunker: Chunker) -> None:
        text = _generate_long_text(200)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        assert len(chunks) > 1

    def test_empty_text_no_chunks(self, chunker: Chunker) -> None:
        assert chunker.chunk("", "test.pdf", 1) == []

    def test_whitespace_only_no_chunks(self, chunker: Chunker) -> None:
        assert chunker.chunk("   \n\n   ", "test.pdf", 1) == []


# ---------------------------------------------------------------------------
# Token limit enforcement
# ---------------------------------------------------------------------------

class TestTokenLimits:
    """Chunks should respect the configured token limit."""

    def test_all_chunks_within_limit(self, chunker: Chunker) -> None:
        text = _generate_long_text(300)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        for chunk in chunks:
            # Allow some variance for overlap tokens.
            assert chunk.token_count <= 512 + 64, (
                f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens"
            )

    def test_chunks_near_target_size(self, chunker: Chunker) -> None:
        """Non-final chunks should be close to the target (not tiny)."""
        text = _generate_long_text(300)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        # All chunks except possibly the last should be substantial.
        for chunk in chunks[:-1]:
            assert chunk.token_count >= 400, (
                f"Chunk {chunk.chunk_id} has only {chunk.token_count} tokens (expected ~512)"
            )

    def test_custom_chunk_size(self) -> None:
        config = Settings(chunk_size=64, chunk_overlap=8)
        chunker = Chunker(config=config)
        text = _generate_long_text(50)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        for chunk in chunks:
            assert chunk.token_count <= 64 + 8  # limit + overlap variance
        assert len(chunks) > 5  # small chunks → many chunks


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    """Overlap between consecutive chunks."""

    def test_overlap_exists_between_consecutive_chunks(self, chunker: Chunker) -> None:
        text = _generate_long_text(200)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        # For each pair of consecutive chunks, the start of chunk N+1
        # should contain some text from the end of chunk N.
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i].text[-40:].lower()
            start_of_next = chunks[i + 1].text[:40].lower()
            # At least one word from the end of the current chunk should
            # appear at the start of the next chunk (overlap).
            words = end_of_current.split()
            if words:
                overlap_found = any(
                    word in start_of_next for word in words[-3:]
                )
                assert overlap_found, (
                    f"No overlap between chunk {i} and {i + 1}: "
                    f"end='{words[-3:]}', start='{start_of_next}'"
                )

    def test_overlap_configurable(self) -> None:
        config = Settings(chunk_size=128, chunk_overlap=0)
        chunker = Chunker(config=config)
        text = _generate_long_text(100)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        # With 0 overlap, consecutive chunks should NOT share text.
        for i in range(len(chunks) - 1):
            end_words = chunks[i].text[-20:].split()
            start_words = chunks[i + 1].text[:20].split()
            # Check no significant overlap.
            shared = set(end_words) & set(start_words)
            # Some incidental overlap is OK (common words like "the").
            # But there should be no structural overlap.
            assert len(shared) <= 2, f"Unexpected large overlap: {shared}"


# ---------------------------------------------------------------------------
# Metadata integrity
# ---------------------------------------------------------------------------

class TestMetadata:
    """Metadata fields on Chunk objects."""

    def test_chunk_id_format(self, chunker: Chunker) -> None:
        chunks = chunker.chunk("Hello world.", "report.pdf", 3, "en")
        assert chunks[0].chunk_id == "report_p3_c0"

    def test_source_file_preserved(self, chunker: Chunker) -> None:
        chunks = chunker.chunk("Hello world.", "annual_report.pdf", 1, "en")
        assert chunks[0].source_file == "annual_report.pdf"

    def test_page_number(self, chunker: Chunker) -> None:
        chunks = chunker.chunk("Hello world.", "test.pdf", 42, "en")
        assert chunks[0].page == 42

    def test_language_stored(self, chunker: Chunker) -> None:
        chunks = chunker.chunk("Hello world.", "test.pdf", 1, "fr")
        assert chunks[0].language == "fr"

    def test_char_offsets(self, chunker: Chunker) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        assert chunks[0].char_end > chunks[0].char_start

    def test_token_count_auto_computed(self, chunker: Chunker) -> None:
        chunks = chunker.chunk("Hello world.", "test.pdf", 1, "en")
        assert chunks[0].token_count > 0

    def test_chunk_indices_sequential(self, chunker: Chunker) -> None:
        text = _generate_long_text(200)
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# Boundary respect
# ---------------------------------------------------------------------------

class TestBoundaries:
    """Chunking should respect natural text boundaries."""

    def test_paragraph_boundaries_respected(self, chunker: Chunker) -> None:
        """Short paragraphs should not be split across chunks."""
        # Create two distinct paragraphs, each well under chunk_size.
        para1 = "First paragraph about apples. " * 10
        para2 = "Second paragraph about oranges. " * 10
        text = para1 + "\n\n" + para2
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        # The combined text is small enough for a single chunk.
        assert len(chunks) == 1

    def test_sentence_not_split(self, chunker: Chunker) -> None:
        """A single short sentence should be in one chunk."""
        text = "This is a single sentence that fits in one chunk."
        chunks = chunker.chunk(text, "test.pdf", 1, "en")
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text.strip()


# ---------------------------------------------------------------------------
# Multi-page scenarios
# ---------------------------------------------------------------------------

class TestMultiPage:
    """Chunking across multiple pages of the same document."""

    def test_chunk_indices_reset_per_page(self, chunker: Chunker) -> None:
        """Each page's chunks should start at index 0."""
        text = _generate_long_text(100)
        page1_chunks = chunker.chunk(text, "doc.pdf", 1, "en")
        page2_chunks = chunker.chunk(text, "doc.pdf", 2, "en")
        assert page1_chunks[0].chunk_index == 0
        assert page2_chunks[0].chunk_index == 0

    def test_page_numbers_correct(self, chunker: Chunker) -> None:
        text = "Some text for testing."
        chunks_p1 = chunker.chunk(text, "doc.pdf", 1)
        chunks_p2 = chunker.chunk(text, "doc.pdf", 2)
        chunks_p3 = chunker.chunk(text, "doc.pdf", 3)
        assert chunks_p1[0].page == 1
        assert chunks_p2[0].page == 2
        assert chunks_p3[0].page == 3

    def test_chunk_ids_unique_across_pages(self, chunker: Chunker) -> None:
        text = "Some text for testing."
        c1 = chunker.chunk(text, "doc.pdf", 1)
        c2 = chunker.chunk(text, "doc.pdf", 2)
        assert c1[0].chunk_id != c2[0].chunk_id


# ---------------------------------------------------------------------------
# Integration: cleaner + chunker
# ---------------------------------------------------------------------------

class TestCleanerChunkerIntegration:
    """Text cleaned then chunked should work end-to-end."""

    def test_cleaned_text_chunks_correctly(self, chunker: Chunker) -> None:
        from rag.processing.cleaner import TextCleaner

        cleaner = TextCleaner()
        raw = "Hello    world\n\nThis is   a test."
        cleaned = cleaner.clean(raw)
        chunks = chunker.chunk(cleaned, "test.pdf", 1, "en")
        assert len(chunks) >= 1
        # Cleaned text should not have excessive whitespace in chunks.
        for chunk in chunks:
            assert "    " not in chunk.text  # no 4+ spaces
