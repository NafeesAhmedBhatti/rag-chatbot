"""
Shared data models for the RAG pipeline.

These dataclasses flow between pipeline stages (ingestion → processing
→ embeddings → retrieval → generation). Defining them centrally
prevents circular imports between modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExtractionMethod(str, Enum):
    """How text was extracted from a PDF page."""

    NATIVE = "native"  # PyMuPDF get_text()
    OCR = "ocr"        # Tesseract or RapidOCR on a rendered image


@dataclass(frozen=True)
class PageContent:
    """Text extracted from a single PDF page.

    Attributes
    ----------
    page:
        1-based page number (human-readable).
    text:
        Extracted text (may be empty if the page was blank or had no
        recognisable text).
    extraction_method:
        How the text was obtained (native vs OCR).
    char_count:
        Length of ``text`` in characters. Redundant with ``len(text)``
        but stored explicitly for fast filtering without re-measuring.
    """

    page: int
    text: str
    extraction_method: ExtractionMethod
    char_count: int = field(default=0)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "char_count", len(self.text))


@dataclass(frozen=True)
class Chunk:
    """A semantically-bounded text chunk with metadata for retrieval.

    Produced by ``rag.processing.chunker.Chunker`` from a page of text.
    Each chunk is the unit that gets embedded, stored in the vector DB,
    and returned as a citation source.

    Attributes
    ----------
    chunk_id:
        Unique identifier: ``{source_file}_p{page}_c{chunk_index}``.
    text:
        The chunk's text content (already cleaned/normalized).
    source_file:
        Original PDF filename (without directory path).
    page:
        1-based page number in the source PDF.
    char_start:
        Character offset where this chunk starts within the page text.
    char_end:
        Character offset where this chunk ends (exclusive).
    chunk_index:
        Index of this chunk within the page (0-based).
    language:
        ISO 639-1 language code (e.g. "en") detected for the page.
    token_count:
        Number of BGE tokens in ``text`` (for sizing/verification).
    """

    chunk_id: str
    text: str
    source_file: str
    page: int
    char_start: int
    char_end: int
    chunk_index: int
    language: str
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_count", self._count_tokens())

    def _count_tokens(self) -> int:
        """Count BGE tokens. Uses a cached tokenizer on first call."""
        return _BGE_TOKENIZER.count(self.text)


@dataclass(frozen=True)
class LanguageResult:
    """Result of language detection for a text.

    Attributes
    ----------
    code:
        ISO 639-1 language code (e.g. "en", "fr").
    reliable:
        Whether the detection is confident (langdetect's ``is_reliable``).
    """

    code: str
    reliable: bool


# ---------------------------------------------------------------------------
# Token counter — lazily-loaded BGE tokenizer wrapper.
# Imported by ``Chunk.__post_init__`` so that token counts are always
# accurate without each call site needing to manage a tokenizer instance.
# ---------------------------------------------------------------------------

class _TokenizerWrapper:
    """Lazy singleton wrapper around the BGE AutoTokenizer.

    The tokenizer is only loaded on first use (~0.5s). Subsequent calls
    are instant. Used by ``Chunk._count_tokens`` for accurate sizing.
    """

    def __init__(self) -> None:
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # noqa: WPS433

            self._tokenizer = AutoTokenizer.from_pretrained(
                "BAAI/bge-small-en-v1.5"
            )
        return self._tokenizer

    def count(self, text: str) -> int:
        """Return the number of BGE tokens in ``text``."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        """Return BGE token IDs for ``text`` (no special tokens)."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


_BGE_TOKENIZER = _TokenizerWrapper()
