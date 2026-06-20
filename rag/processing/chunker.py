"""
Token-aware recursive text chunking.

Splits text into chunks that are sized in BGE tokens (not characters),
respecting paragraph and sentence boundaries where possible. Each
chunk carries full metadata (source file, page number, character
offsets, language) for retrieval and citation.

Algorithm
---------
The chunker uses a **recursive character splitter** strategy:

    1. Split the text on the largest meaningful boundary
       (``\\n\\n`` → paragraphs).
    2. If a resulting piece exceeds ``chunk_size`` tokens, split it
       on the next-smaller boundary (``\\n`` → lines).
    3. Continue recursing through ``. `` (sentences), `` `` (words),
       and finally character-level if needed.
    4. Merge consecutive pieces until adding the next one would
       exceed ``chunk_size`` tokens.
    5. Apply overlap: the last ``overlap`` tokens of chunk N are
       prepended to chunk N+1.

This produces chunks that respect natural text boundaries (no
mid-sentence splits unless a sentence itself exceeds ``chunk_size``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag.config import Settings, settings as default_settings
from rag.models import Chunk, _BGE_TOKENIZER

logger = logging.getLogger(__name__)

# Separator hierarchy: largest semantic boundary first.
# Each entry is (separator, is_recursive).
# - Paragraph boundaries (double newline).
# - Line boundaries (single newline).
# - Sentence boundaries (period + space).
# - Word boundaries (space).
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", " "]


class Chunker:
    """Token-aware recursive text chunker.

    Parameters
    ----------
    config:
        Application settings (provides ``chunk_size``, ``chunk_overlap``,
        and the embedding model name). Defaults to the global singleton.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self._tokenizer = _BGE_TOKENIZER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        source_file: str,
        page: int,
        language: str = "en",
    ) -> list[Chunk]:
        """Split a page of text into metadata-rich chunks.

        Parameters
        ----------
        text:
            Cleaned text from a single PDF page.
        source_file:
            Original PDF filename (used for chunk IDs and citations).
        page:
            1-based page number.
        language:
            ISO 639-1 language code for the page.

        Returns
        -------
        list[Chunk]
            Chunks in reading order. May be empty if ``text`` is empty
            or whitespace-only.
        """
        if not text or not text.strip():
            logger.debug("Empty text for %s p.%d, skipping chunking", source_file, page)
            return []

        # Recursive split into token-bounded pieces.
        pieces = self._split_text(text)

        # Merge pieces into final chunks with overlap.
        chunks = self._merge_with_overlap(pieces, text, source_file, page, language)

        logger.debug(
            "Chunked %s p.%d: %d chunks (avg %d tokens)",
            source_file,
            page,
            len(chunks),
            sum(c.token_count for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    def chunk_page_content(
        self,
        page_text: str,
        source_file: str,
        page: int,
        language: str = "en",
    ) -> list[Chunk]:
        """Alias for ``chunk()`` — matches the interface in patterns.md."""
        return self.chunk(page_text, source_file, page, language)

    # ------------------------------------------------------------------
    # Recursive splitting
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text into pieces under ``chunk_size`` tokens.

        Uses the separator hierarchy to respect natural boundaries.
        Returns a list of text pieces, each under the token limit
        (unless a single token-spanning unit — like a very long word —
        exceeds the limit).
        """
        return self._recursive_split(text, _SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split ``text`` using the separator hierarchy.

        If the text is already under the token limit, return it as a
        single piece. Otherwise, split on the first available separator
        and recurse on each part.
        """
        if self._count_tokens(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # If we've exhausted all separators, return the text as-is
        # (it's a single unbreakable unit — e.g., a very long URL or
        # token with no spaces).
        if not separators:
            return [text.strip()] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split on the current separator.
        parts = self._split_on_separator(text, separator)

        results: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if self._count_tokens(part) <= self.chunk_size:
                results.append(part)
            else:
                # Recurse with the next smaller separator.
                results.extend(
                    self._recursive_split(part, remaining_separators)
                )

        return results

    def _split_on_separator(self, text: str, separator: str) -> list[str]:
        """Split text on ``separator``, preserving the separator at the
        end of each piece (so sentence boundaries aren't lost).
        """
        if separator == " ":
            # Word-level: split by whitespace.
            return text.split(" ")

        # For other separators, split and re-append the separator.
        parts = text.split(separator)
        # Re-append the separator to each part except the last.
        return [p + separator for p in parts[:-1]] + [parts[-1]]

    # ------------------------------------------------------------------
    # Merging with overlap
    # ------------------------------------------------------------------

    def _merge_with_overlap(
        self,
        pieces: list[str],
        full_text: str,
        source_file: str,
        page: int,
        language: str,
    ) -> list[Chunk]:
        """Merge pieces into chunks, applying overlap between adjacent chunks.

        Strategy:
            - Accumulate pieces until the next piece would exceed
              ``chunk_size`` tokens.
            - Start the next chunk with the overlap (last N tokens)
              from the current chunk.
            - Track character offsets in the original text.
        """
        if not pieces:
            return []

        # Extract the stem of the filename for chunk IDs.
        # e.g., "annual_report.pdf" → "annual_report"
        source_stem = Path(source_file).stem

        chunks: list[Chunk] = []
        current_pieces: list[str] = []
        current_tokens = 0
        chunk_index = 0

        # For character offset tracking, we search for piece text in
        # the full page text and track our position.
        char_cursor = 0

        for piece in pieces:
            piece_tokens = self._count_tokens(piece)

            # If this piece alone exceeds chunk_size, emit it as its
            # own chunk (shouldn't happen often after recursive split).
            if piece_tokens > self.chunk_size:
                # Flush current accumulation first.
                if current_pieces:
                    chunks.append(
                        self._build_chunk(
                            current_pieces,
                            full_text,
                            source_stem,
                            page,
                            chunk_index,
                            language,
                            char_cursor,
                        )
                    )
                    chunk_index += 1
                    current_pieces = []
                    current_tokens = 0

                chunks.append(
                    self._build_chunk(
                        [piece],
                        full_text,
                        source_stem,
                        page,
                        chunk_index,
                        language,
                        char_cursor,
                    )
                )
                chunk_index += 1
                continue

            # Check if adding this piece would exceed chunk_size.
            if current_tokens + piece_tokens > self.chunk_size and current_pieces:
                # Flush current chunk.
                chunk_text = "".join(current_pieces)
                chunks.append(
                    self._build_chunk_from_text(
                        chunk_text,
                        full_text,
                        source_stem,
                        page,
                        chunk_index,
                        language,
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap from the previous chunk.
                overlap_text = self._get_overlap(chunk_text)
                current_pieces = [overlap_text] if overlap_text else []
                current_tokens = self._count_tokens(overlap_text) if overlap_text else 0

            current_pieces.append(piece)
            current_tokens += piece_tokens

        # Flush remaining pieces.
        if current_pieces:
            chunk_text = "".join(current_pieces)
            chunks.append(
                self._build_chunk_from_text(
                    chunk_text,
                    full_text,
                    source_stem,
                    page,
                    chunk_index,
                    language,
                )
            )

        return chunks

    def _get_overlap(self, chunk_text: str) -> str:
        """Extract the last ``chunk_overlap`` tokens from ``chunk_text``.

        Returns the decoded text of those tokens. If the chunk is
        shorter than ``chunk_overlap``, returns the full chunk.
        """
        token_ids = self._tokenizer.encode(chunk_text)
        if len(token_ids) <= self.chunk_overlap:
            return chunk_text

        overlap_ids = token_ids[-self.chunk_overlap:]
        overlap_text = self._tokenizer.decode(overlap_ids)
        return overlap_text.strip()

    def _build_chunk_from_text(
        self,
        chunk_text: str,
        full_text: str,
        source_stem: str,
        page: int,
        chunk_index: int,
        language: str,
    ) -> Chunk:
        """Build a Chunk from its text, computing char offsets by
        searching for the text in the full page text.
        """
        chunk_text = chunk_text.strip()
        # Find character offsets by locating this chunk in the full text.
        # For overlap chunks, the chunk text may not be a contiguous
        # substring — in that case we use 0/len as fallback.
        char_start = full_text.find(chunk_text[:100])  # search by prefix
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(chunk_text)

        return Chunk(
            chunk_id=f"{source_stem}_p{page}_c{chunk_index}",
            text=chunk_text,
            source_file=f"{source_stem}.pdf",
            page=page,
            char_start=char_start,
            char_end=char_end,
            chunk_index=chunk_index,
            language=language,
        )

    def _build_chunk(
        self,
        pieces: list[str],
        full_text: str,
        source_stem: str,
        page: int,
        chunk_index: int,
        language: str,
        char_cursor: int,
    ) -> Chunk:
        """Build a Chunk from a list of pieces (used for oversized pieces)."""
        chunk_text = "".join(pieces).strip()
        return self._build_chunk_from_text(
            chunk_text,
            full_text,
            source_stem,
            page,
            chunk_index,
            language,
        )

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count BGE tokens in ``text``."""
        return self._tokenizer.count(text)
