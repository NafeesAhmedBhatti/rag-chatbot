"""
LLM-based answer generation with citation extraction.

The ``Generator`` takes a ``RetrievalResult`` (the chunks found by the
retriever) and produces a ``GenerationResult`` containing:

1. A natural-language answer grounded in the retrieved chunks.
2. Parsed citation objects (filename + page number) extracted from
   the LLM's output.

Uses the OpenAI-compatible Chat Completions API (works with OpenAI,
Azure OpenAI, or any compatible gateway such as ``llm.drytis.ai``).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rag.config import Settings, settings as default_settings
from rag.exceptions import GenerationError
from rag.generation.prompt import build_chat_messages
from rag.retrieval.retriever import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation regex
# ---------------------------------------------------------------------------
# Matches patterns like:
#   [report.pdf, p.5]
#   [report.pdf, p.5][notes.pdf, p.10]
#   [report.pdf p.5]          (comma optional)
#   [report.pdf, p.5, p.6]    (multiple pages, same file)
#
# Group 1 = filename, Group 2 = page number (or comma-separated pages).
#
_CITATION_RE = re.compile(
    r"\[([^\]]+?)\s*,?\s*p\.?\s*(\d+(?:\s*,\s*p\.?\s*\d+)*)\]",
    re.IGNORECASE,
)

# Fallback: "[p.5]" without a filename (rare, but some LLMs do this).
_PAGE_ONLY_CITATION_RE = re.compile(
    r"\[p\.?\s*(\d+)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Citation:
    """A single citation parsed from the LLM's answer.

    Attributes
    ----------
    filename:
        Source PDF filename (may be empty if the LLM omitted it).
    page:
        Page number in the source PDF.
    raw:
        The raw citation string as it appeared in the answer.
    """

    filename: str
    page: int
    raw: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON API responses)."""
        return {
            "filename": self.filename,
            "page": self.page,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class GenerationResult:
    """The outcome of an answer-generation call.

    Attributes
    ----------
    answer:
        The LLM's natural-language answer.
    citations:
        Parsed citation objects extracted from the answer.
    model:
        The LLM model name used.
    latency_ms:
        Wall-clock time for the LLM API call.
    tokens_used:
        Total tokens reported by the API (0 if unavailable).
    used_fallback:
        True if the generator produced a no-context fallback answer
        instead of calling the LLM.
    """

    answer: str
    citations: list[Citation]
    model: str
    latency_ms: float
    tokens_used: int
    used_fallback: bool = False


class Generator:
    """Generates grounded answers using an OpenAI-compatible LLM.

    Parameters
    ----------
    config:
        Application settings (provides API key, base URL, model name,
        temperature, max tokens, timeout). Defaults to the global
        singleton.
    client:
        Pre-constructed OpenAI client instance (for testing/injection).
        If ``None``, a client is created lazily on first use.
    """

    def __init__(
        self,
        config: Settings | None = None,
        client: Any = None,
    ) -> None:
        self.config = config or default_settings
        self._client = client

    # ------------------------------------------------------------------
    # Lazy client loading
    # ------------------------------------------------------------------

    def _load_client(self) -> Any:
        """Create the OpenAI client on first use."""
        if self._client is not None:
            return self._client

        from openai import OpenAI  # noqa: WPS433

        if not self.config.openai_api_key:
            raise GenerationError(
                "OPENAI_API_KEY is not set. Cannot generate answers."
            )

        self._client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            timeout=self.config.llm_timeout_seconds,
        )
        logger.info(
            "OpenAI client initialized (base_url=%s)",
            self.config.openai_base_url,
        )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, retrieval_result: RetrievalResult) -> GenerationResult:
        """Generate an answer from retrieved chunks.

        Parameters
        ----------
        retrieval_result:
            The output of ``Retriever.retrieve()``.

        Returns
        -------
        GenerationResult
            The answer, parsed citations, and timing metadata.

        Raises
        ------
        GenerationError
            If the LLM API call fails or the key is missing.
        """
        # If no chunks passed the threshold, return a fallback answer
        # without calling the LLM (saves tokens + latency).
        if not retrieval_result.chunks:
            logger.info(
                "No chunks for query '%s'; returning no-context fallback",
                retrieval_result.query[:80],
            )
            return GenerationResult(
                answer=(
                    "I don't know based on the provided documents. "
                    "No relevant passages were found for your question."
                ),
                citations=[],
                model=self.config.llm_model,
                latency_ms=0.0,
                tokens_used=0,
                used_fallback=True,
            )

        # Build the prompt.
        messages = build_chat_messages(
            retrieval_result.query, retrieval_result.chunks
        )

        client = self._load_client()
        t0 = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                "LLM API call failed: {}".format(exc)
            ) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Extract the answer text.
        answer = response.choices[0].message.content or ""
        answer = answer.strip()

        # Parse token usage if available.
        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = getattr(response.usage, "total_tokens", 0) or 0

        # Extract citations.
        citations = extract_citations(
            answer, retrieval_result.chunks
        )

        logger.info(
            "Generated answer (%d chars, %d citations, %d tokens, %.1f ms)",
            len(answer),
            len(citations),
            tokens_used,
            latency_ms,
        )

        return GenerationResult(
            answer=answer,
            citations=citations,
            model=self.config.llm_model,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )


# ---------------------------------------------------------------------------
# Citation extraction (module-level functions for testability)
# ---------------------------------------------------------------------------

def extract_citations(
    answer: str,
    chunks: list[RetrievedChunk] | None = None,
) -> list[Citation]:
    """Parse ``[filename, p.X]`` citations from an answer string.

    Parameters
    ----------
    answer:
        The LLM-generated answer text.
    chunks:
        The retrieved chunks (used as a fallback to assign a filename
        when the LLM writes ``[p.X]`` without naming a source).

    Returns
    -------
    list[Citation]
        Deduplicated citations, preserving first-seen order.
    """
    if not answer:
        return []

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    # Track page numbers that were already cited via the primary pattern,
    # so the fallback doesn't duplicate them.
    cited_pages: set[int] = set()

    # Primary pattern: [filename, p.X]
    for match in _CITATION_RE.finditer(answer):
        filename = match.group(1).strip().rstrip(",")

        # Skip if "filename" is actually a page-only reference like "p.5"
        # (these will be handled by the fallback below).
        if not filename or re.match(r"^p\.?\s*\d+$", filename.strip(), re.IGNORECASE):
            continue

        page_str = match.group(2)

        # Handle multi-page citations like "5, 6, 7".
        page_nums = re.findall(r"\d+", page_str)
        for page_num in page_nums:
            page = int(page_num)
            cited_pages.add(page)
            key = (filename.lower(), page)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        filename=filename,
                        page=page,
                        raw=match.group(0),
                    )
                )

    # Fallback: [p.X] without a filename → assign from chunks if available.
    if chunks:
        # Build a map of page → source_file from retrieved chunks.
        page_to_source: dict[int, str] = {}
        for chunk in chunks:
            if chunk.page not in page_to_source:
                page_to_source[chunk.page] = chunk.source_file

        for match in _PAGE_ONLY_CITATION_RE.finditer(answer):
            page = int(match.group(1))
            # Skip if already cited via the primary pattern.
            if page in cited_pages:
                continue
            filename = page_to_source.get(page, "unknown")
            key = (filename.lower(), page)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        filename=filename,
                        page=page,
                        raw=match.group(0),
                    )
                )
    else:
        # No chunks: still extract [p.X] citations with 'unknown' filename.
        for match in _PAGE_ONLY_CITATION_RE.finditer(answer):
            page = int(match.group(1))
            if page in cited_pages:
                continue
            key = ("unknown", page)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        filename="unknown",
                        page=page,
                        raw=match.group(0),
                    )
                )

    return citations
