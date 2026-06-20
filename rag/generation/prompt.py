"""
Prompt construction for grounded answer generation.

Builds a chat-completion prompt that instructs the LLM to:

1. Answer **only** using the provided context chunks.
2. Cite sources using ``[filename, p.X]`` notation.
3. Say "I don't know" if the context doesn't contain the answer.

The prompt is a list of ``{"role": ..., "content": ...}`` dicts ready
for the OpenAI Chat Completions API (``client.chat.completions.create``).
"""

from __future__ import annotations

import logging
from typing import Any

from rag.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — instructs the LLM on citation format and grounding rules.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the user's question using ONLY the information provided "
    "in the context passages below. "
    "If the context does not contain enough information to answer, "
    "respond with: \"I don't know based on the provided documents.\"\n\n"
    "CRITICAL RULES:\n"
    "1. Every factual claim must be followed by a citation in the format "
    "[filename, p.X] where X is the page number.\n"
    "2. If multiple sources support a claim, list all: "
    "[filename1, p.X][filename2, p.Y].\n"
    "3. Do not use any outside knowledge. Only use the provided context.\n"
    "4. Do not make up information or speculate.\n"
    "5. Keep the answer concise and directly address the question.\n"
    "6. If the question is ambiguous, state your interpretation and answer "
    "based on the context."
)

# Hard cap on total context characters to stay within token limits.
# ~12,000 chars ≈ 3,000 tokens (room for a 1K-token answer within a 4K window).
_MAX_CONTEXT_CHARS = 12_000


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block.

    Each chunk is rendered as::

        [1] Source: filename.pdf, Page 5
        <chunk text>

    Parameters
    ----------
    chunks:
        Retrieved chunks (already filtered by score threshold).

    Returns
    -------
    str
        Formatted context string, truncated to ``_MAX_CONTEXT_CHARS``.
    """
    if not chunks:
        return ""

    lines: list[str] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks, start=1):
        header = "[{}] Source: {}, Page {}".format(
            idx, chunk.source_file, chunk.page
        )
        entry = "{}\n{}".format(header, chunk.text)

        if total_chars + len(entry) > _MAX_CONTEXT_CHARS:
            logger.warning(
                "Context truncated at chunk %d/%d (max %d chars)",
                idx - 1,
                len(chunks),
                _MAX_CONTEXT_CHARS,
            )
            break

        lines.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(lines)


def build_chat_messages(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """Build the full chat-completion message list.

    Parameters
    ----------
    query:
        The user's question.
    chunks:
        Retrieved chunks to use as context.

    Returns
    -------
    list[dict[str, str]]
        Messages in OpenAI Chat format::

            [
                {"role": "system", "content": "<system prompt>"},
                {"role": "user", "content": "<context + question>"},
            ]
    """
    context_block = build_context_block(chunks)

    if context_block:
        user_content = (
            "Context passages:\n\n"
            "{}\n\n"
            "---\n\n"
            "Question: {}\n\n"
            "Answer the question using ONLY the context above. "
            "Cite sources as [filename, p.X]."
        ).format(context_block, query)
    else:
        user_content = (
            "No relevant context was found for the question: "
            "'{}'.\n\n"
            "Respond with: \"I don't know based on the provided documents.\""
        ).format(query)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.debug(
        "Built prompt: %d chars context, %d chunks",
        len(context_block),
        len(chunks),
    )

    return messages
