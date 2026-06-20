"""
Custom exception hierarchy for the RAG Chatbot.

All application errors inherit from ``RagError`` so the API layer can
catch a single base class and translate it to an HTTP response.
"""

from __future__ import annotations


class RagError(Exception):
    """Base exception for all RAG pipeline errors."""


class IngestionError(RagError):
    """Raised when a PDF cannot be parsed or processed."""


class EmbeddingError(RagError):
    """Raised when embedding generation fails."""


class RetrievalError(RagError):
    """Raised when vector search fails."""


class GenerationError(RagError):
    """Raised when the LLM answer-generation step fails."""
