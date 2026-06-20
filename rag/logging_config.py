"""
Structured logging setup for the RAG Chatbot.

Configures the root logger and a consistent format across all modules.
Call ``setup_logging()`` once at application startup (before any
module-level ``logging.getLogger`` calls produce records).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

# Standard log format: timestamp, level, logger name, message.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging with a consistent format.

    Parameters
    ----------
    level:
        Logging level as a string ("DEBUG", "INFO", ...) or an ``int``
        (``logging.INFO``). Defaults to ``"INFO"``.
    """
    if isinstance(level, str):
        level = level.upper()
    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Reset any existing handlers so re-calls (tests, reload) don't
    # accumulate duplicate handlers.
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

    # Quieten noisy third-party loggers that would flood output.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (for metadata)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
