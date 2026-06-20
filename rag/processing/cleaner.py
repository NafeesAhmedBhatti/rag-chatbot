"""
Text cleaning and normalization.

Takes raw text from the PDF loader (which may contain encoding
artifacts, broken hyphenation, excessive whitespace, and Unicode
oddities from OCR) and produces clean, normalized text ready for
chunking and embedding.

Cleaning steps (applied in order):
    1. Unicode NFC normalization (canonical composition).
    2. Remove control characters (except tab/newline).
    3. Fix common ligatures (ﬁ→fi, ﬂ→fl, etc.).
    4. Normalize smart quotes and dashes to ASCII.
    5. De-hyphenate words split across line breaks (e.g. "exam-\nple" → "example").
    6. Collapse multiple whitespace/newlines into single spaces.
    7. Strip leading/trailing whitespace.

Also provides ``LanguageDetector`` for per-page language identification.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from rag.models import LanguageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precompiled regex patterns
# ---------------------------------------------------------------------------

# Control characters: Cc (control) and Cf (format) categories, except
# \t (tab) and \n (newline) which we preserve for whitespace collapsing.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Ligatures and special Unicode characters → ASCII equivalents.
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ
    "\ufb06": "st",   # ﬆ
}

# Smart quotes, dashes, and other punctuation normalization.
_PUNCTUATION_MAP: dict[str, str] = {
    "\u2018": "'",   # ‘
    "\u2019": "'",   # ’
    "\u201a": "'",   # ‚
    "\u201b": "'",   # ‛
    "\u201c": '"',   # “
    "\u201d": '"',   # ”
    "\u201e": '"',   # „
    "\u2013": "-",   # – (en dash)
    "\u2014": "-",   # — (em dash)
    "\u2026": "...", # … (ellipsis)
    "\u00a0": " ",   # non-breaking space
    "\u2028": "\n",  # line separator
    "\u2029": "\n",  # paragraph separator
}

# De-hyphenation: a word split by a hyphen at end of line.
# Matches: "exam-" followed by optional whitespace then "ple" at line start.
# Captures the two parts so we can join them.
_HYPHEN_SPLIT_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

# Multiple whitespace (including newlines) → single space.
_MULTI_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# TextCleaner
# ---------------------------------------------------------------------------

class TextCleaner:
    """Clean and normalize text from PDF/OCR extraction.

    Stateless and thread-safe — the same instance can be shared across
    threads. All methods are pure functions of their input.
    """

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to ``text``.

        Parameters
        ----------
        text:
            Raw text from PDF extraction or OCR.

        Returns
        -------
        str
            Cleaned, normalized text.
        """
        if not text:
            return ""

        # 1. Unicode NFC normalization.
        text = unicodedata.normalize("NFC", text)

        # 2. Remove control characters (keep \t and \n).
        text = _CONTROL_CHAR_RE.sub("", text)

        # 3. Fix ligatures.
        for ligature, replacement in _LIGATURE_MAP.items():
            text = text.replace(ligature, replacement)

        # 4. Normalize smart quotes and dashes.
        for char, replacement in _PUNCTUATION_MAP.items():
            text = text.replace(char, replacement)

        # 5. De-hyphenate words split across lines.
        text = _HYPHEN_SPLIT_RE.sub(r"\1\2", text)

        # 6. Collapse whitespace.
        text = _MULTI_WS_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

        # 7. Strip leading/trailing whitespace.
        text = text.strip()

        return text


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------

class LanguageDetector:
    """Detect the language of a text using langdetect.

    Wraps langdetect with error handling and deterministic seeding
    (langdetect uses a probabilistic algorithm; seeding ensures
    consistent results across runs).
    """

    def __init__(self) -> None:
        # Set the seed for deterministic detection.
        try:
            from langdetect import detector_factory

            detector_factory.DetectorFactory.seed = 0
        except Exception:  # noqa: BLE001
            logger.warning("Could not set langdetect seed; detection may vary")

    def detect(self, text: str) -> LanguageResult:
        """Detect the language of ``text``.

        Parameters
        ----------
        text:
            Text to analyze. Should be at least a few sentences for
            reliable detection.

        Returns
        -------
        LanguageResult
            Contains the ISO 639-1 language code and a ``reliable``
            flag. Returns ``code="en", reliable=False`` as a safe
            default if detection fails.
        """
        if not text or len(text.strip()) < 10:
            logger.debug("Text too short for reliable language detection")
            return LanguageResult(code="en", reliable=False)

        try:
            from langdetect import detect_langs

            results = detect_langs(text)
            if not results:
                return LanguageResult(code="en", reliable=False)

            top = results[0]
            # langdetect's is_reliable: true if top-1 probability > 0.5.
            is_reliable = top.prob > 0.5
            return LanguageResult(code=top.lang, reliable=is_reliable)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Language detection failed: %s", exc)
            return LanguageResult(code="en", reliable=False)
