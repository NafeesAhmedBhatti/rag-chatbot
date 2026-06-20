"""
Tests for rag.processing.cleaner.TextCleaner and LanguageDetector.

Covers:
    - Unicode NFC normalization
    - Control character removal
    - Ligature fixing
    - Smart quote / dash normalization
    - De-hyphenation across line breaks
    - Whitespace collapsing
    - Empty / edge-case inputs
    - Language detection (English, French, Spanish, short text, empty)
"""

from __future__ import annotations

import unicodedata

import pytest

from rag.processing.cleaner import LanguageDetector, TextCleaner


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector()


# ---------------------------------------------------------------------------
# TextCleaner tests
# ---------------------------------------------------------------------------

class TestUnicodeNormalization:
    """NFC normalization and basic Unicode handling."""

    def test_nfc_normalization(self, cleaner: TextCleaner) -> None:
        # NFD (decomposed) form of é = e + combining accent.
        nfd = "caf\u0065\u0301"  # café in NFD
        nfc = "caf\u00e9"         # café in NFC
        result = cleaner.clean(nfd)
        assert result == unicodedata.normalize("NFC", nfd)

    def test_already_clean_text_unchanged(self, cleaner: TextCleaner) -> None:
        text = "This is already clean text with no issues."
        assert cleaner.clean(text) == text


class TestControlChars:
    """Removal of control characters."""

    def test_removes_null_byte(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("Hello\x00World") == "HelloWorld"

    def test_removes_bel(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("Hello\x07World") == "HelloWorld"

    def test_collapses_tab_to_space(self, cleaner: TextCleaner) -> None:
        # Tabs in PDF text are formatting artifacts; they should be
        # collapsed to a single space (consistent whitespace handling).
        assert cleaner.clean("Hello\tWorld") == "Hello World"

    def test_preserves_newline(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("Hello\nWorld") == "Hello\nWorld"

    def test_removes_multiple_control_chars(self, cleaner: TextCleaner) -> None:
        text = "A\x01\x02\x03B"
        assert cleaner.clean(text) == "AB"


class TestLigatures:
    """Fixing Unicode ligatures to ASCII equivalents."""

    def test_fixes_fi_ligature(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("\ufb01nancial") == "financial"

    def test_fixes_fl_ligature(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("\uFB02ow") == "flow"

    def test_fixes_ff_ligature(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("\uFB00ord") == "fford"

    def test_fixes_ffi_ligature(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("\uFB03ce") == "ffice"

    def test_fixes_multiple_ligatures(self, cleaner: TextCleaner) -> None:
        text = "\ufb01nancial \uFB02ow \uFB03ce"
        assert cleaner.clean(text) == "financial flow ffice"


class TestSmartQuotes:
    """Normalization of smart quotes and dashes to ASCII."""

    def test_fixes_left_double_quote(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("\u201cHello\u201d") == '"Hello"'

    def test_fixes_left_single_quote(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("\u2018Hello\u2019") == "'Hello'"

    def test_fixes_en_dash(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("en\u2013dash") == "en-dash"

    def test_fixes_em_dash(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("em\u2014dash") == "em-dash"

    def test_fixes_ellipsis(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("wait\u2026") == "wait..."

    def test_fixes_nonbreaking_space(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("a\u00a0b") == "a b"


class TestDeHyphenation:
    """Fixing words split by hyphens across line breaks."""

    def test_basic_dehyphenation(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("exam-\nple") == "example"

    def test_dehyphenation_with_spaces(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("exam-\n ple") == "example"

    def test_preserves_non_linebreak_hyphens(self, cleaner: TextCleaner) -> None:
        # Hyphens NOT at line breaks should be preserved.
        assert cleaner.clean("well-known") == "well-known"

    def test_dehyphenation_multiple(self, cleaner: TextCleaner) -> None:
        text = "exam-\nple and an-\nother"
        assert cleaner.clean(text) == "example and another"


class TestWhitespace:
    """Collapsing excessive whitespace."""

    def test_collapses_multiple_spaces(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("hello    world") == "hello world"

    def test_collapses_tabs(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("hello\t\tworld") == "hello world"

    def test_collapses_multiple_newlines(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("para1\n\n\n\npara2") == "para1\n\npara2"

    def test_strips_leading_trailing(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("  hello  ") == "hello"

    def test_collapses_mixed_whitespace(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("  hello \t world  ") == "hello world"


class TestEdgeCases:
    """Empty and degenerate inputs."""

    def test_empty_string(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("") == ""

    def test_only_whitespace(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("   \n\n\t  ") == ""

    def test_single_char(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("A") == "A"

    def test_preserves_paragraphs(self, cleaner: TextCleaner) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        assert cleaner.clean(text) == "First paragraph.\n\nSecond paragraph."


# ---------------------------------------------------------------------------
# LanguageDetector tests
# ---------------------------------------------------------------------------

class TestLanguageDetector:
    """Language detection via langdetect."""

    ENGLISH_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is used for testing language detection. "
        "It should be long enough for reliable results."
    )

    FRENCH_TEXT = (
        "Le renard brun rapide saute par-dessus le chien paresseux. "
        "Cette phrase est utilisée pour tester la détection de langue. "
        "Elle devrait être assez longue pour des résultats fiables."
    )

    SPANISH_TEXT = (
        "El rápido zorro marrón salta sobre el perro perezoso. "
        "Esta oración se usa para probar la detección de idioma. "
        "Debe ser lo suficientemente larga para resultados confiables."
    )

    def test_detects_english(self, detector: LanguageDetector) -> None:
        result = detector.detect(self.ENGLISH_TEXT)
        assert result.code == "en"

    def test_detects_french(self, detector: LanguageDetector) -> None:
        result = detector.detect(self.FRENCH_TEXT)
        assert result.code == "fr"

    def test_detects_spanish(self, detector: LanguageDetector) -> None:
        result = detector.detect(self.SPANISH_TEXT)
        assert result.code == "es"

    def test_short_text_returns_unreliable(self, detector: LanguageDetector) -> None:
        result = detector.detect("hi")
        assert result.reliable is False

    def test_empty_text_returns_default(self, detector: LanguageDetector) -> None:
        result = detector.detect("")
        assert result.code == "en"
        assert result.reliable is False

    def test_language_code_is_iso639_1(self, detector: LanguageDetector) -> None:
        result = detector.detect(self.ENGLISH_TEXT)
        assert len(result.code) == 2  # ISO 639-1 codes are 2 letters
