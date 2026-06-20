"""
Tests for rag.ingestion.pdf_loader.

Covers:
    - Native text extraction (PyMuPDF)
    - OCR extraction (Tesseract) on image-only PDFs
    - Mixed PDFs (per-page heuristic)
    - Error handling (missing file, non-PDF, empty PDF)
    - PageContent dataclass behaviour
    - load_to_dict summary
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.exceptions import IngestionError
from rag.ingestion.pdf_loader import PDFLoader
from rag.models import ExtractionMethod, PageContent

# Path to generated sample PDFs.
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
TEXT_NATIVE_PDF = SAMPLE_DIR / "text_native.pdf"
SCANNED_IMAGE_PDF = SAMPLE_DIR / "scanned_image.pdf"
MIXED_PDF = SAMPLE_DIR / "mixed.pdf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader() -> PDFLoader:
    """Default PDFLoader (uses global OCR fallback chain)."""
    return PDFLoader()


# ---------------------------------------------------------------------------
# PageContent unit tests
# ---------------------------------------------------------------------------

class TestPageContent:
    """Tests for the PageContent dataclass."""

    def test_char_count_auto_computed(self) -> None:
        pc = PageContent(
            page=1,
            text="hello world",
            extraction_method=ExtractionMethod.NATIVE,
        )
        assert pc.char_count == 11

    def test_char_count_empty_text(self) -> None:
        pc = PageContent(
            page=1,
            text="",
            extraction_method=ExtractionMethod.NATIVE,
        )
        assert pc.char_count == 0

    def test_frozen(self) -> None:
        pc = PageContent(
            page=1,
            text="abc",
            extraction_method=ExtractionMethod.NATIVE,
        )
        with pytest.raises((AttributeError, Exception)):
            pc.page = 2  # type: ignore[misc]

    def test_extraction_method_values(self) -> None:
        assert ExtractionMethod.NATIVE.value == "native"
        assert ExtractionMethod.OCR.value == "ocr"


# ---------------------------------------------------------------------------
# Native text extraction
# ---------------------------------------------------------------------------

class TestNativeExtraction:
    """Tests for native (selectable-text) PDF extraction."""

    def test_returns_correct_page_count(self, loader: PDFLoader) -> None:
        pages = loader.load(TEXT_NATIVE_PDF)
        assert len(pages) == 3

    def test_all_pages_native_method(self, loader: PDFLoader) -> None:
        pages = loader.load(TEXT_NATIVE_PDF)
        for page in pages:
            assert page.extraction_method is ExtractionMethod.NATIVE

    def test_pages_have_text_content(self, loader: PDFLoader) -> None:
        pages = loader.load(TEXT_NATIVE_PDF)
        for page in pages:
            assert page.char_count > 100

    def test_page_numbers_are_sequential_1_based(self, loader: PDFLoader) -> None:
        pages = loader.load(TEXT_NATIVE_PDF)
        for i, page in enumerate(pages):
            assert page.page == i + 1

    def test_text_contains_known_content(self, loader: PDFLoader) -> None:
        pages = loader.load(TEXT_NATIVE_PDF)
        # The test PDF contains the word "Page" on each page header.
        full_text = " ".join(p.text for p in pages)
        assert "Page 1" in full_text
        assert "Page 2" in full_text
        assert "Page 3" in full_text

    def test_text_contains_paragraph_content(self, loader: PDFLoader) -> None:
        pages = loader.load(TEXT_NATIVE_PDF)
        full_text = " ".join(p.text for p in pages)
        assert "Artificial intelligence" in full_text


# ---------------------------------------------------------------------------
# OCR extraction (scanned/image-only PDFs)
# ---------------------------------------------------------------------------

class TestOCRExtraction:
    """Tests for OCR extraction on image-only PDFs."""

    def test_returns_correct_page_count(self, loader: PDFLoader) -> None:
        pages = loader.load(SCANNED_IMAGE_PDF)
        assert len(pages) == 2

    def test_all_pages_use_ocr(self, loader: PDFLoader) -> None:
        pages = loader.load(SCANNED_IMAGE_PDF)
        for page in pages:
            assert page.extraction_method is ExtractionMethod.OCR

    def test_ocr_produces_text(self, loader: PDFLoader) -> None:
        pages = loader.load(SCANNED_IMAGE_PDF)
        for page in pages:
            assert page.char_count > 50  # OCR should extract meaningful text

    def test_ocr_text_is_readable(self, loader: PDFLoader) -> None:
        """OCR should extract words that are recognisable (not garbage)."""
        pages = loader.load(SCANNED_IMAGE_PDF)
        full_text = " ".join(p.text for p in pages).lower()
        # The test PDF contains "Scanned Page" headers — OCR should find them.
        assert "scanned" in full_text or "page" in full_text


# ---------------------------------------------------------------------------
# Mixed PDFs (per-page heuristic)
# ---------------------------------------------------------------------------

class TestMixedPDF:
    """Tests for PDFs containing both native-text and image-only pages."""

    def test_returns_correct_page_count(self, loader: PDFLoader) -> None:
        pages = loader.load(MIXED_PDF)
        assert len(pages) == 4

    def test_pages_1_2_are_native(self, loader: PDFLoader) -> None:
        pages = loader.load(MIXED_PDF)
        assert pages[0].extraction_method is ExtractionMethod.NATIVE
        assert pages[1].extraction_method is ExtractionMethod.NATIVE

    def test_pages_3_4_are_ocr(self, loader: PDFLoader) -> None:
        pages = loader.load(MIXED_PDF)
        assert pages[2].extraction_method is ExtractionMethod.OCR
        assert pages[3].extraction_method is ExtractionMethod.OCR

    def test_all_pages_have_text(self, loader: PDFLoader) -> None:
        pages = loader.load(MIXED_PPDF := MIXED_PDF)
        for page in pages:
            assert page.char_count > 50

    def test_ocr_pages_have_more_text_than_threshold(self, loader: PDFLoader) -> None:
        pages = loader.load(MIXED_PDF)
        for page in pages:
            if page.extraction_method is ExtractionMethod.OCR:
                assert page.char_count >= 20  # above the OCR threshold


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error conditions."""

    def test_missing_file_raises_ingestion_error(self, loader: PDFLoader) -> None:
        with pytest.raises(IngestionError, match="not found"):
            loader.load("/workspace/data/sample/nonexistent.pdf")

    def test_non_pdf_file_raises_ingestion_error(self, loader: PDFLoader, tmp_path: Path) -> None:
        bad_file = tmp_path / "not_a_pdf.pdf"
        bad_file.write_text("this is not a PDF")
        with pytest.raises(IngestionError, match="Failed to open PDF"):
            loader.load(bad_file)

    def test_wrong_extension_raises_ingestion_error(
        self, loader: PDFLoader, tmp_path: Path
    ) -> None:
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("plain text")
        with pytest.raises(IngestionError, match="Expected a .pdf"):
            loader.load(txt_file)

    def test_directory_raises_ingestion_error(self, loader: PDFLoader, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="not a file"):
            loader.load(tmp_path)


# ---------------------------------------------------------------------------
# OCR threshold configuration
# ---------------------------------------------------------------------------

class TestOCRThreshold:
    """Tests that the OCR threshold config is respected."""

    def test_high_threshold_forces_ocr_on_all_pages(self) -> None:
        """With a threshold of 10000, even text pages trigger OCR."""
        config = Settings(ocr_char_threshold=10000)
        loader = PDFLoader(config=config)
        pages = loader.load(TEXT_NATIVE_PDF)
        # All pages should now use OCR (their native text is ~1000 chars < 10000).
        for page in pages:
            assert page.extraction_method is ExtractionMethod.OCR


# ---------------------------------------------------------------------------
# load_to_dict summary
# ---------------------------------------------------------------------------

class TestLoadToDict:
    """Tests for the summary-dict helper."""

    def test_native_pdf_summary(self, loader: PDFLoader) -> None:
        result = loader.load_to_dict(TEXT_NATIVE_PDF)
        assert result["filename"] == "text_native.pdf"
        assert result["pages"] == 3
        assert result["ocr_pages"] == 0
        assert result["total_characters"] > 0

    def test_scanned_pdf_summary(self, loader: PDFLoader) -> None:
        result = loader.load_to_dict(SCANNED_IMAGE_PDF)
        assert result["filename"] == "scanned_image.pdf"
        assert result["pages"] == 2
        assert result["ocr_pages"] == 2
        assert result["total_characters"] > 0

    def test_mixed_pdf_summary(self, loader: PDFLoader) -> None:
        result = loader.load_to_dict(MIXED_PDF)
        assert result["filename"] == "mixed.pdf"
        assert result["pages"] == 4
        assert result["ocr_pages"] == 2
        assert result["total_characters"] > 0
