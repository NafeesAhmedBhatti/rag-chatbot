"""
PDF ingestion: text extraction for native and scanned PDFs.

Design
------
``PDFLoader`` processes a PDF page-by-page. For each page it first
attempts native text extraction via PyMuPDF (``page.get_text()``).
If the extracted text is below ``ocr_char_threshold`` characters
(default 20) — indicating an image-only or scanned page — the page is
rendered to a bitmap and passed to OCR.

OCR engine selection (fail-safe chain):
    1. **Tesseract** (primary) — fastest, highest quality on the box.
    2. **RapidOCR** (fallback) — pure Python ONNX runtime, used only if
       the Tesseract binary is unavailable or throws an error.

The class is designed to handle large PDFs (200+ pages) efficiently:
PyMuPDF streams pages lazily, OCR is invoked only on pages that need
it, and images are rendered at a configurable DPI to balance quality
and memory.

Thread safety: each ``load()`` call opens its own ``fitz.Document``,
so multiple calls can run concurrently. The OCR engines are stateless
and safe to share.
"""

from __future__ import annotations

import io
import logging
import shutil
from pathlib import Path
from typing import Protocol

import fitz  # PyMuPDF

from rag.config import Settings, settings as default_settings
from rag.exceptions import IngestionError
from rag.models import ExtractionMethod, PageContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR engine protocol — allows swapping Tesseract / RapidOCR / mocks.
# ---------------------------------------------------------------------------

class OCREngine(Protocol):
    """Interface for an OCR engine."""

    def image_to_text(self, image_bytes: bytes, *, dpi: int, language: str) -> str:
        """Extract text from a PNG/JPEG image (bytes)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable engine name (for logging/metadata)."""
        ...


# ---------------------------------------------------------------------------
# Tesseract OCR engine
# ---------------------------------------------------------------------------

class TesseractEngine:
    """OCR via the system Tesseract binary (through pytesseract).

    Requires the ``tesseract-ocr`` system package and at least one
    language pack (e.g. ``tesseract-ocr-eng``).
    """

    def __init__(self) -> None:
        import pytesseract  # noqa: WPS433 — lazy import

        # Verify the binary exists on PATH.
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin is None:
            raise IngestionError(
                "Tesseract binary not found on PATH. "
                "Install with: apt-get install -y tesseract-ocr"
            )
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin
        self._pytesseract = pytesseract

    @property
    def name(self) -> str:
        return "tesseract"

    def image_to_text(self, image_bytes: bytes, *, dpi: int, language: str) -> str:
        from PIL import Image  # noqa: WPS433

        image = Image.open(io.BytesIO(image_bytes))
        return self._pytesseract.image_to_string(
            image,
            lang=language,
            config="--dpi {}".format(dpi),
        ).strip()


# ---------------------------------------------------------------------------
# RapidOCR fallback engine
# ---------------------------------------------------------------------------

class RapidOCREngine:
    """OCR via RapidOCR (ONNX runtime, no system binary needed).

    Downloads ONNX models on first use (~50 MB) and caches them.
    """

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433

        self._engine = RapidOCR()
        self._dpi = 300

    @property
    def name(self) -> str:
        return "rapidocr"

    def image_to_text(self, image_bytes: bytes, *, dpi: int, language: str) -> str:
        from PIL import Image  # noqa: WPS433

        image = Image.open(io.BytesIO(image_bytes))
        result, _elapsed = self._engine(image)
        if result is None:
            return ""
        # RapidOCR returns list of [bbox, text, confidence].
        lines = [item[1] for item in result]
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# OCR engine factory (Tesseract → RapidOCR fallback chain)
# ---------------------------------------------------------------------------

_ocr_engine: OCREngine | None = None
_ocr_init_attempted = False


def _get_ocr_engine() -> OCREngine | None:
    """Return an OCR engine, preferring Tesseract.

    Tries Tesseract first. If the binary is missing or initialization
    fails, falls back to RapidOCR. If both fail, returns ``None`` —
    callers should treat ``None`` as "OCR unavailable".
    """
    global _ocr_engine, _ocr_init_attempted

    if _ocr_init_attempted:
        return _ocr_engine

    _ocr_init_attempted = True

    # --- Try Tesseract ---
    try:
        _ocr_engine = TesseractEngine()
        logger.info("OCR engine initialized: tesseract")
        return _ocr_engine
    except (IngestionError, Exception) as exc:  # noqa: BLE001
        logger.warning("Tesseract unavailable (%s), trying RapidOCR fallback", exc)

    # --- Try RapidOCR ---
    try:
        _ocr_engine = RapidOCREngine()
        logger.info("OCR engine initialized: rapidocr (fallback)")
        return _ocr_engine
    except Exception as exc:  # noqa: BLE001
        logger.error("RapidOCR also unavailable (%s); OCR disabled", exc)
        _ocr_engine = None
        return None


def reset_ocr_engine() -> None:
    """Reset the cached OCR engine (for testing)."""
    global _ocr_engine, _ocr_init_attempted
    _ocr_engine = None
    _ocr_init_attempted = False


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

class PDFLoader:
    """Extract text from PDF files with OCR fallback.

    Parameters
    ----------
    config:
        Application settings. Defaults to the module-level singleton.
    ocr_engine:
        Injected OCR engine (for testing). If ``None``, uses the
        global Tesseract→RapidOCR fallback chain.
    """

    def __init__(
        self,
        config: Settings | None = None,
        ocr_engine: OCREngine | None = None,
    ) -> None:
        self.config = config or default_settings
        self._ocr_engine_override = ocr_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, pdf_path: str | Path) -> list[PageContent]:
        """Extract text from all pages of a PDF.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.

        Returns
        -------
        list[PageContent]
            One entry per page, in page order (1-based page numbers).

        Raises
        ------
        IngestionError
            If the file cannot be opened, is not a PDF, or is corrupted.
        """
        path = Path(pdf_path)
        self._validate_path(path)

        logger.info("Loading PDF: %s", path.name)

        try:
            document = fitz.open(str(path))
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(
                "Failed to open PDF '{}': {}".format(path.name, exc)
            ) from exc

        if document.is_encrypted:
            document.close()
            raise IngestionError(
                "PDF '{}' is encrypted; password-protected PDFs are not supported".format(
                    path.name
                )
            )

        total_pages = document.page_count
        logger.info(
            "PDF '%s' has %d page(s); OCR threshold=%d chars",
            path.name,
            total_pages,
            self.config.ocr_char_threshold,
        )

        pages: list[PageContent] = []
        ocr_page_count = 0

        try:
            for page_index in range(total_pages):
                page_num = page_index + 1  # 1-based for humans
                page = document.load_page(page_index)
                content = self._extract_page(page, page_num)
                if content.extraction_method is ExtractionMethod.OCR:
                    ocr_page_count += 1
                pages.append(content)
        finally:
            document.close()

        logger.info(
            "PDF '%s' processed: %d pages (%d via OCR)",
            path.name,
            len(pages),
            ocr_page_count,
        )
        return pages

    def load_to_dict(self, pdf_path: str | Path) -> dict:
        """Load a PDF and return a summary dict (for API responses).

        Includes filename, page count, OCR page count, and total chars.
        """
        path = Path(pdf_path)
        pages = self.load(path)
        ocr_count = sum(
            1 for p in pages if p.extraction_method is ExtractionMethod.OCR
        )
        total_chars = sum(p.char_count for p in pages)
        return {
            "filename": path.name,
            "pages": len(pages),
            "ocr_pages": ocr_count,
            "total_characters": total_chars,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_path(self, path: Path) -> None:
        """Check that the path exists and looks like a PDF."""
        if not path.exists():
            raise IngestionError("File not found: {}".format(path))
        if not path.is_file():
            raise IngestionError("Path is not a file: {}".format(path))
        if path.suffix.lower() != ".pdf":
            raise IngestionError(
                "Expected a .pdf file, got '{}': {}".format(path.suffix, path)
            )

    def _extract_page(
        self,
        page: fitz.Page,
        page_num: int,
    ) -> PageContent:
        """Extract text from a single page.

        Strategy:
            1. Try native extraction (``get_text``).
            2. If text length < threshold, fall back to OCR.
        """
        native_text = page.get_text("text").strip()

        if len(native_text) >= self.config.ocr_char_threshold:
            logger.debug(
                "Page %d: native text (%d chars)", page_num, len(native_text)
            )
            return PageContent(
                page=page_num,
                text=native_text,
                extraction_method=ExtractionMethod.NATIVE,
            )

        # --- OCR fallback ---
        logger.debug(
            "Page %d: only %d native chars (threshold %d) — attempting OCR",
            page_num,
            len(native_text),
            self.config.ocr_char_threshold,
        )
        ocr_text = self._ocr_page(page)

        # Prefer OCR text if it produced more content; otherwise keep
        # whatever native text we got (better than nothing).
        if len(ocr_text) > len(native_text):
            return PageContent(
                page=page_num,
                text=ocr_text,
                extraction_method=ExtractionMethod.OCR,
            )

        # OCR didn't help — return the (short) native text, marked native.
        logger.debug(
            "Page %d: OCR did not improve result (%d→%d chars), keeping native",
            page_num,
            len(native_text),
            len(ocr_text),
        )
        return PageContent(
            page=page_num,
            text=native_text,
            extraction_method=ExtractionMethod.NATIVE,
        )

    def _ocr_page(self, page: fitz.Page) -> str:
        """Render a page to an image and run OCR on it.

        Returns an empty string if no OCR engine is available.
        """
        engine = self._resolve_ocr_engine()
        if engine is None:
            logger.warning("No OCR engine available; returning empty text for page")
            return ""

        # Render the page to a PNG image at the configured DPI.
        zoom = self.config.ocr_dpi / 72.0  # 72 DPI is PDF default
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")

        try:
            return engine.image_to_text(
                image_bytes,
                dpi=self.config.ocr_dpi,
                language=self.config.ocr_language,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR failed on page: %s", exc)
            return ""

    def _resolve_ocr_engine(self) -> OCREngine | None:
        """Return the OCR engine to use.

        Priority: constructor-injected engine → global fallback chain.
        """
        if self._ocr_engine_override is not None:
            return self._ocr_engine_override
        return _get_ocr_engine()
