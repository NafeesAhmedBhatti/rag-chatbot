"""
Generate test PDFs for Phase 2 verification.

Creates three PDFs in data/sample/:
    1. text_native.pdf  — native selectable text (tests native extraction)
    2. scanned_image.pdf — image-only pages (tests OCR path)
    3. mixed.pdf        — mix of text pages and image pages (tests heuristic)

Run:  python scripts/generate_test_pdfs.py
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

# --- Paths ---
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared text content — multi-paragraph, enough for native extraction tests.
# ---------------------------------------------------------------------------

LOREM = (
    "Retrieval-Augmented Generation (RAG) combines information retrieval with "
    "text generation to produce answers grounded in source documents. "
    "The system first retrieves relevant passages from a knowledge base, "
    "then feeds them as context to a language model."
)

PARAGRAPHS = [
    "Artificial intelligence has transformed how we interact with documents. "
    "Natural language processing enables machines to understand human language "
    "with remarkable accuracy. Machine learning models learn patterns from "
    "large datasets to make predictions about new data.",

    "The quick brown fox jumps over the lazy dog. This pangram contains every "
    "letter of the English alphabet. It is commonly used for testing font "
    "rendering and keyboard layouts across different systems.",

    "Climate change poses significant challenges to global agriculture. "
    "Rising temperatures affect crop yields, while changing precipitation "
    "patterns lead to droughts in some regions and floods in others. "
    "Farmers must adapt their practices to ensure food security.",

    "The theory of relativity, proposed by Albert Einstein, fundamentally "
    "changed our understanding of space and time. Special relativity describes "
    "the physics of moving bodies in the absence of gravitational fields. "
    "General relativity extends this to include gravity as curvature of spacetime.",
]


def _render_text_to_image(text: str, width: int = 1240, height: int = 1754) -> bytes:
    """Render text as a PNG image (simulating a scanned page).

    Uses a monospace font to ensure OCR can read it reliably.
    Returns PNG bytes.
    """
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # Use the default PIL font (always available); scale via ImageFont.
    # Try a TrueType font first, fall back to default.
    font_size = 28
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Word-wrap the text to fit the image width.
    max_chars = 75  # approx for monospace at this size
    wrapped_lines: list[str] = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=max_chars) or [""]
        wrapped_lines.extend(wrapped)
        wrapped_lines.append("")  # blank line between paragraphs

    y = 80
    line_height = font_size + 14
    for line in wrapped_lines:
        draw.text((80, y), line, fill="black", font=font)
        y += line_height
        if y > height - 80:
            break

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Native text PDF
# ---------------------------------------------------------------------------

def create_text_native_pdf(output_path: Path) -> None:
    """Create a 3-page PDF with native selectable text."""
    doc = fitz.open()

    for page_num in range(1, 4):
        page = doc.new_page(width=612, height=792)  # US Letter
        rect = fitz.Rect(72, 72, 540, 720)  # margins
        body = f"Page {page_num}\n\n" + "\n\n".join(PARAGRAPHS)
        page.insert_textbox(
            rect,
            body,
            fontsize=12,
            fontname="helv",
            color=(0, 0, 0),
        )

    doc.save(str(output_path))
    doc.close()


# ---------------------------------------------------------------------------
# 2. Image-only (scanned) PDF
# ---------------------------------------------------------------------------

def create_scanned_image_pdf(output_path: Path) -> None:
    """Create a 2-page PDF where each page is a full-page image (no native text).

    This simulates a scanned document — native extraction yields nothing,
    forcing the OCR path.
    """
    doc = fitz.open()

    for page_num in range(1, 3):
        text = f"Scanned Page {page_num}\n\n" + "\n".join(PARAGRAPHS)
        image_bytes = _render_text_to_image(text)

        # Create a page sized to match the image aspect ratio.
        image = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = image.size
        page = doc.new_page(width=612, height=792)
        # Insert image to fill the page.
        page.insert_image(fitz.Rect(0, 0, 612, 792), stream=image_bytes)

    doc.save(str(output_path))
    doc.close()


# ---------------------------------------------------------------------------
# 3. Mixed PDF (some text pages, some image pages)
# ---------------------------------------------------------------------------

def create_mixed_pdf(output_path: Path) -> None:
    """Create a 4-page PDF: pages 1-2 native text, pages 3-4 image-only.

    This tests the per-page heuristic: the loader should use native
    extraction for pages 1-2 and OCR for pages 3-4.
    """
    doc = fitz.open()

    # Pages 1-2: native text
    for page_num in range(1, 3):
        page = doc.new_page(width=612, height=792)
        rect = fitz.Rect(72, 72, 540, 720)
        body = f"Mixed Document Page {page_num}\n\n" + "\n\n".join(PARAGRAPHS)
        page.insert_textbox(
            rect,
            body,
            fontsize=12,
            fontname="helv",
            color=(0, 0, 0),
        )

    # Pages 3-4: image-only (no native text)
    for page_num in range(3, 5):
        text = f"Mixed Document Image Page {page_num}\n\n" + "\n".join(PARAGRAPHS)
        image_bytes = _render_text_to_image(text)
        page = doc.new_page(width=612, height=792)
        page.insert_image(fitz.Rect(0, 0, 612, 792), stream=image_bytes)

    doc.save(str(output_path))
    doc.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating test PDFs in", SAMPLE_DIR)

    native_pdf = SAMPLE_DIR / "text_native.pdf"
    create_text_native_pdf(native_pdf)
    print(f"  ✓ {native_pdf.name}")

    scanned_pdf = SAMPLE_DIR / "scanned_image.pdf"
    create_scanned_image_pdf(scanned_pdf)
    print(f"  ✓ {scanned_pdf.name}")

    mixed_pdf = SAMPLE_DIR / "mixed.pdf"
    create_mixed_pdf(mixed_pdf)
    print(f"  ✓ {mixed_pdf.name}")

    print("Done. {} PDFs created.".format(3))


if __name__ == "__main__":
    main()
