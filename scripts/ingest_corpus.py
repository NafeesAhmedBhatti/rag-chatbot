#!/usr/bin/env python3
"""
Batch PDF ingestion script.

Processes all PDFs in a directory through the full RAG pipeline:
    1. PDFLoader    — extract text (native + OCR fallback).
    2. TextCleaner  — normalize, clean, de-hyphenate.
    3. LanguageDetector — detect language per page.
    4. Chunker      — split into token-bounded chunks with metadata.
    5. Embedder     — generate BGE embeddings (batch).
    6. FAISSStore   — store vectors + metadata (idempotent re-ingestion).

Usage:
    python3 scripts/ingest_corpus.py --dir /workspace/data/pdfs
    python3 scripts/ingest_corpus.py --dir /workspace/data/pdfs --clear
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure /workspace is on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import settings  # noqa: E402
from rag.exceptions import RagError  # noqa: E402
from rag.ingestion.pdf_loader import PDFLoader  # noqa: E402
from rag.processing.cleaner import LanguageDetector, TextCleaner  # noqa: E402
from rag.processing.chunker import Chunker  # noqa: E402
from rag.embeddings.embedder import Embedder  # noqa: E402
from rag.vectorstore.faiss_store import FAISSStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def ingest_directory(pdf_dir: Path, clear: bool = False) -> dict:
    """Ingest all PDFs in ``pdf_dir`` into the FAISS index.

    Parameters
    ----------
    pdf_dir:
        Directory containing .pdf files.
    clear:
        If True, remove existing index before ingesting.

    Returns
    -------
    dict
        Summary with total_files, total_pages, total_chunks, elapsed_s.
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", pdf_dir)
        return {"total_files": 0, "total_pages": 0, "total_chunks": 0, "elapsed_s": 0}

    logger.info("Found %d PDF(s) to ingest", len(pdfs))

    # Initialize components.
    loader = PDFLoader(config=settings)
    cleaner = TextCleaner()
    detector = LanguageDetector()
    chunker = Chunker(config=settings)
    embedder = Embedder(config=settings)
    store = FAISSStore(config=settings, dimension=embedder.dimension)

    # Clear or load existing index.
    if clear:
        logger.info("Clearing existing index")
        store.initialize()
    else:
        if not store.load():
            store.initialize()

    t_start = time.perf_counter()
    total_pages = 0
    total_chunks = 0

    for pdf_path in pdfs:
        file_t0 = time.perf_counter()
        logger.info("Processing: %s", pdf_path.name)

        try:
            # Step 1: Load PDF pages.
            pages = loader.load(pdf_path)
            logger.info("  Extracted %d pages", len(pages))

            # Step 2-3: Clean + detect language per page, then chunk.
            all_chunks = []
            for page in pages:
                cleaned = cleaner.clean(page.text)
                if not cleaned.strip():
                    continue
                lang_result = detector.detect(cleaned)
                page_chunks = chunker.chunk(
                    text=cleaned,
                    source_file=pdf_path.name,
                    page=page.page,
                    language=lang_result.code,
                )
                all_chunks.extend(page_chunks)

            total_pages += len(pages)
            logger.info("  Created %d chunks", len(all_chunks))

            if not all_chunks:
                logger.warning("  No text extracted from %s", pdf_path.name)
                continue

            # Step 4: Idempotent re-ingestion — remove old chunks for this file.
            store.remove_by_source(pdf_path.name)

            # Step 5: Embed.
            chunk_texts = [c.text for c in all_chunks]
            vectors = embedder.encode(chunk_texts)

            # Step 6: Store.
            metadata_list = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source_file": c.source_file,
                    "page": c.page,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "chunk_index": c.chunk_index,
                    "language": c.language,
                    "token_count": c.token_count,
                }
                for c in all_chunks
            ]
            store.add(vectors, metadata_list)
            total_chunks += len(all_chunks)

            elapsed = time.perf_counter() - file_t0
            logger.info(
                "  Done: %d chunks in %.1fs", len(all_chunks), elapsed
            )

        except RagError as exc:
            logger.error("  FAILED: %s", exc)
            continue

    # Persist.
    store.save()
    elapsed_s = round(time.perf_counter() - t_start, 2)

    summary = {
        "total_files": len(pdfs),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "elapsed_s": elapsed_s,
    }
    logger.info(
        "Ingestion complete: %d files, %d pages, %d chunks in %.1fs",
        summary["total_files"],
        summary["total_pages"],
        summary["total_chunks"],
        elapsed_s,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch PDF ingestion")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(settings.pdf_dir),
        help="Directory containing PDFs",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before ingesting",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.dir)
    if not pdf_dir.exists():
        logger.error("Directory not found: %s", pdf_dir)
        sys.exit(1)

    summary = ingest_directory(pdf_dir, clear=args.clear)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
