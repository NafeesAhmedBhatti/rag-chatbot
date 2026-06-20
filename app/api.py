"""
API route handlers for the RAG Chatbot.

Endpoints:
  - GET  /api/health   — liveness probe.
  - GET  /api/stats    — corpus + index statistics.
  - POST /api/chat     — ask a question, get cited answer + chunks.
  - POST /api/ingest   — upload a PDF, ingest it into the index.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.config import settings
from rag.exceptions import RagError
from rag.logging_config import utc_now_iso

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    app_env: str
    version: str
    timestamp: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    index_size_mb: float
    embedding_model: str
    vector_dimensions: int
    llm_model: str
    last_updated: str


class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    pages: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class CitationItem(BaseModel):
    filename: str
    page: int
    raw: str


class ChunkItem(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page: int
    score: float
    token_count: int = 0


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
    retrieved_chunks: list[ChunkItem]
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    model: str
    top_k: int
    total_chunks_found: int


class IngestResponse(BaseModel):
    status: str
    filename: str
    pages: int
    chunks: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_PROCESS_START = time.monotonic()
_APP_VERSION = "0.1.0"

# Lazy-initialized pipeline components (shared across requests).
_embedder = None
_store = None
_retriever = None
_generator = None


def _get_pipeline():
    """Lazily initialize and cache the RAG pipeline components."""
    global _embedder, _store, _retriever, _generator

    if _embedder is None:
        logger.info("Initializing RAG pipeline components")
        from rag.embeddings.embedder import Embedder
        from rag.vectorstore.faiss_store import FAISSStore
        from rag.retrieval.retriever import Retriever
        from rag.generation.generator import Generator

        _embedder = Embedder(config=settings)
        _store = FAISSStore(config=settings, dimension=_embedder.dimension)
        if not _store.load():
            _store.initialize()
        _retriever = Retriever(_embedder, _store, config=settings)
        _generator = Generator(config=settings)
        logger.info(
            "RAG pipeline ready (index size: %d)", _store.size
        )

    return _embedder, _store, _retriever, _generator


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health."""
    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        version=_APP_VERSION,
        timestamp=utc_now_iso(),
        uptime_seconds=round(time.monotonic() - _PROCESS_START, 3),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Return corpus and index statistics."""
    embedder, store, _, _ = _get_pipeline()

    stats_dict = store.get_stats()
    return StatsResponse(
        total_documents=stats_dict.get("total_documents", 0),
        total_chunks=stats_dict.get("total_chunks", 0),
        index_size_mb=stats_dict.get("index_size_mb", 0.0),
        embedding_model=settings.embedding_model,
        vector_dimensions=stats_dict.get("dimension", 384),
        llm_model=settings.llm_model,
        last_updated=utc_now_iso(),
    )


@router.get("/documents", response_model=DocumentsResponse)
async def documents() -> DocumentsResponse:
    """Return the list of indexed documents with per-file stats."""
    _, store, _, _ = _get_pipeline()

    doc_map: dict[str, dict] = {}
    for meta in store._metadata.values():
        sf = meta.get("source_file", "unknown")
        if sf not in doc_map:
            doc_map[sf] = {"filename": sf, "chunks": 0, "pages": set()}
        doc_map[sf]["chunks"] += 1
        doc_map[sf]["pages"].add(meta.get("page", 0))

    docs = sorted(
        [
            DocumentInfo(
                filename=d["filename"],
                chunks=d["chunks"],
                pages=len(d["pages"]),
            )
            for d in doc_map.values()
        ],
        key=lambda x: x.filename,
    )
    return DocumentsResponse(documents=docs, total=len(docs))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Answer a question using retrieval-augmented generation.

    Flow: embed query → FAISS search → score filter → LLM generation → citations.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    _, store, retriever, generator = _get_pipeline()

    if store.size == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents have been ingested yet. Upload PDFs via POST /api/ingest first.",
        )

    logger.info("Chat query: %s", question[:120])

    try:
        # Step 1: Retrieve.
        retrieval_result = retriever.retrieve(question)

        # Step 2: Generate.
        gen_result = generator.generate(retrieval_result)

        # Build chunk items.
        chunk_items = [
            ChunkItem(
                chunk_id=c.chunk_id,
                text=c.text,
                source_file=c.source_file,
                page=c.page,
                score=c.score,
                token_count=c.metadata.get("token_count", 0),
            )
            for c in retrieval_result.chunks
        ]

        # Build citation items.
        citation_items = [
            CitationItem(filename=cite.filename, page=cite.page, raw=cite.raw)
            for cite in gen_result.citations
        ]

        total_latency = retrieval_result.latency_ms + gen_result.latency_ms

        return ChatResponse(
            answer=gen_result.answer,
            citations=citation_items,
            retrieved_chunks=chunk_items,
            latency_ms=round(total_latency, 2),
            retrieval_latency_ms=round(retrieval_result.latency_ms, 2),
            generation_latency_ms=round(gen_result.latency_ms, 2),
            model=gen_result.model,
            top_k=settings.top_k,
            total_chunks_found=retrieval_result.total_found,
        )

    except HTTPException:
        raise
    except RagError as exc:
        logger.error("RAG pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error in chat: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a single PDF file.

    The file is saved to the PDF directory, then processed through the
    full pipeline: extraction → cleaning → chunking → embedding → storage.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save the uploaded file.
    file_path = settings.pdf_dir / file.filename
    logger.info("Ingesting uploaded file: %s", file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    # Run the pipeline on the single file.
    from rag.ingestion.pdf_loader import PDFLoader
    from rag.processing.cleaner import LanguageDetector, TextCleaner
    from rag.processing.chunker import Chunker

    embedder, store, _, _ = _get_pipeline()

    loader = PDFLoader(config=settings)
    cleaner = TextCleaner()
    detector = LanguageDetector()
    chunker = Chunker(config=settings)

    t0 = time.perf_counter()

    try:
        pages = loader.load(file_path)
        logger.info("Extracted %d pages from %s", len(pages), file.filename)

        all_chunks = []
        for page in pages:
            cleaned = cleaner.clean(page.text)
            if not cleaned.strip():
                continue
            lang_result = detector.detect(cleaned)
            page_chunks = chunker.chunk(
                text=cleaned,
                source_file=file.filename,
                page=page.page,
                language=lang_result.code,
            )
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the PDF. It may be corrupted or empty.",
            )

        # Idempotent: remove old chunks for this file first.
        store.remove_by_source(file.filename)

        # Embed and store.
        chunk_texts = [c.text for c in all_chunks]
        vectors = embedder.encode(chunk_texts)

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
        store.save()

        elapsed = round(time.perf_counter() - t0, 2)
        logger.info(
            "Ingested %s: %d pages, %d chunks in %.1fs",
            file.filename, len(pages), len(all_chunks), elapsed,
        )

        return IngestResponse(
            status="ok",
            filename=file.filename,
            pages=len(pages),
            chunks=len(all_chunks),
            elapsed_s=elapsed,
        )

    except HTTPException:
        raise
    except RagError as exc:
        logger.error("Ingestion error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error in ingest: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
