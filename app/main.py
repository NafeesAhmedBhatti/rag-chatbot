"""
FastAPI application factory and lifespan management.

Creates the FastAPI app, configures logging, ensures data directories
exist, mounts the static frontend, and includes the API router.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from rag.config import settings
from rag.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Absolute path to the static assets directory (sibling of this file).
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle.

    Startup:
        1. Configure structured logging.
        2. Ensure all data directories exist.
        3. (Later phases) Load the embedding model and FAISS index
           into ``app.state``.

    Shutdown:
        1. (Later phases) Flush the FAISS index to disk.
    """
    log_level = "DEBUG" if settings.app_debug else "INFO"
    setup_logging(level=log_level)
    logger.info("Starting RAG Chatbot (env=%s, debug=%s)", settings.app_env, settings.app_debug)

    settings.ensure_directories()
    logger.info("Data directories ready: %s", settings.data_dir)

    app.state.ready = True
    logger.info("Application ready — serving RAG Chatbot UI at /")

    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("Shutting down RAG Chatbot")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="RAG Chatbot",
        description=(
            "Retrieval-Augmented Generation chatbot with PDF ingestion, OCR "
            "support, semantic search, and cited answer generation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # API routes under /api/*
    app.include_router(api_router)

    # Serve static frontend assets (CSS, JS).
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Root: serve the chat UI (index.html) at "/".
    # Using a route instead of mounting the whole directory so that we
    # retain control over the root path and can add a proper 404 later.
    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """Serve the single-page chat UI."""
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """Serve favicon."""
        return FileResponse(
            str(_STATIC_DIR / "favicon.svg"),
            media_type="image/svg+xml",
        )

    return app


# The module-level ``app`` is what Uvicorn imports (``app.main:app``).
app = create_app()
