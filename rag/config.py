"""
Central configuration for the RAG Chatbot.

All tunable parameters live here as typed fields on the ``Settings``
dataclass. Values are read from environment variables (with sensible
defaults) via ``pydantic-settings``. The platform's env-key system
writes these to ``/workspace/.env`` at deploy time.

Usage::

    from rag.config import settings
    print(settings.chunk_size)

Never import ``os.getenv`` directly in application code - go through
``settings`` so that all configuration is centralized and type-checked.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Path constants - the canonical locations for data inside the container.
# These are defaults; they can be overridden via env vars below.
# ---------------------------------------------------------------------------
WORKSPACE_DIR = Path("/workspace")
DEFAULT_DATA_DIR = WORKSPACE_DIR / "data"
DEFAULT_PDF_DIR = DEFAULT_DATA_DIR / "pdfs"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "index"
DEFAULT_SAMPLE_DIR = DEFAULT_DATA_DIR / "sample"


class Settings(BaseSettings):
    """Typed application configuration.

    Every field maps to an uppercase environment variable of the same
    name (case-insensitive). Secrets (``openai_api_key``) are loaded
    from the ``.env`` file that the platform generates.
    """

    model_config = SettingsConfigDict(
        env_file=str(WORKSPACE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: str = "development"
    app_debug: bool = False

    # --- LLM (answer generation) ---
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "drytis/MiniMax-M3"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 30

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    # --- Chunking ---
    chunk_size: int = 512          # max tokens per chunk
    chunk_overlap: int = 64        # overlap tokens between adjacent chunks

    # --- Retrieval ---
    top_k: int = 5                 # number of chunks to retrieve
    score_threshold: float = 0.3   # min cosine similarity to include

    # --- OCR ---
    ocr_char_threshold: int = 20   # pages with fewer chars trigger OCR
    ocr_dpi: int = 300             # render DPI for image-based pages
    ocr_language: str = "eng"

    # --- Paths ---
    data_dir: Path = DEFAULT_DATA_DIR
    pdf_dir: Path = DEFAULT_PDF_DIR
    index_dir: Path = DEFAULT_INDEX_DIR
    sample_dir: Path = DEFAULT_SAMPLE_DIR

    # --- Derived paths (read-only properties) ---
    @property
    def faiss_index_path(self) -> Path:
        """Path to the persisted FAISS binary index."""
        return self.index_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        """Path to the chunk metadata sidecar JSON."""
        return self.index_dir / "metadata.json"

    @property
    def is_production(self) -> bool:
        """True when running in the production deployment."""
        return self.app_env == "production"

    def ensure_directories(self) -> None:
        """Create all data directories if they do not exist.

        Called once at application startup so that ingestion and
        persistence never fail on a missing path.
        """
        for path in (
            self.data_dir,
            self.pdf_dir,
            self.index_dir,
            self.sample_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton ``Settings`` instance.

    Cached for the lifetime of the process so that env-var lookups
    happen exactly once. Call ``get_settings.cache_clear()`` only in
    tests that need to swap configuration.
    """
    return Settings()  # type: ignore[call-arg]


# Module-level singleton for convenient import in modules that do not
# need the test-swap capability.
settings = get_settings()
