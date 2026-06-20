"""
Unit tests for rag.config.Settings.

Verifies that configuration loads correctly with defaults and that
environment-variable overrides work as expected.
"""

from __future__ import annotations

import pytest

from rag.config import Settings


class TestDefaults:
    """Default values when no env vars are set."""

    def test_default_chunk_size(self) -> None:
        s = Settings()
        assert s.chunk_size == 512

    def test_default_chunk_overlap(self) -> None:
        s = Settings()
        assert s.chunk_overlap == 64

    def test_default_top_k(self) -> None:
        s = Settings()
        assert s.top_k == 5

    def test_default_score_threshold(self) -> None:
        s = Settings()
        assert s.score_threshold == 0.3

    def test_default_embedding_model(self) -> None:
        s = Settings()
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_default_llm_model(self) -> None:
        s = Settings()
        assert s.llm_model == "drytis/MiniMax-M3"


class TestEnvOverride:
    """Environment variables override defaults."""

    def test_chunk_size_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "256")
        s = Settings()
        assert s.chunk_size == 256

    def test_top_k_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOP_K", "10")
        s = Settings()
        assert s.top_k == 10

    def test_app_debug_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_DEBUG", "true")
        s = Settings()
        assert s.app_debug is True


class TestDerivedProperties:
    """Computed properties on Settings."""

    def test_faiss_index_path(self) -> None:
        s = Settings()
        assert s.faiss_index_path.name == "faiss.index"

    def test_metadata_path(self) -> None:
        s = Settings()
        assert s.metadata_path.name == "metadata.json"

    def test_is_production_false_by_default(self) -> None:
        s = Settings()
        assert s.is_production is False


class TestEnsureDirectories:
    """Directory creation helper."""

    def test_ensure_directories_creates_paths(self, tmp_path) -> None:
        s = Settings(
            data_dir=tmp_path / "data",
            pdf_dir=tmp_path / "data/pdfs",
            index_dir=tmp_path / "data/index",
            sample_dir=tmp_path / "data/sample",
        )
        s.ensure_directories()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "data/pdfs").is_dir()
        assert (tmp_path / "data/index").is_dir()
        assert (tmp_path / "data/sample").is_dir()
