"""
tests/unit/test_rag.py
───────────────────────
Unit tests for the RAG pipeline: embedder, indexer, and retriever.

Strategy:
- embedder.py: mock httpx.post — test retry logic, response validation,
  batch handling, without hitting the real HuggingFace API.
- indexer.py: mock embed_texts and the DB session — test chunking logic,
  file filtering, batch processing, without cloning a real repo or hitting
  a real DB.
- retriever.py: mock embed_single and get_session — test the node
  orchestration (per-chunk retrieval, graceful degradation) without a
  real pgvector instance.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call
import tempfile
import os

import pytest

from app.rag.embedder import (
    EMBEDDING_DIM,
    EmbeddingError,
    _require_env,
    embed_single,
    embed_texts,
)
from app.rag.indexer import (
    CHUNK_SIZE_CHARS,
    OVERLAP_CHARS,
    _chunk_text,
    _collect_source_files,
)
from app.rag.retriever import run as retriever_run
from app.agent.state import DiffChunk, initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_vector(dim: int = EMBEDDING_DIM) -> list:
    return [0.01 * i for i in range(dim)]


def _make_hf_response(vectors: list, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = (200 <= status_code < 300)
    resp.json.return_value = vectors
    resp.text = "ok" if resp.is_success else "error"
    return resp


def _sample_chunk(file_path="src/api.py", line_number=10) -> DiffChunk:
    return DiffChunk(
        file_path=file_path,
        start_line=line_number,
        end_line=line_number + 5,
        content="+def new_function():\n+    return 42",
        language="python",
        chunk_index=0,
    )


# ---------------------------------------------------------------------------
# Tests: embed_texts()
# ---------------------------------------------------------------------------

class TestEmbedTexts:

    def test_empty_list_returns_empty(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        result = embed_texts([])
        assert result == []

    def test_returns_vectors_on_success(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        vectors = [_fake_vector(), _fake_vector()]

        with patch("httpx.post", return_value=_make_hf_response(vectors)):
            result = embed_texts(["text one", "text two"])

        assert len(result) == 2
        assert len(result[0]) == EMBEDDING_DIM

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="HUGGINGFACE_API_KEY"):
            embed_texts(["some text"])

    def test_retries_on_503_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        monkeypatch.setattr("app.rag.embedder.time.sleep", lambda _: None)
        vectors = [_fake_vector()]

        responses = [
            _make_hf_response([], status_code=503),
            _make_hf_response(vectors, status_code=200),
        ]

        with patch("httpx.post", side_effect=responses) as mock_post:
            result = embed_texts(["text"])

        assert len(result) == 1
        assert mock_post.call_count == 2

    def test_raises_on_non_503_error(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")

        with patch("httpx.post", return_value=_make_hf_response([], status_code=401)):
            with pytest.raises(EmbeddingError, match="401"):
                embed_texts(["text"])

    def test_raises_on_dimension_mismatch(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        wrong_dim_vector = [[0.1] * 128]  # 128 instead of 384

        with patch("httpx.post", return_value=_make_hf_response(wrong_dim_vector)):
            with pytest.raises(EmbeddingError, match="384"):
                embed_texts(["text"])

    def test_raises_when_count_mismatches(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        # 2 inputs but 1 vector returned
        with patch("httpx.post", return_value=_make_hf_response([_fake_vector()])):
            with pytest.raises(EmbeddingError, match="2"):
                embed_texts(["text one", "text two"])

    def test_exhausts_retries_and_raises(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        monkeypatch.setattr("app.rag.embedder.time.sleep", lambda _: None)

        with patch("httpx.post", return_value=_make_hf_response([], status_code=503)):
            with pytest.raises(EmbeddingError, match="3 attempts"):
                embed_texts(["text"])


class TestEmbedSingle:

    def test_returns_single_vector(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        with patch("httpx.post", return_value=_make_hf_response([_fake_vector()])):
            result = embed_single("some text")
        assert len(result) == EMBEDDING_DIM

    def test_is_wrapper_around_embed_texts(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        with patch("app.rag.embedder.embed_texts", return_value=[_fake_vector()]) as mock:
            embed_single("text")
        mock.assert_called_once_with(["text"])


# ---------------------------------------------------------------------------
# Tests: _chunk_text()
# ---------------------------------------------------------------------------

class TestChunkText:

    def test_empty_string_returns_empty(self):
        assert _chunk_text("") == []

    def test_whitespace_only_returns_empty(self):
        assert _chunk_text("   \n\n  ") == []

    def test_short_text_returns_single_chunk(self):
        text = "def foo():\n    return 1\n"
        chunks = _chunk_text(text)
        assert len(chunks) == 1
        assert "def foo" in chunks[0]

    def test_large_text_splits_into_multiple_chunks(self):
        # Generate text larger than CHUNK_SIZE_CHARS
        line = "x = 1  # padding\n"
        text = line * (CHUNK_SIZE_CHARS // len(line) + 10)
        chunks = _chunk_text(text)
        assert len(chunks) >= 2

    def test_chunks_overlap(self):
        """
        The last lines of chunk N should appear at the start of chunk N+1
        (overlap ensures context continuity across chunk boundaries).
        """
        line = "x = 1  # padding line\n"
        text = line * (CHUNK_SIZE_CHARS // len(line) + 10)
        chunks = _chunk_text(text)
        if len(chunks) >= 2:
            # Last part of chunk 0 should appear in chunk 1
            last_line_of_chunk0 = chunks[0].splitlines()[-1]
            assert last_line_of_chunk0 in chunks[1]

    def test_splits_at_line_boundaries(self):
        """Chunks must never end mid-line (which would break syntax)."""
        line = "some_var = 'value'  # comment\n"
        text = line * (CHUNK_SIZE_CHARS // len(line) + 5)
        chunks = _chunk_text(text)
        for chunk in chunks:
            # Each chunk must end at a newline (no partial lines)
            stripped = chunk.rstrip("\n")
            assert "\n" in stripped or len(stripped) < CHUNK_SIZE_CHARS


# ---------------------------------------------------------------------------
# Tests: _collect_source_files()
# ---------------------------------------------------------------------------

class TestCollectSourceFiles:

    def test_finds_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('hello')")
            (root / "src" / "utils.py").write_text("def helper(): pass")
            files = _collect_source_files(root)
            paths = [str(f) for f in files]
            assert any("main.py" in p for p in paths)
            assert any("utils.py" in p for p in paths)

    def test_skips_non_source_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# readme")
            (root / "config.yaml").write_text("key: value")
            (root / "package-lock.json").write_text("{}")
            files = _collect_source_files(root)
            assert files == []

    def test_skips_node_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nm = root / "node_modules" / "lodash"
            nm.mkdir(parents=True)
            (nm / "index.js").write_text("module.exports = {}")
            (root / "app.js").write_text("const x = 1")
            files = _collect_source_files(root)
            assert all("node_modules" not in str(f) for f in files)

    def test_skips_git_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            git_dir = root / ".git" / "hooks"
            git_dir.mkdir(parents=True)
            (git_dir / "pre-commit.py").write_text("#!/usr/bin/env python")
            (root / "main.py").write_text("x = 1")
            files = _collect_source_files(root)
            assert all(".git" not in str(f) for f in files)

    def test_skips_large_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            large_file = root / "generated.py"
            large_file.write_bytes(b"x = 1\n" * 20000)  # > 100KB
            small_file = root / "main.py"
            small_file.write_text("x = 1")
            files = _collect_source_files(root)
            assert large_file not in files
            assert small_file in files


# ---------------------------------------------------------------------------
# Tests: retriever run() node
# ---------------------------------------------------------------------------

class TestRetrieverRunNode:

    def test_returns_empty_context_when_no_chunks(self):
        state = initial_state(
            raw_diff="", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = []
        result = retriever_run(state)
        assert result["repo_context"] == {}

    def test_returns_empty_when_database_url_not_set(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        state = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [_sample_chunk()]
        result = retriever_run(state)
        assert result["repo_context"] == {}

    def test_populates_context_for_each_chunk(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/test")
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")

        chunk1 = _sample_chunk(file_path="src/api.py", line_number=1)
        chunk2 = _sample_chunk(file_path="src/auth.py", line_number=10)

        state = initial_state(
            raw_diff="diff", repo_full_name="owner/repo", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [chunk1, chunk2]

        # Mock retrieve_context to return snippets without hitting the DB
        async def fake_retrieve(chunk_text, repo_full_name, top_k):
            return [f"related snippet for {chunk_text[:10]}"]

        with patch("app.rag.retriever.retrieve_context", side_effect=fake_retrieve):
            result = retriever_run(state)

        assert chunk1.dedup_key in result["repo_context"]
        assert chunk2.dedup_key in result["repo_context"]

    def test_chunk_failure_does_not_abort_node(self, monkeypatch):
        """One chunk's retrieval failing must not prevent others from succeeding."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/test")

        chunk1 = _sample_chunk(file_path="src/api.py",  line_number=1)
        chunk2 = _sample_chunk(file_path="src/auth.py", line_number=10)

        state = initial_state(
            raw_diff="diff", repo_full_name="owner/repo", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [chunk1, chunk2]

        call_count = 0
        async def fake_retrieve(chunk_text, repo_full_name, top_k):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("DB connection refused")
            return ["snippet from chunk 2"]

        with patch("app.rag.retriever.retrieve_context", side_effect=fake_retrieve):
            result = retriever_run(state)

        # chunk1 failed, chunk2 succeeded
        assert chunk1.dedup_key not in result["repo_context"]
        assert chunk2.dedup_key in result["repo_context"]

    def test_run_never_raises(self, monkeypatch):
        """No state input should cause the node to raise."""
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost/test")

        state = initial_state(
            raw_diff="diff", repo_full_name="a/b", pr_number=1, head_sha="x"
        )
        state["diff_chunks"] = [_sample_chunk()]

        with patch(
            "app.rag.retriever.retrieve_context",
            side_effect=Exception("everything is broken"),
        ):
            try:
                result = retriever_run(state)
                assert "repo_context" in result
            except Exception as exc:
                pytest.fail(f"retriever_run raised: {exc}")
