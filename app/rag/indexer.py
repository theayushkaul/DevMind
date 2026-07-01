"""
app/rag/indexer.py
───────────────────
Repository indexer: embeds a repo's source files and writes them to pgvector.

This is a one-off operation triggered per repository, not per PR. Once a
repo is indexed, retriever.py handles all subsequent lookups. The DB's
`repositories.indexed_at` column tracks whether a repo has been indexed.

Execution model:
  The indexer is NOT called on every PR event — that would be enormously
  slow (cloning + embedding a large codebase takes minutes). Instead:

  1. When the Lambda handler processes a PR, it checks repositories.indexed_at.
  2. If NULL → trigger indexing (first PR from this repo).
  3. If set → skip indexing, go straight to retrieval.

  Re-indexing happens on a schedule (e.g. a daily Lambda cron) or when
  the developer explicitly requests it via a webhook comment "/devmind reindex".

Chunking strategy for indexing (different from diff chunking):
  - Split each file into 512-token chunks with 64-token overlap.
  - Overlap ensures functions that span a chunk boundary have their signature
    visible in both the chunk that contains the def and the chunk that
    contains the body — the LLM needs both to understand the function.
  - Token counting uses a simple word-based approximation (1 token ≈ 4 chars)
    rather than a real tokenizer, to avoid adding tiktoken as a dependency.
    For chunking purposes this is accurate enough.

Files we index:
  Same language extensions as DiffParserNode recognises: .py, .js, .ts,
  .java, .go, .rs, .rb, .php, .cs, .cpp, .c, .kt, .swift.
  Configuration files (.yaml, .toml, .json) are excluded — they're usually
  auto-generated or too noisy for semantic search.

Batch size:
  We embed BATCH_SIZE chunks per HF API call. bge-small-en-v1.5 can handle
  64 inputs per call. We use 32 to stay well within limits and keep memory
  usage bounded for very large repos.
"""

from __future__ import annotations

import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CodeEmbedding, Repository
from app.rag.embedder import embed_texts, EmbeddingError

logger = logging.getLogger(__name__)

# Extensions to index — must match DiffParserNode's EXTENSION_TO_LANGUAGE keys
INDEXABLE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".rs", ".rb", ".php",
    ".cs", ".cpp", ".c", ".h", ".hpp",
    ".kt", ".swift",
})

# Chunking constants (character-based approximation of token counts)
# 512 tokens × 4 chars/token = 2048 chars per chunk
# 64 tokens × 4 chars/token = 256 chars overlap
CHUNK_SIZE_CHARS = 2048
OVERLAP_CHARS = 256

# Number of chunks per HuggingFace API call
BATCH_SIZE = 32


class IndexingError(Exception):
    """Raised when repository indexing fails unrecoverably."""


async def index_repository(
    session: AsyncSession,
    repository: Repository,
    github_token: str,
) -> int:
    """
    Clone a GitHub repository, embed all source files, and write to pgvector.

    This is an idempotent operation — re-indexing a repo first deletes all
    existing code_embeddings rows for that repo, then inserts fresh ones.
    This avoids stale embeddings after large refactors.

    Args:
        session:      Active async DB session.
        repository:   Repository ORM object (must have full_name set).
        github_token: Installation access token for cloning the private repo.

    Returns:
        Total number of chunks indexed.

    Raises:
        IndexingError: if cloning or embedding fails.
    """
    repo_full_name = repository.full_name
    logger.info("indexer: starting indexing for %s", repo_full_name)

    # Step 1: Clone the repo into a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_url = f"https://x-access-token:{github_token}@github.com/{repo_full_name}.git"
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "--quiet", clone_url, tmpdir],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            raise IndexingError(
                f"Failed to clone {repo_full_name}: {exc.stderr.decode()[:200]}"
            ) from exc
        except subprocess.TimeoutExpired:
            raise IndexingError(f"Clone of {repo_full_name} timed out after 120s")

        # Step 2: Collect all indexable source files
        source_files = _collect_source_files(Path(tmpdir))
        logger.info(
            "indexer: found %d source files in %s",
            len(source_files), repo_full_name,
        )

        if not source_files:
            logger.warning("indexer: no indexable source files found in %s", repo_full_name)
            return 0

        # Step 3: Chunk all files into (file_path, chunk_index, text) tuples
        all_chunks: List[Tuple[str, int, str]] = []
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                repo_relative_path = str(file_path.relative_to(tmpdir)).replace("\\", "/")
                chunks = _chunk_text(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append((repo_relative_path, i, chunk))
            except Exception as exc:
                logger.debug("indexer: skipping %s — %s", file_path, exc)
                continue

        logger.info("indexer: produced %d chunks from %s", len(all_chunks), repo_full_name)

        # Step 4: Delete existing embeddings for this repo (idempotent re-index)
        await _delete_existing_embeddings(session, repository.id)

        # Step 5: Embed in batches and write to DB
        total_indexed = await _embed_and_store(
            session=session,
            repo_id=repository.id,
            all_chunks=all_chunks,
        )

    logger.info(
        "indexer: completed %s — %d chunks indexed",
        repo_full_name, total_indexed,
    )
    return total_indexed


def _collect_source_files(root: Path) -> List[Path]:
    """
    Walk the cloned repo directory and return all indexable source files.

    Skips:
    - Hidden directories (.git, .github, .venv, node_modules, etc.)
    - Files larger than 100KB (likely auto-generated or minified)
    - Files with non-indexable extensions
    """
    files = []
    skip_dirs = {".git", ".github", "node_modules", ".venv", "venv",
                 "__pycache__", ".mypy_cache", "dist", "build", ".next"}

    for path in root.rglob("*"):
        # Skip hidden dirs and known noise dirs
        if any(part in skip_dirs or part.startswith(".") for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in INDEXABLE_EXTENSIONS:
            continue
        # Skip files over 100KB — likely generated or minified
        if path.stat().st_size > 100_000:
            logger.debug("indexer: skipping large file %s", path)
            continue
        files.append(path)

    return files


def _chunk_text(text: str) -> List[str]:
    """
    Split a source file's text into overlapping chunks.

    Uses character count as a proxy for token count (1 token ≈ 4 chars).
    Chunks split at line boundaries rather than mid-line to preserve
    syntactic coherence — partial lines confuse the embedding model.
    """
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    chunks = []
    current_chars = 0
    current_lines: List[str] = []

    for line in lines:
        current_lines.append(line)
        current_chars += len(line)

        if current_chars >= CHUNK_SIZE_CHARS:
            chunk = "".join(current_lines)
            chunks.append(chunk)

            # Backtrack for overlap — keep the last N chars worth of lines
            overlap_lines: List[str] = []
            overlap_chars = 0
            for prev_line in reversed(current_lines):
                if overlap_chars + len(prev_line) > OVERLAP_CHARS:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_chars += len(prev_line)

            current_lines = overlap_lines
            current_chars = overlap_chars

    # Flush the last partial chunk
    if current_lines:
        remaining = "".join(current_lines).strip()
        if remaining:
            chunks.append(remaining)

    return chunks


async def _delete_existing_embeddings(
    session: AsyncSession,
    repo_id: uuid.UUID,
) -> None:
    """Delete all code_embeddings rows for a repo before re-indexing."""
    from sqlalchemy import delete
    from app.db.models import CodeEmbedding
    await session.execute(
        delete(CodeEmbedding).where(CodeEmbedding.repo_id == repo_id)
    )
    logger.debug("indexer: deleted existing embeddings for repo %s", repo_id)


async def _embed_and_store(
    session: AsyncSession,
    repo_id: uuid.UUID,
    all_chunks: List[Tuple[str, int, str]],
) -> int:
    """
    Embed chunks in batches and write CodeEmbedding rows to the DB.

    Returns the total number of chunks successfully indexed.
    """
    total = 0

    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
        batch_texts = [chunk_text for _, _, chunk_text in batch]

        try:
            vectors = embed_texts(batch_texts)
        except EmbeddingError as exc:
            logger.error(
                "indexer: embedding batch %d-%d failed: %s — skipping",
                batch_start, batch_start + len(batch), exc,
            )
            continue

        for (file_path, chunk_index, chunk_text), vector in zip(batch, vectors):
            # Store vector as PostgreSQL array literal: '{0.1,0.2,...}'
            # pgvector accepts this format via a text cast in the migration column.
            vector_str = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

            embedding = CodeEmbedding(
                repo_id=repo_id,
                file_path=file_path,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                embedding=vector_str,
            )
            session.add(embedding)
            total += 1

        await session.flush()
        logger.debug(
            "indexer: flushed batch %d-%d (%d chunks)",
            batch_start, batch_start + len(batch), len(batch),
        )

    return total
