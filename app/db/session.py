"""
app/db/session.py
──────────────────
Async SQLAlchemy session factory and database repository functions.

Why async?
    FastAPI and Lambda both run in async contexts. Blocking DB calls
    (synchronous SQLAlchemy) would block the event loop during I/O — the
    equivalent of sleeping in the middle of serving a request. asyncpg +
    SQLAlchemy's async extension gives us true non-blocking DB access.

Session lifecycle:
    Lambda is short-lived: one invocation, one DB operation (or a small
    sequence), then done. We create a session per handler invocation using
    get_session() as an async context manager, commit or rollback, and close.
    There's no persistent connection pool that survives across invocations —
    Lambda's execution environment may be recycled, and holding open DB
    connections across cold starts causes pool exhaustion.

    The engine is module-level (created once per Lambda warm instance) and
    reused across invocations on the same warm instance — this is the correct
    Lambda pattern. The session is per-invocation.

Repository functions:
    Rather than scattering raw ORM queries across handler.py, we define named
    repository functions here: get_or_create_repository(), create_review(),
    etc. Each function takes a session and returns a model object.

    This keeps the handler readable ("create a review row" instead of
    "construct a Review object, add it to session, flush") and makes the
    DB interaction independently testable by injecting a mock session.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import select

from app.agent.state import ReviewFinding
from app.db.models import (
    Base,
    CodeEmbedding,
    DLQEvent,
    Repository,
    Review,
    ReviewComment,
    ReviewStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine — created once per Lambda warm instance
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None


def _get_engine():
    global _engine, _session_factory
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Format: postgresql+asyncpg://user:password@host:5432/dbname"
            )

        # Ensure the URL uses the asyncpg driver
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        _engine = create_async_engine(
            database_url,
            # Pool settings tuned for Lambda:
            # - pool_size=2: Lambda rarely needs more than 1-2 concurrent
            #   connections per warm instance.
            # - max_overflow=3: Allow brief spikes.
            # - pool_pre_ping=True: Discard stale connections after Lambda
            #   instance hibernation (prevents "SSL connection has been closed"
            #   errors on the first query after a cold-ish start).
            pool_size=2,
            max_overflow=3,
            pool_pre_ping=True,
            echo=os.environ.get("ENVIRONMENT") == "development",
        )
        _session_factory = async_sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine, _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager providing a database session.

    Usage:
        async with get_session() as session:
            repo = await get_or_create_repository(session, ...)
            await session.commit()

    Automatically rolls back on exception and closes the session on exit.
    """
    _, session_factory = _get_engine()
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Repository functions
# ---------------------------------------------------------------------------

async def get_or_create_repository(
    session: AsyncSession,
    github_repo_id: int,
    full_name: str,
    installation_id: Optional[int] = None,
) -> Repository:
    """
    Fetch an existing Repository row or create one if it doesn't exist.

    Uses SELECT then INSERT pattern (not INSERT ... ON CONFLICT) because
    we need to return the full ORM object either way, and SQLAlchemy's
    async session doesn't expose ON CONFLICT DO NOTHING with RETURNING
    cleanly. The SELECT is cheap (indexed on github_repo_id).

    Args:
        session:        Active async session.
        github_repo_id: The numeric GitHub repository ID (stable across renames).
        full_name:      e.g. "ayushkaul/devmind". May change if repo is renamed.
        installation_id: GitHub App installation ID for this repo's owner.

    Returns:
        Repository ORM object (either fetched or newly created).
    """
    result = await session.execute(
        select(Repository).where(Repository.github_repo_id == github_repo_id)
    )
    repo = result.scalar_one_or_none()

    if repo is None:
        repo = Repository(
            github_repo_id=github_repo_id,
            full_name=full_name,
            installation_id=installation_id,
        )
        session.add(repo)
        await session.flush()  # flush to get the generated ID without committing
        logger.info("db: created new repository row for %s", full_name)
    else:
        # Update mutable fields in case they've changed (repo rename, new install)
        if repo.full_name != full_name:
            repo.full_name = full_name
        if installation_id and repo.installation_id != installation_id:
            repo.installation_id = installation_id

    return repo


async def create_review(
    session: AsyncSession,
    repo_id: uuid.UUID,
    pr_number: int,
    head_sha: str,
) -> Review:
    """
    Create a new Review row in 'queued' status.

    Called by the Lambda handler immediately on dequeuing an event,
    before any processing begins. Provides a DB-level audit trail even
    if the handler crashes mid-processing.

    Returns:
        Newly created Review ORM object (flushed, not yet committed).
    """
    review = Review(
        repo_id=repo_id,
        pr_number=pr_number,
        head_sha=head_sha,
        status=ReviewStatus.QUEUED,
    )
    session.add(review)
    await session.flush()
    logger.info(
        "db: created review row %s for PR #%d sha=%s",
        review.id, pr_number, head_sha[:8],
    )
    return review


async def mark_review_processing(
    session: AsyncSession,
    review: Review,
) -> None:
    """Transition a review from 'queued' to 'processing'."""
    review.status = ReviewStatus.PROCESSING
    await session.flush()


async def complete_review(
    session: AsyncSession,
    review: Review,
    comments_posted: int,
    tokens_used: int,
    latency_ms: int,
    findings: List[ReviewFinding],
) -> None:
    """
    Mark a review as completed and persist all comment rows.

    Writes ReviewComment rows for every finding in `findings`, then
    updates the Review row with final metrics.

    Args:
        session:          Active async session.
        review:           The Review ORM object to update.
        comments_posted:  Number of comments successfully posted to GitHub.
        tokens_used:      Total LLM tokens consumed across all checker nodes.
        latency_ms:       End-to-end processing latency in milliseconds.
        findings:         The final_comments list from CommentSynthesizerNode.
    """
    # Insert ReviewComment rows
    for finding in findings:
        comment = ReviewComment(
            review_id=review.id,
            file_path=finding.file_path,
            line_number=finding.line_number if finding.line_number >= 0 else None,
            category=finding.category,
            severity=finding.severity,
            comment_body=finding.comment,
            confidence=finding.confidence,
            source_node=finding.source_node,
        )
        session.add(comment)

    # Update Review metrics
    review.status = ReviewStatus.COMPLETED
    review.comments_posted = comments_posted
    review.tokens_used = tokens_used
    review.latency_ms = latency_ms
    review.completed_at = datetime.now(timezone.utc)

    await session.flush()
    logger.info(
        "db: completed review %s — %d comments, %d tokens, %dms",
        review.id, comments_posted, tokens_used, latency_ms,
    )


async def fail_review(
    session: AsyncSession,
    review: Review,
    error_message: str,
) -> None:
    """Mark a review as failed with an error message."""
    review.status = ReviewStatus.FAILED
    review.error_message = error_message
    review.completed_at = datetime.now(timezone.utc)
    await session.flush()
    logger.error("db: failed review %s — %s", review.id, error_message)


async def write_dlq_event(
    session: AsyncSession,
    payload: dict,
    failure_reason: str,
    retry_count: int = 0,
) -> DLQEvent:
    """
    Persist a failed event to the dead letter queue table.

    Called when the Lambda handler encounters a permanent failure
    (malformed payload, invalid installation ID, etc.) and returns 200
    to QStash to stop retries. The event is preserved here for manual
    inspection and replay.
    """
    event = DLQEvent(
        payload=payload,
        failure_reason=failure_reason,
        retry_count=retry_count,
    )
    session.add(event)
    await session.flush()
    logger.warning(
        "db: wrote DLQ event %s — reason: %s", event.id, failure_reason
    )
    return event
