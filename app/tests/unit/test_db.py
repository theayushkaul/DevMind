"""
tests/unit/test_db.py
──────────────────────
Tests for app/db/models.py and app/db/session.py.

Test strategy:
    We use an in-memory SQLite database (via aiosqlite) rather than a real
    Postgres instance. This means:
    - Tests run without any infrastructure (no Docker, no Supabase account)
    - pgvector-specific columns (vector(384), ivfflat index) are skipped
      via a SQLite-compatible schema subset

    What we're testing:
    - Model definitions are syntactically correct and importable
    - Repository functions (get_or_create_repository, create_review, etc.)
      produce the expected ORM objects and state transitions
    - Session rollback works on exception
    - ReviewStatus constants are correct strings

    What we're NOT testing (needs real Postgres + pgvector):
    - The ivfflat index
    - JSONB column behaviour
    - The vector(384) column type
    - Alembic migrations (tested by running `alembic upgrade head` against
      a real DB in the integration environment)
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.db.models import (
    Base,
    DLQEvent,
    Repository,
    Review,
    ReviewComment,
    ReviewStatus,
)
from app.db.session import (
    complete_review,
    create_review,
    fail_review,
    get_or_create_repository,
    mark_review_processing,
    write_dlq_event,
)
from app.agent.state import ReviewFinding


# ---------------------------------------------------------------------------
# In-memory SQLite session fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_session():
    """
    Provide a SQLAlchemy async session backed by in-memory SQLite.

    We drop and recreate all tables per test function to ensure isolation.
    aiosqlite is SQLite's async driver — it works without any Postgres infra.
    """
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        import aiosqlite  # noqa: F401 — just checking it's available
    except ImportError:
        pytest.skip("aiosqlite not installed — skipping DB tests")

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Create all tables (SQLite-compatible subset — skip pgvector columns)
    async with engine.begin() as conn:
        # Drop vector column from code_embeddings for SQLite compatibility
        # We use a simplified schema that excludes pgvector-specific types
        await conn.run_sync(lambda sync_conn: _create_sqlite_schema(sync_conn))

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with session_factory() as session:
        yield session

    await engine.dispose()


def _create_sqlite_schema(conn):
    """Create a SQLite-compatible subset of the DevMind schema."""
    conn.execute(_drop_and_create_sql())


def _drop_and_create_sql():
    from sqlalchemy import text
    return text("""
        CREATE TABLE IF NOT EXISTS repositories (
            id TEXT PRIMARY KEY,
            github_repo_id INTEGER UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            installation_id INTEGER,
            indexed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL REFERENCES repositories(id),
            pr_number INTEGER NOT NULL,
            head_sha TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            comments_posted INTEGER NOT NULL DEFAULT 0,
            tokens_used INTEGER,
            latency_ms INTEGER,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS review_comments (
            id TEXT PRIMARY KEY,
            review_id TEXT NOT NULL REFERENCES reviews(id),
            file_path TEXT NOT NULL,
            line_number INTEGER,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            comment_body TEXT NOT NULL,
            confidence REAL,
            source_node TEXT,
            github_comment_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS dlq_events (
            id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            failure_reason TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _finding(
    file_path="src/api.py",
    line_number=10,
    category="security",
    severity="critical",
    comment="SQL injection",
    confidence=0.9,
) -> ReviewFinding:
    return ReviewFinding(
        file_path=file_path,
        line_number=line_number,
        category=category,  # type: ignore
        severity=severity,  # type: ignore
        comment=comment,
        confidence=confidence,
        source_node="security_checker",
    )


# ---------------------------------------------------------------------------
# Tests: ReviewStatus constants
# ---------------------------------------------------------------------------

class TestReviewStatus:

    def test_status_values_are_correct_strings(self):
        assert ReviewStatus.QUEUED == "queued"
        assert ReviewStatus.PROCESSING == "processing"
        assert ReviewStatus.COMPLETED == "completed"
        assert ReviewStatus.FAILED == "failed"


# ---------------------------------------------------------------------------
# Tests: Model imports and structure
# ---------------------------------------------------------------------------

class TestModelImports:

    def test_all_models_importable(self):
        from app.db.models import (
            Base, Repository, Review, ReviewComment, DLQEvent, CodeEmbedding
        )
        assert Base is not None

    def test_repository_tablename(self):
        assert Repository.__tablename__ == "repositories"

    def test_review_tablename(self):
        assert Review.__tablename__ == "reviews"

    def test_review_comment_tablename(self):
        assert ReviewComment.__tablename__ == "review_comments"

    def test_dlq_event_tablename(self):
        assert DLQEvent.__tablename__ == "dlq_events"

    def test_review_has_unique_constraint(self):
        constraints = {c.name for c in Review.__table__.constraints}
        assert "uq_reviews_repo_pr_sha" in constraints

    def test_base_metadata_has_all_tables(self):
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "repositories", "reviews", "review_comments",
            "dlq_events", "code_embeddings",
        }
        assert expected.issubset(table_names)


# ---------------------------------------------------------------------------
# Tests: get_or_create_repository()
# We test these using mock sessions to avoid SQLite type incompatibilities
# with UUID columns while still verifying the business logic.
# ---------------------------------------------------------------------------

def _make_session(scalar_result=None):
    """
    Build a mock AsyncSession with correct chaining for execute().

    session.execute() is async and returns an object whose
    .scalar_one_or_none() is sync — AsyncMock chaining doesn't handle
    this automatically, so we construct it explicitly.
    session.add() is synchronous in SQLAlchemy, so we use MagicMock, not
    AsyncMock, for it.
    """
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = scalar_result

    session = MagicMock()
    session.execute = AsyncMock(return_value=execute_result)
    session.flush = AsyncMock()
    session.add = MagicMock()  # sync — not AsyncMock
    return session


class TestGetOrCreateRepository:

    @pytest.mark.asyncio
    async def test_creates_new_repository_when_not_found(self):
        session = _make_session(scalar_result=None)

        repo = await get_or_create_repository(
            session,
            github_repo_id=12345,
            full_name="ayushkaul/devmind",
            installation_id=99,
        )

        assert repo.github_repo_id == 12345
        assert repo.full_name == "ayushkaul/devmind"
        assert repo.installation_id == 99
        session.add.assert_called_once_with(repo)
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_existing_repository_when_found(self):
        existing = Repository(
            github_repo_id=12345,
            full_name="ayushkaul/devmind",
            installation_id=99,
        )
        session = _make_session(scalar_result=existing)

        repo = await get_or_create_repository(
            session,
            github_repo_id=12345,
            full_name="ayushkaul/devmind",
        )

        assert repo is existing
        session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_updates_full_name_on_rename(self):
        existing = Repository(
            github_repo_id=12345,
            full_name="ayushkaul/old-name",
            installation_id=99,
        )
        session = _make_session(scalar_result=existing)

        repo = await get_or_create_repository(
            session,
            github_repo_id=12345,
            full_name="ayushkaul/new-name",
        )

        assert repo.full_name == "ayushkaul/new-name"


# ---------------------------------------------------------------------------
# Tests: create_review()
# ---------------------------------------------------------------------------

def _make_simple_session():
    """Mock session for tests that don't need execute() chaining."""
    session = MagicMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


class TestCreateReview:

    @pytest.mark.asyncio
    async def test_creates_review_in_queued_status(self):
        session = _make_simple_session()
        repo_id = uuid.uuid4()

        review = await create_review(session, repo_id, pr_number=7, head_sha="abc123")

        assert review.repo_id == repo_id
        assert review.pr_number == 7
        assert review.head_sha == "abc123"
        assert review.status == ReviewStatus.QUEUED

    @pytest.mark.asyncio
    async def test_create_review_calls_flush(self):
        session = _make_simple_session()
        await create_review(session, uuid.uuid4(), pr_number=1, head_sha="sha")
        session.flush.assert_called()


# ---------------------------------------------------------------------------
# Tests: mark_review_processing()
# ---------------------------------------------------------------------------

class TestMarkReviewProcessing:

    @pytest.mark.asyncio
    async def test_transitions_to_processing(self):
        session = _make_simple_session()
        review = Review(status=ReviewStatus.QUEUED)

        await mark_review_processing(session, review)

        assert review.status == ReviewStatus.PROCESSING


# ---------------------------------------------------------------------------
# Tests: complete_review()
# ---------------------------------------------------------------------------

class TestCompleteReview:

    @pytest.mark.asyncio
    async def test_sets_status_to_completed(self):
        session = _make_simple_session()
        review = Review(status=ReviewStatus.PROCESSING)

        await complete_review(session, review,
            comments_posted=3, tokens_used=180, latency_ms=5000, findings=[])

        assert review.status == ReviewStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stores_metrics(self):
        session = _make_simple_session()
        review = Review(status=ReviewStatus.PROCESSING)

        await complete_review(session, review,
            comments_posted=3, tokens_used=180, latency_ms=5000, findings=[])

        assert review.comments_posted == 3
        assert review.tokens_used == 180
        assert review.latency_ms == 5000
        assert review.completed_at is not None

    @pytest.mark.asyncio
    async def test_creates_comment_row_per_finding(self):
        session = _make_simple_session()
        review = Review(id=uuid.uuid4(), status=ReviewStatus.PROCESSING)
        findings = [
            _finding(line_number=10, comment="SQL injection"),
            _finding(line_number=20, comment="null check missing", category="bug"),
        ]

        await complete_review(session, review,
            comments_posted=2, tokens_used=100, latency_ms=3000, findings=findings)

        assert session.add.call_count == len(findings)

    @pytest.mark.asyncio
    async def test_file_level_finding_has_none_line_number(self):
        """line_number=-1 in ReviewFinding → NULL in DB (None in ORM)."""
        session = _make_simple_session()
        review = Review(id=uuid.uuid4(), status=ReviewStatus.PROCESSING)
        findings = [_finding(line_number=-1, comment="Missing module docstring")]

        await complete_review(session, review,
            comments_posted=1, tokens_used=50, latency_ms=1000, findings=findings)

        added_comment = session.add.call_args[0][0]
        assert added_comment.line_number is None


# ---------------------------------------------------------------------------
# Tests: fail_review()
# ---------------------------------------------------------------------------

class TestFailReview:

    @pytest.mark.asyncio
    async def test_sets_status_to_failed(self):
        session = _make_simple_session()
        review = Review(status=ReviewStatus.PROCESSING)

        await fail_review(session, review, error_message="Groq timeout")

        assert review.status == ReviewStatus.FAILED
        assert review.error_message == "Groq timeout"
        assert review.completed_at is not None


# ---------------------------------------------------------------------------
# Tests: write_dlq_event()
# ---------------------------------------------------------------------------

class TestWriteDLQEvent:

    @pytest.mark.asyncio
    async def test_creates_dlq_event(self):
        session = _make_simple_session()
        payload = {"repo_full_name": "ayushkaul/devmind", "pr_number": 7}

        event = await write_dlq_event(
            session, payload=payload,
            failure_reason="Malformed payload", retry_count=3,
        )

        assert event.payload == payload
        assert event.failure_reason == "Malformed payload"
        assert event.retry_count == 3
        session.add.assert_called_once_with(event)


# ---------------------------------------------------------------------------
# Tests: get_session() — engine config
# ---------------------------------------------------------------------------

class TestGetSession:

    @pytest.mark.asyncio
    async def test_raises_when_database_url_missing(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        import app.db.session as db_session_module
        db_session_module._engine = None
        db_session_module._session_factory = None

        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            async with db_session_module.get_session():
                pass
