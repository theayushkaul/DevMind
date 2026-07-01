"""
app/db/models.py
─────────────────
SQLAlchemy ORM models for all five DevMind tables.

Table overview (mirrors the SQL schema in the project plan, Section 4):

    repositories    — GitHub repos DevMind has seen. One row per repo.
    reviews         — One row per PR review attempt. Tracks status and metrics.
    review_comments — Individual findings posted as GitHub review comments.
    dlq_events      — Failed events that exhausted QStash retries.
    code_embeddings — pgvector embeddings for RAG retrieval (built by indexer).

Design decisions:

UUID primary keys everywhere:
    Generated in Python (uuid.uuid4()) rather than relying on Postgres's
    gen_random_uuid(). This means we know the ID before the INSERT, which
    lets us pass it to related rows in the same transaction without a
    round-trip to get the auto-generated value. Also portable across DBs
    if we ever run tests against SQLite.

server_default vs default:
    Timestamp columns use server_default=func.now() — the timestamp is
    set by Postgres at INSERT time, not by Python. This is more correct:
    the "created_at" should reflect when the row hit the DB, not when the
    Python object was constructed (which could be seconds earlier if the
    session was held open).

    The exception is id columns, which use default=uuid.uuid4 (a Python
    callable) so the UUID is known before the INSERT.

Relationships:
    Defined with lazy="selectin" for async compatibility. SQLAlchemy async
    sessions don't support lazy loading (it would require an implicit I/O
    call inside an async context, which SQLAlchemy blocks). "selectin"
    issues a SELECT IN query eagerly when the parent is loaded.

code_embeddings / pgvector:
    The `embedding` column stores vector(384) — a pgvector extension type.
    SQLAlchemy doesn't know about pgvector natively, so we declare it as
    a custom type using Text as the base (Alembic will render the correct
    SQL using the type_annotation_map override in env.py). The actual
    pgvector operations (cosine similarity search) happen in raw SQL via
    the session, not through the ORM — the ORM is only used for INSERT
    and basic SELECT here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    Shared declarative base for all models.
    All models inherit from this — Alembic's env.py imports it to discover
    the full metadata for autogenerate.
    """
    pass


# ---------------------------------------------------------------------------
# repositories
# ---------------------------------------------------------------------------

class Repository(Base):
    """
    A GitHub repository that DevMind has processed at least one PR for.

    Created on first webhook receipt from a new repo. The `indexed_at`
    column is NULL until the RAG indexer has run for this repo — the
    RAGContextNode checks this to decide whether to skip retrieval.
    """
    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    github_repo_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    installation_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    reviews: Mapped[List["Review"]] = relationship(
        "Review", back_populates="repository", lazy="selectin"
    )
    code_embeddings: Mapped[List["CodeEmbedding"]] = relationship(
        "CodeEmbedding", back_populates="repository", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Repository {self.full_name!r}>"


# ---------------------------------------------------------------------------
# reviews
# ---------------------------------------------------------------------------

class ReviewStatus:
    """String constants for the reviews.status column."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Review(Base):
    """
    One row per PR review attempt.

    The UNIQUE(repo_id, pr_number, head_sha) constraint mirrors the Redis
    idempotency key — provides a second layer of deduplication at the DB
    level so duplicate reviews can't slip through even if Redis is
    temporarily unavailable.

    `status` transitions:
        queued → processing → completed
                            → failed
    """
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint(
            "repo_id", "pr_number", "head_sha",
            name="uq_reviews_repo_pr_sha",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    head_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=ReviewStatus.QUEUED)
    comments_posted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    repository: Mapped["Repository"] = relationship(
        "Repository", back_populates="reviews", lazy="selectin"
    )
    comments: Mapped[List["ReviewComment"]] = relationship(
        "ReviewComment", back_populates="review", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Review repo={self.repo_id} pr={self.pr_number} status={self.status!r}>"


# ---------------------------------------------------------------------------
# review_comments
# ---------------------------------------------------------------------------

class ReviewComment(Base):
    """
    One row per individual finding posted as a GitHub inline review comment.

    `github_comment_id` is populated after the GitHub API call succeeds —
    it's the ID GitHub assigns to the posted comment. NULL means the
    comment was generated but not yet posted (or posting failed).
    """
    __tablename__ = "review_comments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reviews.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    line_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    comment_body: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_node: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    github_comment_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    review: Mapped["Review"] = relationship(
        "Review", back_populates="comments", lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<ReviewComment {self.severity}/{self.category} "
            f"{self.file_path}:{self.line_number}>"
        )


# ---------------------------------------------------------------------------
# dlq_events
# ---------------------------------------------------------------------------

class DLQEvent(Base):
    """
    Dead Letter Queue — events that exhausted QStash retries.

    Written by the Lambda handler when a permanent failure occurs, or by a
    separate DLQ consumer when QStash moves a message to the dead letter
    queue. These rows are for manual inspection and replay — the `payload`
    column contains the original PR event dict so it can be re-submitted
    once the underlying issue is resolved.
    """
    __tablename__ = "dlq_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<DLQEvent id={self.id} retries={self.retry_count}>"


# ---------------------------------------------------------------------------
# code_embeddings  (pgvector)
# ---------------------------------------------------------------------------

class CodeEmbedding(Base):
    """
    Vector embeddings for RAG retrieval.

    Each row is one chunk of a repository's source file, embedded using
    BAAI/bge-small-en-v1.5 (384 dimensions). The ivfflat index on the
    embedding column (created in the migration) enables fast approximate
    nearest-neighbour search at query time.

    The `embedding` column is declared as Text here — SQLAlchemy doesn't
    know about pgvector's `vector(384)` type natively. The Alembic migration
    uses a raw `op.execute("CREATE EXTENSION IF NOT EXISTS vector")` and
    renders the column as `vector(384)` directly in SQL. When reading
    embeddings back, we use raw SQL (session.execute) rather than the ORM.

    Indexed by (repo_id, file_path, chunk_index) to support efficient
    lookup of all chunks for a given file during re-indexing.
    """
    __tablename__ = "code_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repo_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Declared as Text in SQLAlchemy — rendered as vector(384) in migration SQL.
    # See migration 0001_initial_schema.py for the raw SQL that creates this column.
    embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    repository: Mapped["Repository"] = relationship(
        "Repository", back_populates="code_embeddings", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<CodeEmbedding {self.file_path}[{self.chunk_index}]>"
