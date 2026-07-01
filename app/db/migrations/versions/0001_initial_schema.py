"""Initial schema — all five DevMind tables

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00.000000

Creates:
    - pgvector extension (required for vector(384) column type)
    - repositories table
    - reviews table
    - review_comments table
    - dlq_events table
    - code_embeddings table  (with ivfflat index for cosine similarity)

Note on pgvector:
    The `embedding` column uses pgvector's `vector(384)` type which SQLAlchemy
    doesn't know about natively. We create it via op.execute() with raw SQL
    rather than trying to map it through the ORM type system. The column is
    rendered as TEXT in the SQLAlchemy model (models.py) and treated as a
    native vector in raw SQL queries in retriever.py.

    Before running this migration, pgvector must be installed on your Postgres
    instance. Supabase includes it by default. For self-hosted Postgres:
        CREATE EXTENSION IF NOT EXISTS vector;
    or install the pgvector extension package for your Postgres version.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── pgvector extension ──────────────────────────────────────────────────
    # Must be created before any table that uses the vector() type.
    # IF NOT EXISTS makes this safe to re-run.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── repositories ────────────────────────────────────────────────────────
    op.create_table(
        "repositories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("github_repo_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("full_name", sa.Text(), nullable=False),
        sa.Column("installation_id", sa.BigInteger(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # ── reviews ─────────────────────────────────────────────────────────────
    op.create_table(
        "reviews",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "repo_id",
            UUID(as_uuid=True),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pr_number", sa.Integer(), nullable=False),
        sa.Column("head_sha", sa.String(40), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("comments_posted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "repo_id", "pr_number", "head_sha",
            name="uq_reviews_repo_pr_sha",
        ),
    )
    op.create_index("ix_reviews_repo_id", "reviews", ["repo_id"])
    op.create_index("ix_reviews_status", "reviews", ["status"])

    # ── review_comments ──────────────────────────────────────────────────────
    op.create_table(
        "review_comments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "review_id",
            UUID(as_uuid=True),
            sa.ForeignKey("reviews.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=True),
        sa.Column("category", sa.String(20), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("comment_body", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source_node", sa.String(50), nullable=True),
        sa.Column("github_comment_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_review_comments_review_id", "review_comments", ["review_id"])

    # ── dlq_events ───────────────────────────────────────────────────────────
    op.create_table(
        "dlq_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # ── code_embeddings (pgvector) ───────────────────────────────────────────
    # The embedding column uses pgvector's vector(384) type — rendered as raw
    # SQL because SQLAlchemy doesn't have a built-in pgvector type.
    op.create_table(
        "code_embeddings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "repo_id",
            UUID(as_uuid=True),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    # Add the pgvector column separately — can't use sa.Column for custom types
    op.execute("ALTER TABLE code_embeddings ADD COLUMN embedding vector(384)")

    # ivfflat index for approximate nearest-neighbour cosine similarity search.
    # lists=100 is a good default for up to ~1M vectors.
    # Higher lists = faster query, slower build, more memory.
    # Rule of thumb: lists ≈ sqrt(num_rows) for datasets under 1M rows.
    op.execute(
        "CREATE INDEX ix_code_embeddings_vector "
        "ON code_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )
    op.create_index(
        "ix_code_embeddings_repo_file",
        "code_embeddings",
        ["repo_id", "file_path"],
    )


def downgrade() -> None:
    op.drop_table("code_embeddings")
    op.drop_table("dlq_events")
    op.drop_table("review_comments")
    op.drop_table("reviews")
    op.drop_table("repositories")
    op.execute("DROP EXTENSION IF EXISTS vector")
