"""
app/db/migrations/env.py
─────────────────────────
Alembic migration environment — configures how migrations run.

Two modes:
    offline — generates SQL without a DB connection (for review/audit)
    online  — connects to the DB and applies migrations directly

Async support:
    SQLAlchemy 2.0 async engines can't run synchronous migration code
    directly. We use run_sync() to execute Alembic's synchronous migration
    logic within an async connection — the standard pattern for async
    SQLAlchemy + Alembic setups.

    Reference: https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

# Import Base so Alembic's autogenerate can discover all model metadata
from app.db.models import Base

# ---------------------------------------------------------------------------
# Alembic Config
# ---------------------------------------------------------------------------

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata Alembic uses for autogenerate (--autogenerate flag)
target_metadata = Base.metadata


def _get_database_url() -> str:
    """
    Read DATABASE_URL from environment at migration run time.

    Ensures asyncpg driver is used (Alembic would otherwise try to use
    the sync psycopg2 driver based on a plain postgresql:// URL).
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable must be set to run migrations.\n"
            "Example: postgresql+asyncpg://postgres:password@localhost:5432/devmind"
        )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# ---------------------------------------------------------------------------
# Offline mode — generate SQL without connecting
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Emit migration SQL to stdout without a live DB connection.
    Useful for generating SQL to review before applying, or for DBs
    you don't have direct access to (e.g. Supabase managed Postgres).

    Usage: alembic upgrade head --sql
    """
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Tell Alembic to compare column types precisely during autogenerate.
        # Without this, it won't detect column type changes (e.g. Text → String).
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — connect and apply
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    """Run migrations synchronously via an already-connected connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations via run_sync()."""
    engine = create_async_engine(
        _get_database_url(),
        poolclass=pool.NullPool,  # Don't pool connections for migrations
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — runs the async function."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
