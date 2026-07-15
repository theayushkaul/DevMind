"""
scripts/index_repo.py
──────────────────────
One-off script to index a GitHub repository's codebase into pgvector.

Run this once per repository to enable RAG context retrieval during reviews.
After indexing, DevMind can retrieve semantically related code snippets
when reviewing pull requests — giving the LLM context about the broader
codebase, not just the diff in isolation.

Usage:
    # Index a public repo
    python scripts/index_repo.py --repo ayushkaul/devmind

    # Index a private repo (requires GitHub App installation)
    python scripts/index_repo.py --repo ayushkaul/private-repo --installation-id 12345678

    # Check if a repo is already indexed
    python scripts/index_repo.py --repo ayushkaul/devmind --check-only

Prerequisites:
    1. DATABASE_URL must be set (Supabase PostgreSQL + pgvector)
    2. HUGGINGFACE_API_KEY must be set (for embedding)
    3. For private repos: GITHUB_APP_ID and GITHUB_PRIVATE_KEY must be set

What it does:
    1. Creates (or looks up) a repository row in Supabase
    2. Clones the repo at HEAD with --depth=1
    3. Chunks all source files (Python, JS, TS, Java, Go, Rust, etc.)
    4. Embeds chunks in batches via HuggingFace Inference API (BAAI/bge-small-en-v1.5)
    5. Writes embedding rows to code_embeddings table
    6. Updates repositories.indexed_at timestamp

Re-running this script on an already-indexed repo is safe — it deletes
all existing embeddings first and re-indexes from scratch (idempotent).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import get_session, get_or_create_repository
from app.github.client import get_github_client
from app.rag.indexer import index_repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"❌ Error: {name} environment variable is not set.")
        print(f"   Set it in .env or export it before running this script.")
        sys.exit(1)
    return value


async def _get_installation_token(installation_id: int) -> str:
    """
    Generate a GitHub App installation access token for cloning private repos.
    """
    from github import GithubIntegration
    app_id = _require_env("GITHUB_APP_ID")
    private_key = _require_env("GITHUB_PRIVATE_KEY")

    integration = GithubIntegration(
        integration_id=int(app_id),
        private_key=private_key,
    )
    auth = integration.get_access_token(installation_id)
    return auth.token


async def _check_indexed(repo_full_name: str) -> None:
    """Print indexing status for a repository without re-indexing."""
    from sqlalchemy import select
    from app.db.models import Repository

    async with get_session() as session:
        result = await session.execute(
            select(Repository).where(Repository.full_name == repo_full_name)
        )
        repo = result.scalar_one_or_none()

    if repo is None:
        print(f"ℹ️  {repo_full_name} has not been indexed yet (no DB record).")
    elif repo.indexed_at is None:
        print(f"⚠️  {repo_full_name} exists in DB but indexed_at is NULL (indexing may have failed).")
    else:
        print(f"✅ {repo_full_name} was last indexed at {repo.indexed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")


async def _run_indexing(
    repo_full_name: str,
    installation_id: int | None,
    github_repo_id: int,
) -> None:
    """Main async indexing flow."""
    print(f"\n📦 Indexing {repo_full_name}...")

    # Get GitHub token for cloning
    if installation_id:
        print("  → Getting GitHub App installation token...")
        github_token = await _get_installation_token(installation_id)
    else:
        # For public repos, use a personal access token if available
        github_token = os.environ.get("GITHUB_TOKEN", "")
        if not github_token:
            print("  ℹ️  No GITHUB_TOKEN set — attempting to clone as public repo")

    start = time.monotonic()

    async with get_session() as session:
        # Get or create the repository record
        print("  → Creating/updating repository record in Supabase...")
        repository = await get_or_create_repository(
            session=session,
            github_repo_id=github_repo_id,
            full_name=repo_full_name,
            installation_id=installation_id,
        )

        # Run the indexer
        print("  → Cloning repository (depth=1)...")
        print("  → Chunking source files...")
        print("  → Embedding chunks via HuggingFace API (this may take a few minutes)...")

        chunks_indexed = await index_repository(
            session=session,
            repository=repository,
            github_token=github_token,
        )

        # Update indexed_at timestamp
        from datetime import datetime, timezone
        repository.indexed_at = datetime.now(timezone.utc)
        await session.commit()

    elapsed = time.monotonic() - start
    print(f"\n✅ Indexing complete!")
    print(f"   Chunks indexed: {chunks_indexed}")
    print(f"   Time elapsed:   {elapsed:.1f}s")
    print(f"\n{repo_full_name} is now ready for RAG-enhanced code reviews.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index a GitHub repository into pgvector for DevMind RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="OWNER/REPO",
        help="GitHub repository to index (e.g. ayushkaul/devmind)",
    )
    parser.add_argument(
        "--installation-id",
        type=int,
        metavar="ID",
        help="GitHub App installation ID (required for private repos)",
    )
    parser.add_argument(
        "--github-repo-id",
        type=int,
        metavar="ID",
        default=0,
        help="GitHub's numeric repo ID (optional — 0 if unknown)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check indexing status without re-indexing",
    )
    args = parser.parse_args()

    # Validate repo format
    if "/" not in args.repo or args.repo.count("/") != 1:
        print(f"❌ Invalid repo format: {args.repo!r}. Expected OWNER/REPO.")
        sys.exit(1)

    # Always require DATABASE_URL and HF key
    _require_env("DATABASE_URL")

    if args.check_only:
        await _check_indexed(args.repo)
        return

    _require_env("HUGGINGFACE_API_KEY")

    await _run_indexing(
        repo_full_name=args.repo,
        installation_id=args.installation_id,
        github_repo_id=args.github_repo_id,
    )


if __name__ == "__main__":
    asyncio.run(main())
