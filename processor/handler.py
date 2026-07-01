"""
processor/handler.py
─────────────────────
AWS Lambda entry point. This is the function QStash calls after dequeuing
a PR event from the webhook receiver.

Execution flow:
    QStash HTTP POST
        │
        ▼
    handler(event, context)   ← Lambda entry point
        │
        ├── 1. Verify QStash signature   (is this really QStash?)
        ├── 2. Parse event body          (extract PR metadata)
        ├── 3. Create DB row             (audit trail from the start)
        ├── 4. Fetch PR diff             (GitHub API)
        ├── 5. Run agent pipeline        (LangGraph → Groq)
        ├── 6. Post review comments      (GitHub API)
        ├── 7. Persist results to DB     (reviews + review_comments tables)
        └── 8. Return 200               (tells QStash: success, don't retry)

QStash retry contract:
    QStash treats any 2xx response as success and stops retrying.
    Any non-2xx causes QStash to retry with exponential backoff (up to 3x,
    per the "Upstash-Retries: 3" header the webhook receiver set).

    This means our error handling has two tiers:
    - Transient failures (GitHub rate limit, Groq timeout): return 5xx so
      QStash retries. The next attempt might succeed.
    - Permanent failures (malformed payload, invalid installation ID): return
      2xx so QStash stops retrying. Retrying won't fix a bad payload.

    We write failed events to Supabase's dlq_events table for permanent
    failures, and mark the review row as "failed" for transient ones.

Async DB in a sync Lambda handler:
    The DB session layer (app/db/session.py) is async — it uses asyncpg and
    SQLAlchemy's async engine. Lambda handlers are synchronous functions.
    We bridge this with asyncio.run(), which creates a new event loop, runs
    the coroutine to completion, and tears it down. This is safe in Lambda
    because there's no pre-existing event loop in the handler thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import jwt

from app.agent.graph import run_pipeline
from app.db.session import (
    complete_review,
    create_review,
    fail_review,
    get_or_create_repository,
    get_session,
    mark_review_processing,
    write_dlq_event,
)
from app.github.client import get_github_client
from app.github.comment_poster import post_review
from app.github.diff_fetcher import fetch_pr_diff

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.

    Args:
        event:   Lambda event dict. When invoked via Function URL, the HTTP
                 request body is in event["body"] (string), headers in
                 event["headers"].
        context: Lambda context object (execution metadata — not used here).

    Returns:
        A dict with "statusCode" and "body" — Lambda Function URL format.
        2xx → QStash stops retrying.
        5xx → QStash retries (up to the configured retry count).
    """
    # ── Step 1: Verify QStash signature ─────────────────────────────────────
    headers = event.get("headers", {})
    body_str: str = event.get("body", "") or ""

    signature = headers.get("upstash-signature") or headers.get("Upstash-Signature")

    try:
        _verify_qstash_signature(signature, body_str)
    except SignatureError as exc:
        logger.warning("handler: QStash signature verification failed: %s", exc)
        return _response(403, "Invalid QStash signature")

    # ── Step 2: Parse event body ─────────────────────────────────────────────
    try:
        payload = json.loads(body_str)
    except json.JSONDecodeError as exc:
        logger.error("handler: failed to parse event body: %s", exc)
        _try_write_dlq(payload={}, reason=f"Invalid JSON: {exc}")
        return _response(200, "Invalid JSON payload — not retrying")

    try:
        repo_full_name: str  = payload["repo_full_name"]
        pr_number: int       = int(payload["pr_number"])
        head_sha: str        = payload["head_sha"]
        installation_id: int = int(payload["installation_id"])
        github_repo_id: int  = int(payload.get("github_repo_id", 0))
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("handler: missing required fields in payload: %s", exc)
        _try_write_dlq(payload=payload, reason=f"Malformed payload: {exc}")
        return _response(200, f"Malformed payload: {exc} — not retrying")

    logger.info(
        "handler: processing %s PR #%d sha=%s",
        repo_full_name, pr_number, head_sha[:8],
    )

    start_time = time.monotonic()

    # ── Step 3: Create DB row early — audit trail from the start ─────────────
    # We create the review row immediately so there's a DB record even if
    # the handler crashes later. The row starts in "queued" status and
    # transitions through "processing" → "completed" | "failed".
    review_row = None
    try:
        review_row = asyncio.run(_create_review_row(
            github_repo_id=github_repo_id,
            repo_full_name=repo_full_name,
            installation_id=installation_id,
            pr_number=pr_number,
            head_sha=head_sha,
        ))
    except Exception as exc:
        # DB failure on row creation is non-fatal — we log and continue.
        # A missing DB row is better than a dropped review.
        logger.error("handler: failed to create review row: %s", exc)

    # ── Step 4: Fetch PR diff ────────────────────────────────────────────────
    try:
        github_client = get_github_client(installation_id)
        raw_diff = fetch_pr_diff(github_client, repo_full_name, pr_number)
    except Exception as exc:
        logger.error(
            "handler: failed to fetch diff for %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )
        _try_fail_review(review_row, f"Diff fetch failed: {exc}")
        return _response(500, f"Diff fetch failed: {exc}")

    # ── Step 5: Run agent pipeline ───────────────────────────────────────────
    if review_row:
        _try_mark_processing(review_row)

    try:
        final_state = run_pipeline(
            raw_diff=raw_diff,
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
        )
    except Exception as exc:
        logger.error(
            "handler: pipeline failed for %s PR #%d: %s",
            repo_full_name, pr_number, exc,
        )
        _try_fail_review(review_row, f"Pipeline failed: {exc}")
        return _response(500, f"Pipeline failed: {exc}")

    final_comments = final_state.get("final_comments", [])
    tokens_used = final_state.get("tokens_used", 0)
    pipeline_error = final_state.get("error")

    if pipeline_error:
        logger.warning(
            "handler: pipeline completed with partial error for %s PR #%d: %s",
            repo_full_name, pr_number, pipeline_error,
        )

    # ── Step 6: Post review comments ─────────────────────────────────────────
    comments_posted = 0
    if final_comments:
        try:
            comments_posted = post_review(
                client=github_client,
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                head_sha=head_sha,
                findings=final_comments,
            )
        except Exception as exc:
            logger.error(
                "handler: failed to post review for %s PR #%d: %s",
                repo_full_name, pr_number, exc,
            )
            _try_fail_review(review_row, f"Review post failed: {exc}")
            return _response(500, f"Review post failed: {exc}")
    else:
        logger.info(
            "handler: no comments to post for %s PR #%d",
            repo_full_name, pr_number,
        )

    # ── Step 7: Persist results to DB ────────────────────────────────────────
    latency_ms = int((time.monotonic() - start_time) * 1000)

    if review_row:
        try:
            asyncio.run(_complete_review_row(
                review=review_row,
                comments_posted=comments_posted,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                findings=final_comments,
            ))
        except Exception as exc:
            # DB failure on completion is non-fatal — the review was posted
            # to GitHub successfully. Log but don't return 5xx.
            logger.error("handler: failed to persist completion to DB: %s", exc)

    logger.info(
        "handler: completed %s PR #%d — "
        "%d comments posted, %d tokens used, %dms latency",
        repo_full_name, pr_number, comments_posted, tokens_used, latency_ms,
    )

    return _response(200, {
        "repo": repo_full_name,
        "pr_number": pr_number,
        "comments_posted": comments_posted,
        "tokens_used": tokens_used,
        "latency_ms": latency_ms,
        "pipeline_error": pipeline_error,
    })


# ---------------------------------------------------------------------------
# Async DB helpers — called via asyncio.run() from the sync handler
# ---------------------------------------------------------------------------

async def _create_review_row(
    github_repo_id: int,
    repo_full_name: str,
    installation_id: int,
    pr_number: int,
    head_sha: str,
):
    """Create (or fetch) repository and review rows in a single transaction."""
    async with get_session() as session:
        repo = await get_or_create_repository(
            session,
            github_repo_id=github_repo_id,
            full_name=repo_full_name,
            installation_id=installation_id,
        )
        review = await create_review(
            session,
            repo_id=repo.id,
            pr_number=pr_number,
            head_sha=head_sha,
        )
        return review


async def _complete_review_row(review, comments_posted, tokens_used, latency_ms, findings):
    """Persist final metrics and comment rows."""
    async with get_session() as session:
        # Re-attach the review object to this new session
        session.add(review)
        await complete_review(
            session,
            review=review,
            comments_posted=comments_posted,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            findings=findings,
        )


# ---------------------------------------------------------------------------
# Fire-and-forget DB helpers — failures are logged, never raised
# ---------------------------------------------------------------------------

def _try_mark_processing(review) -> None:
    try:
        asyncio.run(_mark_processing(review))
    except Exception as exc:
        logger.error("handler: failed to mark review processing: %s", exc)


def _try_fail_review(review, reason: str) -> None:
    if review is None:
        return
    try:
        asyncio.run(_fail_review_row(review, reason))
    except Exception as exc:
        logger.error("handler: failed to mark review failed: %s", exc)


def _try_write_dlq(payload: dict, reason: str) -> None:
    try:
        asyncio.run(_write_dlq(payload, reason))
    except Exception as exc:
        logger.error("handler: failed to write DLQ event: %s", exc)


async def _mark_processing(review) -> None:
    async with get_session() as session:
        session.add(review)
        await mark_review_processing(session, review)


async def _fail_review_row(review, reason: str) -> None:
    async with get_session() as session:
        session.add(review)
        await fail_review(session, review, reason)


async def _write_dlq(payload: dict, reason: str) -> None:
    async with get_session() as session:
        await write_dlq_event(session, payload=payload, failure_reason=reason)


# ---------------------------------------------------------------------------
# QStash signature verification
# ---------------------------------------------------------------------------

class SignatureError(Exception):
    """Raised when QStash JWT signature verification fails."""


def _verify_qstash_signature(signature: Optional[str], body: str) -> None:
    """
    Verify the QStash JWT signature on an inbound request.

    QStash signs each outbound delivery with a JWT in the Upstash-Signature
    header. The JWT is HS256-signed with QSTASH_CURRENT_SIGNING_KEY.

    QStash provides two keys (current + next) for zero-downtime rotation.
    We try CURRENT first; if that fails, try NEXT. If both fail, reject.
    """
    if not signature:
        raise SignatureError("Missing Upstash-Signature header")

    current_key = os.environ.get("QSTASH_CURRENT_SIGNING_KEY", "")
    next_key = os.environ.get("QSTASH_NEXT_SIGNING_KEY", "")
    lambda_url = os.environ.get("LAMBDA_FUNCTION_URL", "")

    if not current_key:
        raise SignatureError("QSTASH_CURRENT_SIGNING_KEY not configured")

    for key in filter(None, [current_key, next_key]):
        try:
            jwt.decode(
                signature,
                key,
                algorithms=["HS256"],
                options={"verify_aud": False},
                issuer="Upstash",
            )
            decoded = jwt.decode(
                signature,
                key,
                algorithms=["HS256"],
                options={"verify_aud": False, "verify_exp": False},
            )
            sub = decoded.get("sub", "")
            if lambda_url and sub != lambda_url:
                continue
            return
        except jwt.ExpiredSignatureError:
            raise SignatureError("QStash JWT has expired")
        except jwt.InvalidTokenError:
            continue

    raise SignatureError("QStash JWT verification failed with all available keys")


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def _response(status_code: int, body: Any) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body) if isinstance(body, dict) else str(body),
    }
