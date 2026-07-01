"""
app/webhook/router.py
──────────────────────
FastAPI router for the GitHub webhook endpoint.

This is the outermost layer of the system — the thing GitHub calls.
Its only job is to:
    1. Validate the signature (is this really GitHub?)
    2. Check idempotency (have we seen this exact event before?)
    3. Enqueue the event (hand off to QStash for async processing)
    4. Return 200 immediately (within GitHub's 10-second window)

Exactly nothing else. No diff parsing, no LLM calls, no DB writes.
Those all happen in Lambda, triggered by QStash, after this response
has already been sent.

Payload parsing strategy:
We read the raw body bytes BEFORE parsing JSON. This is required because
GitHub's HMAC signature is computed over the raw bytes — parsing to JSON
first (even with .body() then json.loads()) can subtly alter whitespace
or key ordering in some edge cases, causing valid signatures to fail.
FastAPI's Request object gives us access to the raw bytes via await
request.body().

Events we handle:
  - pull_request with action in {opened, synchronize, reopened}

Events we silently ignore (return 200 without enqueuing):
  - pull_request with other actions (closed, labeled, assigned, etc.)
  - Any other event type (push, issues, etc.)
  Returning 200 for ignored events is correct — it tells GitHub "received
  and understood", preventing unnecessary retries.

Events we reject (return 403):
  - Invalid or missing signature
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, Response

from app.webhook.idempotency import IdempotencyStore
from app.webhook.queue import QueuePublishError, publish_pr_event
from app.webhook.validator import SignatureValidationError, validate_signature

logger = logging.getLogger(__name__)

router = APIRouter()

# PR actions that should trigger a review.
# "synchronize" fires when a new commit is pushed to an open PR.
REVIEWABLE_ACTIONS = frozenset({"opened", "synchronize", "reopened"})


@router.post("/webhook")
async def github_webhook(request: Request) -> Response:
    """
    Receive and dispatch GitHub webhook events.

    Returns 200 for all valid requests (including events we intentionally
    ignore), 403 for signature failures, 422 for unparseable payloads.
    Never returns 500 — if enqueuing fails we log the error and still
    return 200 to prevent GitHub from retrying an event that would fail
    for the same reason (misconfigured env vars, etc).

    Design note — why not return 500 on queue failure?
    GitHub retries webhooks on non-2xx responses. If QStash is misconfigured
    (wrong token, wrong URL), every retry would also fail, creating a retry
    storm with no benefit. We log the failure and return 200, which means
    the event is "lost" from the queue — but it's preserved in GitHub's
    webhook delivery log, where it can be manually re-delivered once the
    config is fixed.
    """
    # ── Step 1: Read raw body (must happen before any parsing) ──────────────
    raw_body: bytes = await request.body()

    # ── Step 2: Validate GitHub signature ───────────────────────────────────
    signature = request.headers.get("X-Hub-Signature-256")
    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

    try:
        validate_signature(raw_body, signature, webhook_secret)
    except SignatureValidationError:
        raise HTTPException(status_code=403, detail="Invalid signature")

    # ── Step 3: Parse payload ────────────────────────────────────────────────
    try:
        payload: Dict[str, Any] = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        logger.warning("router: unparseable webhook payload: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid JSON payload")

    # ── Step 4: Filter to relevant events ───────────────────────────────────
    event_type = request.headers.get("X-GitHub-Event", "")

    if event_type != "pull_request":
        logger.debug("router: ignoring event type %r", event_type)
        return Response(content="ignored", status_code=200)

    action = payload.get("action", "")
    if action not in REVIEWABLE_ACTIONS:
        logger.debug("router: ignoring PR action %r", action)
        return Response(content="ignored", status_code=200)

    # ── Step 5: Extract PR metadata ─────────────────────────────────────────
    pr_meta = _extract_pr_metadata(payload)
    if pr_meta is None:
        logger.warning("router: could not extract PR metadata from payload")
        raise HTTPException(status_code=422, detail="Malformed pull_request payload")

    # ── Step 6: Idempotency check ────────────────────────────────────────────
    try:
        store = IdempotencyStore.from_env()
        if store.is_duplicate(
            pr_meta["repo_full_name"],
            pr_meta["pr_number"],
            pr_meta["head_sha"],
        ):
            logger.info(
                "router: duplicate event for %s PR #%d — returning 200 without enqueuing",
                pr_meta["repo_full_name"], pr_meta["pr_number"],
            )
            return Response(content="duplicate", status_code=200)
    except Exception as exc:
        # Redis is unavailable — fail open (proceed without idempotency)
        # rather than fail closed (reject the event). A duplicate review
        # is worse UX than a missing review, but a missed review entirely
        # is worse than a duplicate. Log prominently for alerting.
        logger.error(
            "router: idempotency check failed (%s) — proceeding without dedup",
            exc,
        )

    # ── Step 7: Enqueue for async processing ────────────────────────────────
    try:
        message_id = publish_pr_event(pr_meta)
        logger.info(
            "router: enqueued %s PR #%d sha=%s → QStash message %s",
            pr_meta["repo_full_name"], pr_meta["pr_number"],
            pr_meta["head_sha"][:8], message_id,
        )
    except (QueuePublishError, RuntimeError) as exc:
        # See design note in docstring — return 200 to avoid retry storm.
        logger.error(
            "router: failed to enqueue event for %s PR #%d: %s",
            pr_meta["repo_full_name"], pr_meta["pr_number"], exc,
        )
        return Response(content="enqueue_failed", status_code=200)

    return Response(content="accepted", status_code=200)


def _extract_pr_metadata(payload: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Pull the fields Lambda needs from a GitHub pull_request webhook payload.

    Returns None if any required field is missing or malformed — the router
    will return 422 in that case.

    Fields extracted:
      repo_full_name:   e.g. "ayushkaul/devmind"
      pr_number:        integer PR number
      head_sha:         commit SHA at the tip of the PR branch
      installation_id:  GitHub App installation ID (needed to generate
                        an access token for posting review comments)
      action:           "opened" | "synchronize" | "reopened"
    """
    try:
        pr = payload["pull_request"]
        return {
            "repo_full_name": payload["repository"]["full_name"],
            "pr_number": int(payload["number"]),
            "head_sha": pr["head"]["sha"],
            "installation_id": payload.get("installation", {}).get("id"),
            "action": payload["action"],
        }
    except (KeyError, TypeError, ValueError):
        return None
