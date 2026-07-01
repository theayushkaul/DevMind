"""
app/webhook/queue.py
─────────────────────
QStash publisher: enqueues validated webhook events for async processing.

Why QStash over a raw Redis queue or direct Lambda invocation:

1. At-least-once delivery with automatic retry.
   QStash retries failed deliveries with exponential backoff. If Lambda
   times out or crashes mid-review, QStash retries automatically — no
   manual retry logic needed on our side.

2. Dead Letter Queue (DLQ) semantics.
   After a configurable number of failures, QStash stops retrying and
   the event can be inspected in the Upstash console. Our Supabase
   dlq_events table mirrors this for programmatic inspection.

3. Decoupling from Lambda cold start latency.
   If Lambda is cold, QStash buffers the event and retries. The webhook
   receiver returns 200 immediately regardless of Lambda's state.

4. Simple HTTP interface — no SDK required.
   QStash is just a REST API. We publish by POSTing to:
     POST https://qstash.upstash.io/v2/publish/<destination_url>
   with the event payload as the body and a Bearer token in the header.
   Using httpx directly keeps the dependency surface minimal and lets us
   mock at the HTTP boundary in tests.

Why httpx over requests:
   FastAPI is async-first. httpx provides both sync and async clients with
   an identical interface. We use the sync client here (the webhook handler
   is already in an async context but publish() is a single fast HTTP call
   that doesn't benefit from async in practice), keeping the code simple.
   If we needed to publish to multiple queues concurrently, we'd switch to
   httpx.AsyncClient and await gather().
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)

QSTASH_BASE_URL = "https://qstash.upstash.io/v2/publish"


class QueuePublishError(Exception):
    """Raised when the QStash API returns a non-2xx response."""


def publish_pr_event(payload: Dict[str, Any]) -> str:
    """
    Publish a PR event to QStash for async processing by Lambda.

    Args:
        payload: Dict containing the PR event data. Must include at minimum:
                 repo_full_name, pr_number, head_sha, installation_id.
                 QStash will forward this as the HTTP body to Lambda.

    Returns:
        The QStash message ID (useful for debugging and correlating logs).

    Raises:
        QueuePublishError: if QStash returns a non-2xx response.
        RuntimeError: if required environment variables are not set.
    """
    token = _require_env("QSTASH_TOKEN")
    lambda_url = _require_env("LAMBDA_FUNCTION_URL")

    destination = f"{QSTASH_BASE_URL}/{lambda_url}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        # Retry policy: QStash will retry up to 3 times with exponential backoff.
        # After 3 failures the message goes to the DLQ.
        "Upstash-Retries": "3",
        # Forward to Lambda as a POST — the Lambda handler expects POST.
        "Upstash-Method": "POST",
    }

    body = json.dumps(payload)

    try:
        response = httpx.post(
            destination,
            content=body,
            headers=headers,
            timeout=10.0,  # QStash API call should be fast — 10s is generous
        )
    except httpx.TimeoutException as exc:
        raise QueuePublishError(
            f"Timeout publishing to QStash: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise QueuePublishError(
            f"Network error publishing to QStash: {exc}"
        ) from exc

    if not response.is_success:
        raise QueuePublishError(
            f"QStash returned {response.status_code}: {response.text}"
        )

    message_id = response.json().get("messageId", "unknown")
    logger.info(
        "queue: published PR event for %s PR #%d — QStash message_id=%s",
        payload.get("repo_full_name"), payload.get("pr_number"), message_id,
    )
    return message_id


def _require_env(name: str) -> str:
    """Fetch a required environment variable or raise with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} environment variable is not set. "
            f"Set it in .env (local dev) or Lambda environment config (prod)."
        )
    return value
