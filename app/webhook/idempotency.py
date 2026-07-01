"""
app/webhook/idempotency.py
───────────────────────────
Redis-backed idempotency for webhook deduplication.

Why this is non-optional:
GitHub guarantees at-least-once webhook delivery, not exactly-once. If your
webhook endpoint doesn't respond within 10 seconds, GitHub retries — up to
three times over several minutes. Without idempotency, a single PR open
event becomes three separate code reviews, posting duplicate comments and
burning three times the token budget.

The idempotency key:
    devmind:{repo_full_name}:{pr_number}:{head_sha}

All three components matter:
  - repo_full_name: isolates keys across repositories
  - pr_number:      isolates keys per PR within a repo
  - head_sha:       the commit SHA at the PR tip. This is the critical one.
                    When a developer pushes a new commit to an open PR,
                    GitHub fires another webhook with a DIFFERENT head_sha.
                    That IS a new event (the diff changed) and must trigger
                    a new review. If we keyed only on repo+pr_number, we'd
                    incorrectly skip reviews after every PR update.

TTL rationale:
24 hours. A PR that stays open for more than 24 hours without new commits
will not get re-reviewed by a re-delivered webhook — acceptable. Setting TTL
too short risks processing the same event twice if GitHub retries slowly;
too long wastes Redis memory. 24 hours is the standard industry choice for
webhook idempotency windows.

Redis command choice:
We use SET key value EX ttl NX — the NX flag means "only set if Not eXists".
This is a single atomic operation. The alternative (GET then SET) has a race
condition: two concurrent deliveries of the same event could both GET "not
found", both decide to proceed, and both enqueue the event. NX eliminates
that race at the Redis level.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import redis

logger = logging.getLogger(__name__)

KEY_PREFIX = "devmind"
KEY_TTL_SECONDS = 86_400  # 24 hours


class IdempotencyStore:
    """
    Thin wrapper around Redis for webhook idempotency checks.

    Kept as a class (rather than module-level functions) so it can be
    easily swapped for a mock in tests without monkeypatching global state.

    Usage:
        store = IdempotencyStore.from_env()
        if store.is_duplicate(repo, pr_number, head_sha):
            return  # already processing this event
        # ... enqueue event ...
    """

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "IdempotencyStore":
        """
        Construct from environment variables.

        Uses Upstash Redis REST URL format — the same Redis client works
        for both local Redis (REDIS_URL=redis://localhost:6379) and
        Upstash (REDIS_URL=rediss://...upstash.io:6379, with TLS).
        """
        url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("REDIS_URL")
        if not url:
            raise RuntimeError(
                "Neither UPSTASH_REDIS_REST_URL nor REDIS_URL is set. "
                "Set one in .env for local dev or Lambda environment for prod."
            )
        client = redis.from_url(url, decode_responses=True)
        return cls(client)

    def is_duplicate(
        self,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
    ) -> bool:
        """
        Check whether this event has already been seen and mark it if not.

        Returns:
            True  → duplicate, skip processing
            False → first time we've seen this event, safe to enqueue

        This is a single atomic Redis SET NX operation — no race condition
        between the check and the mark. See module docstring for details.
        """
        key = _build_key(repo_full_name, pr_number, head_sha)

        # SET key "1" EX ttl NX
        # Returns True  if the key was set (first time seen)
        # Returns False if the key already existed (duplicate)
        was_set: bool = self._client.set(
            key,
            "1",
            ex=KEY_TTL_SECONDS,
            nx=True,
        )

        if was_set:
            logger.info(
                "idempotency: new event for %s PR #%d sha=%s — marked and proceeding",
                repo_full_name, pr_number, head_sha[:8],
            )
            return False  # NOT a duplicate
        else:
            logger.info(
                "idempotency: duplicate event for %s PR #%d sha=%s — skipping",
                repo_full_name, pr_number, head_sha[:8],
            )
            return True  # IS a duplicate

    def clear(
        self,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
    ) -> None:
        """
        Remove the idempotency key. Used in tests and for manual retry of
        failed events from the DLQ without waiting for TTL expiry.
        """
        key = _build_key(repo_full_name, pr_number, head_sha)
        self._client.delete(key)
        logger.debug("idempotency: cleared key %s", key)


def _build_key(repo_full_name: str, pr_number: int, head_sha: str) -> str:
    """
    Build the Redis key for a given PR event.

    Format: devmind:{repo_full_name}:{pr_number}:{head_sha}
    Example: devmind:ayushkaul/devmind:42:abc123def456

    Kept as a module-level function (not a method) so test_idempotency.py
    can import and test the key format independently of the Redis client.
    """
    return f"{KEY_PREFIX}:{repo_full_name}:{pr_number}:{head_sha}"
