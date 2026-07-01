"""
app/webhook/validator.py
─────────────────────────
GitHub webhook signature validation.

Why this matters:
Without signature validation, anyone on the internet who discovers your
webhook URL can POST arbitrary payloads to it, potentially triggering
code reviews on fake PRs, exhausting your Groq token budget, or worse —
injecting malicious diff content designed to manipulate the LLM's output.

How GitHub signs webhooks:
When you configure a webhook secret in GitHub App settings, GitHub computes
HMAC-SHA256(secret, raw_request_body) and sends the hex digest in the
X-Hub-Signature-256 header as "sha256=<hex>".

Your job: recompute the same HMAC with the same secret, compare the two
digests. If they match, the payload came from GitHub. If not, reject it.

Critical implementation detail — timing attacks:
A naive `computed == provided` string comparison leaks timing information:
Python's == exits early on the first mismatched character, so an attacker
can statistically measure response times to determine how many leading
characters of their forged signature are correct, eventually reconstructing
a valid one character by character.

`hmac.compare_digest()` is constant-time regardless of where the mismatch
occurs. Always use it for secret comparison. Never use == or !=.

References:
  https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
"""

from __future__ import annotations

import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Hub-Signature-256"
SIGNATURE_PREFIX = "sha256="


class SignatureValidationError(Exception):
    """
    Raised when a webhook request fails signature validation.

    Deliberately carries no detail about WHY validation failed — we don't
    want to leak information to an attacker about whether the header was
    missing, malformed, or just wrong. The HTTP layer converts this to a
    generic 403 response.
    """


def validate_signature(
    payload: bytes,
    signature_header: str | None,
    secret: str,
) -> None:
    """
    Validate a GitHub webhook request's HMAC-SHA256 signature.

    Args:
        payload:          The raw, unparsed request body bytes. Must be the
                          raw bytes — not a decoded string, not a parsed JSON
                          dict. GitHub signs the raw body before any parsing.
        signature_header: The value of the X-Hub-Signature-256 header, e.g.
                          "sha256=abc123...". None if the header was absent.
        secret:           The webhook secret configured in the GitHub App
                          settings. Must match exactly.

    Raises:
        SignatureValidationError: if the signature is missing, malformed,
                                  or does not match. No detail exposed.

    Returns:
        None on success (no return value — either succeeds or raises).
    """
    if not signature_header:
        logger.warning("validator: missing %s header", SIGNATURE_HEADER)
        raise SignatureValidationError("Missing signature header")

    if not signature_header.startswith(SIGNATURE_PREFIX):
        logger.warning("validator: malformed signature header (no sha256= prefix)")
        raise SignatureValidationError("Malformed signature header")

    provided_digest = signature_header[len(SIGNATURE_PREFIX):]

    computed_digest = _compute_digest(payload, secret)

    if not hmac.compare_digest(computed_digest, provided_digest):
        logger.warning("validator: signature mismatch — request rejected")
        raise SignatureValidationError("Signature mismatch")

    logger.debug("validator: signature valid")


def _compute_digest(payload: bytes, secret: str) -> str:
    """
    Compute HMAC-SHA256(secret, payload) and return the hex digest.

    Separated from validate_signature() so it can be tested independently
    without needing to construct a full signature header string.
    """
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
