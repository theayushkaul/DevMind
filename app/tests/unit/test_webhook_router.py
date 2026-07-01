"""
tests/unit/test_webhook_router.py
────────────────────────────────────
Tests for app/webhook/router.py using FastAPI's TestClient.

TestClient wraps the ASGI app and sends real HTTP requests through the full
FastAPI middleware stack — routing, dependency injection, exception handlers —
without needing a running server. This is the right level to test the router:
we want to catch anything that breaks the HTTP contract (wrong status codes,
missing headers, unhandled exceptions becoming 500s).

Mocking strategy:
- validate_signature: patched to either pass (do nothing) or raise
  SignatureValidationError, without needing a real HMAC secret.
- IdempotencyStore: patched at the class level so from_env() returns a mock
  that we control (is_duplicate returns True or False as needed).
- publish_pr_event: patched to return a fake message_id or raise
  QueuePublishError, without hitting the real QStash API.

All three are patched at their import site in router.py, not at their
definition site — same principle as the graph tests.
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SECRET = "test-secret"
REPO = "ayushkaul/devmind"
PR_NUMBER = 7
HEAD_SHA = "abc123def456"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signature(payload: bytes, secret: str = SECRET) -> str:
    digest = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def _pr_payload(action: str = "opened") -> dict:
    return {
        "action": action,
        "number": PR_NUMBER,
        "repository": {"full_name": REPO},
        "pull_request": {
            "head": {"sha": HEAD_SHA},
        },
        "installation": {"id": 12345},
    }


def _post_webhook(
    payload: dict,
    event: str = "pull_request",
    signature: str | None = None,
    secret: str = SECRET,
) -> ...:
    body = json.dumps(payload).encode()
    sig = signature if signature is not None else _make_signature(body, secret)
    return client.post(
        "/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Event": event,
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    """Inject required env vars for every test."""
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "redis://localhost:6379")
    monkeypatch.setenv("QSTASH_TOKEN", "test-qstash-token")
    monkeypatch.setenv("LAMBDA_FUNCTION_URL", "https://lambda.example.com/handler")


@pytest.fixture
def mock_idempotency_pass():
    """Idempotency check passes — event is new."""
    mock_store = MagicMock()
    mock_store.is_duplicate.return_value = False
    with patch("app.webhook.router.IdempotencyStore") as mock_class:
        mock_class.from_env.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_idempotency_duplicate():
    """Idempotency check — event is a duplicate."""
    mock_store = MagicMock()
    mock_store.is_duplicate.return_value = True
    with patch("app.webhook.router.IdempotencyStore") as mock_class:
        mock_class.from_env.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_queue_success():
    with patch(
        "app.webhook.router.publish_pr_event",
        return_value="msg-id-abc123",
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Tests: signature validation
# ---------------------------------------------------------------------------

class TestSignatureValidation:

    def test_valid_signature_accepted(self, mock_idempotency_pass, mock_queue_success):
        resp = _post_webhook(_pr_payload())
        assert resp.status_code == 200

    def test_missing_signature_returns_403(self):
        payload = json.dumps(_pr_payload()).encode()
        resp = client.post(
            "/webhook",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "pull_request",
                # No X-Hub-Signature-256
            },
        )
        assert resp.status_code == 403

    def test_wrong_signature_returns_403(self):
        resp = _post_webhook(_pr_payload(), signature="sha256=wrongdigest")
        assert resp.status_code == 403

    def test_wrong_secret_returns_403(self):
        resp = _post_webhook(_pr_payload(), secret="wrong-secret")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tests: event filtering
# ---------------------------------------------------------------------------

class TestEventFiltering:

    def test_opened_pr_accepted(self, mock_idempotency_pass, mock_queue_success):
        resp = _post_webhook(_pr_payload(action="opened"))
        assert resp.status_code == 200
        assert resp.text == "accepted"

    def test_synchronize_pr_accepted(self, mock_idempotency_pass, mock_queue_success):
        resp = _post_webhook(_pr_payload(action="synchronize"))
        assert resp.status_code == 200
        assert resp.text == "accepted"

    def test_reopened_pr_accepted(self, mock_idempotency_pass, mock_queue_success):
        resp = _post_webhook(_pr_payload(action="reopened"))
        assert resp.status_code == 200
        assert resp.text == "accepted"

    def test_closed_pr_ignored(self):
        resp = _post_webhook(_pr_payload(action="closed"))
        assert resp.status_code == 200
        assert resp.text == "ignored"

    def test_labeled_pr_ignored(self):
        resp = _post_webhook(_pr_payload(action="labeled"))
        assert resp.status_code == 200
        assert resp.text == "ignored"

    def test_push_event_ignored(self, mock_idempotency_pass):
        resp = _post_webhook(_pr_payload(), event="push")
        assert resp.status_code == 200
        assert resp.text == "ignored"

    def test_issues_event_ignored(self, mock_idempotency_pass):
        resp = _post_webhook(_pr_payload(), event="issues")
        assert resp.status_code == 200
        assert resp.text == "ignored"


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_duplicate_event_returns_200_without_enqueuing(
        self, mock_idempotency_duplicate, mock_queue_success
    ):
        resp = _post_webhook(_pr_payload())
        assert resp.status_code == 200
        assert resp.text == "duplicate"
        mock_queue_success.assert_not_called()

    def test_new_event_is_enqueued(self, mock_idempotency_pass, mock_queue_success):
        resp = _post_webhook(_pr_payload())
        assert resp.status_code == 200
        mock_queue_success.assert_called_once()

    def test_redis_failure_proceeds_without_dedup(self, mock_queue_success):
        """
        Redis unavailable → fail open (proceed without idempotency) rather
        than fail closed (drop the event). Returns 200 with "accepted".
        """
        with patch("app.webhook.router.IdempotencyStore") as mock_class:
            mock_class.from_env.side_effect = Exception("Redis connection refused")
            resp = _post_webhook(_pr_payload())

        assert resp.status_code == 200
        assert resp.text == "accepted"


# ---------------------------------------------------------------------------
# Tests: queue publishing
# ---------------------------------------------------------------------------

class TestQueuePublishing:

    def test_enqueue_failure_still_returns_200(self, mock_idempotency_pass):
        """
        QStash unavailable → return 200 to prevent GitHub retry storm.
        See router.py docstring for the full rationale.
        """
        from app.webhook.queue import QueuePublishError
        with patch(
            "app.webhook.router.publish_pr_event",
            side_effect=QueuePublishError("QStash 503"),
        ):
            resp = _post_webhook(_pr_payload())

        assert resp.status_code == 200
        assert resp.text == "enqueue_failed"

    def test_published_payload_contains_required_fields(
        self, mock_idempotency_pass, mock_queue_success
    ):
        """The payload forwarded to Lambda must contain all fields it needs."""
        _post_webhook(_pr_payload(action="opened"))

        published = mock_queue_success.call_args.args[0]
        assert published["repo_full_name"] == REPO
        assert published["pr_number"] == PR_NUMBER
        assert published["head_sha"] == HEAD_SHA
        assert published["action"] == "opened"


# ---------------------------------------------------------------------------
# Tests: payload validation
# ---------------------------------------------------------------------------

class TestPayloadValidation:

    def test_malformed_json_returns_422(self):
        body = b"not json {"
        sig = _make_signature(body)
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
            },
        )
        assert resp.status_code == 422

    def test_missing_pull_request_key_returns_422(self):
        payload = {"action": "opened", "number": 1, "repository": {"full_name": REPO}}
        # Missing "pull_request" key entirely
        body = json.dumps(payload).encode()
        sig = _make_signature(body)
        resp = client.post(
            "/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
            },
        )
        assert resp.status_code == 422

    def test_health_endpoint_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
