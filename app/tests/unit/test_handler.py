"""
tests/unit/test_handler.py
───────────────────────────
Unit tests for processor/handler.py.

Mocks four external boundaries:
  - _verify_qstash_signature  (skip crypto in unit tests)
  - get_github_client         (no real GitHub App credentials)
  - fetch_pr_diff             (no real GitHub API call)
  - run_pipeline              (no real LLM calls)
  - post_review               (no real GitHub API call)
  - asyncio.run               (no real DB calls — patched to a no-op)

DB isolation strategy:
  asyncio.run() is patched to return a fake review object for DB-creating
  calls, and None for fire-and-forget calls. This lets us verify that the
  handler calls the right DB operations at the right points without needing
  a real DB or async test infrastructure.
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import uuid

import pytest
import jwt

from processor.handler import (
    SignatureError,
    _response,
    _verify_qstash_signature,
    handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO = "ayushkaul/devmind"
PR_NUMBER = 7
HEAD_SHA = "abc123def456"
INSTALLATION_ID = 99999
GITHUB_REPO_ID = 12345678


def _make_event(payload: dict, signature: str = "valid-sig") -> dict:
    return {
        "body": json.dumps(payload),
        "headers": {"upstash-signature": signature},
    }


def _valid_payload(**overrides) -> dict:
    base = {
        "repo_full_name": REPO,
        "pr_number": PR_NUMBER,
        "head_sha": HEAD_SHA,
        "installation_id": INSTALLATION_ID,
        "github_repo_id": GITHUB_REPO_ID,
    }
    base.update(overrides)
    return base


def _fake_review():
    """Minimal fake review object returned by the DB create call."""
    review = MagicMock()
    review.id = uuid.uuid4()
    return review


def _pipeline_state(findings=None, tokens=180, error=None):
    from app.agent.state import ReviewFinding
    return {
        "final_comments": findings or [],
        "tokens_used": tokens,
        "error": error,
        "diff_chunks": [],
        "security_findings": [],
        "bug_findings": [],
        "style_findings": [],
    }


def _finding(severity="warning", comment="test issue", line_number=10):
    from app.agent.state import ReviewFinding
    return ReviewFinding(
        file_path="src/api.py",
        line_number=line_number,
        category="security",  # type: ignore
        severity=severity,    # type: ignore
        comment=comment,
        confidence=0.8,
        source_node="security_checker",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_signature_verify():
    with patch("processor.handler._verify_qstash_signature"):
        yield


@pytest.fixture
def mock_asyncio_run():
    """
    Patch asyncio.run so DB coroutines don't execute.
    First call (create_review_row) returns a fake review object.
    Subsequent calls (mark_processing, complete, etc.) return None.
    """
    fake_review = _fake_review()
    call_count = 0

    def side_effect(coro):
        nonlocal call_count
        call_count += 1
        try:
            asyncio.get_event_loop().run_until_complete(coro)
        except Exception:
            coro.close()  # noqa: close unawaited coroutine
        if call_count == 1:
            return fake_review  # first call = create_review_row
        return None

    with patch("processor.handler.asyncio.run", side_effect=side_effect) as m:
        yield m, fake_review


@pytest.fixture
def happy_path(mock_asyncio_run):
    """All external calls succeed — the full happy path."""
    _, fake_review = mock_asyncio_run
    with patch("processor.handler.get_github_client", return_value=MagicMock()):
        with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
            with patch("processor.handler.run_pipeline", return_value=_pipeline_state()):
                with patch("processor.handler.post_review", return_value=0):
                    yield fake_review


# ---------------------------------------------------------------------------
# Tests: signature verification
# ---------------------------------------------------------------------------

class TestSignatureVerification:

    def test_invalid_signature_returns_403(self):
        with patch(
            "processor.handler._verify_qstash_signature",
            side_effect=SignatureError("bad sig"),
        ):
            result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 403

    def test_valid_signature_proceeds(self, happy_path):
        result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 200


# ---------------------------------------------------------------------------
# Tests: payload parsing
# ---------------------------------------------------------------------------

class TestPayloadParsing:

    def test_invalid_json_returns_200_no_retry(self, mock_asyncio_run):
        event = {"body": "not json {{{", "headers": {"upstash-signature": "sig"}}
        result = handler(event, None)
        assert result["statusCode"] == 200
        assert "not retrying" in result["body"].lower()

    def test_missing_field_returns_200_no_retry(self, mock_asyncio_run):
        payload = {"repo_full_name": REPO}
        result = handler(_make_event(payload), None)
        assert result["statusCode"] == 200
        assert "not retrying" in result["body"].lower()


# ---------------------------------------------------------------------------
# Tests: DB integration
# ---------------------------------------------------------------------------

class TestDBIntegration:

    def test_asyncio_run_called_to_create_review_row(self, mock_asyncio_run, happy_path):
        handler(_make_event(_valid_payload()), None)
        mock_run, _ = mock_asyncio_run
        # First asyncio.run call = _create_review_row
        assert mock_run.call_count >= 1

    def test_db_failure_on_create_does_not_abort_handler(self, mock_asyncio_run):
        """DB unavailable at row-creation time → handler continues, returns 200."""
        mock_run, _ = mock_asyncio_run
        mock_run.side_effect = Exception("DB connection refused")

        with patch("processor.handler.get_github_client", return_value=MagicMock()):
            with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
                with patch("processor.handler.run_pipeline", return_value=_pipeline_state()):
                    with patch("processor.handler.post_review", return_value=0):
                        result = handler(_make_event(_valid_payload()), None)

        assert result["statusCode"] == 200

    def test_db_failure_on_complete_does_not_abort_handler(self):
        """DB unavailable at completion time → review was posted, return 200 anyway."""
        fake_review = _fake_review()
        call_count = 0

        def asyncio_side_effect(coro):
            nonlocal call_count
            call_count += 1
            coro.close()  # noqa: close unawaited coroutine
            if call_count == 1:
                return fake_review    # create_review_row succeeds
            elif call_count == 2:
                return None           # mark_processing succeeds
            else:
                raise Exception("DB write failed")  # complete_review fails

        with patch("processor.handler.asyncio.run", side_effect=asyncio_side_effect):
            with patch("processor.handler.get_github_client", return_value=MagicMock()):
                with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
                    with patch("processor.handler.run_pipeline", return_value=_pipeline_state()):
                        with patch("processor.handler.post_review", return_value=0):
                            result = handler(_make_event(_valid_payload()), None)

        assert result["statusCode"] == 200


# ---------------------------------------------------------------------------
# Tests: diff fetching
# ---------------------------------------------------------------------------

class TestDiffFetching:

    def test_diff_fetch_failure_returns_500(self, mock_asyncio_run):
        with patch("processor.handler.get_github_client", return_value=MagicMock()):
            with patch(
                "processor.handler.fetch_pr_diff",
                side_effect=Exception("GitHub rate limited"),
            ):
                result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 500


# ---------------------------------------------------------------------------
# Tests: pipeline execution
# ---------------------------------------------------------------------------

class TestPipelineExecution:

    def test_pipeline_failure_returns_500(self, mock_asyncio_run):
        with patch("processor.handler.get_github_client", return_value=MagicMock()):
            with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
                with patch(
                    "processor.handler.run_pipeline",
                    side_effect=Exception("Groq timeout"),
                ):
                    result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 500

    def test_pipeline_partial_error_still_returns_200(self, mock_asyncio_run):
        with patch("processor.handler.get_github_client", return_value=MagicMock()):
            with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
                with patch(
                    "processor.handler.run_pipeline",
                    return_value=_pipeline_state(error="SecurityChecker failed"),
                ):
                    with patch("processor.handler.post_review", return_value=0):
                        result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 200

    def test_no_findings_skips_post_review(self, mock_asyncio_run):
        with patch("processor.handler.get_github_client", return_value=MagicMock()):
            with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
                with patch(
                    "processor.handler.run_pipeline",
                    return_value=_pipeline_state(findings=[]),
                ):
                    with patch("processor.handler.post_review") as mock_post:
                        handler(_make_event(_valid_payload()), None)
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: review posting
# ---------------------------------------------------------------------------

class TestReviewPosting:

    def test_post_review_failure_returns_500(self, mock_asyncio_run):
        with patch("processor.handler.get_github_client", return_value=MagicMock()):
            with patch("processor.handler.fetch_pr_diff", return_value="diff --git"):
                with patch(
                    "processor.handler.run_pipeline",
                    return_value=_pipeline_state(findings=[_finding()]),
                ):
                    with patch(
                        "processor.handler.post_review",
                        side_effect=Exception("GitHub 503"),
                    ):
                        result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 500

    def test_success_response_contains_metrics(self, happy_path):
        result = handler(_make_event(_valid_payload()), None)
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["repo"] == REPO
        assert body["pr_number"] == PR_NUMBER
        assert "tokens_used" in body
        assert "latency_ms" in body
        assert "comments_posted" in body


# ---------------------------------------------------------------------------
# Tests: _verify_qstash_signature
# ---------------------------------------------------------------------------

class TestVerifyQstashSignature:

    def test_missing_signature_raises(self, monkeypatch):
        monkeypatch.setenv("QSTASH_CURRENT_SIGNING_KEY", "secret")
        with pytest.raises(SignatureError, match="Missing"):
            _verify_qstash_signature(None, "body")

    def test_missing_key_env_var_raises(self, monkeypatch):
        monkeypatch.delenv("QSTASH_CURRENT_SIGNING_KEY", raising=False)
        with pytest.raises(SignatureError, match="not configured"):
            _verify_qstash_signature("some-jwt", "body")

    def test_valid_jwt_passes(self, monkeypatch):
        secret = "test-signing-key"
        lambda_url = "https://lambda.example.com/handler"
        monkeypatch.setenv("QSTASH_CURRENT_SIGNING_KEY", secret)
        monkeypatch.setenv("QSTASH_NEXT_SIGNING_KEY", "")
        monkeypatch.setenv("LAMBDA_FUNCTION_URL", lambda_url)

        token = jwt.encode(
            {"iss": "Upstash", "sub": lambda_url, "exp": int(time.time()) + 300},
            secret, algorithm="HS256",
        )
        _verify_qstash_signature(token, "body")  # must not raise

    def test_expired_jwt_raises(self, monkeypatch):
        secret = "test-signing-key"
        monkeypatch.setenv("QSTASH_CURRENT_SIGNING_KEY", secret)
        monkeypatch.setenv("QSTASH_NEXT_SIGNING_KEY", "")

        token = jwt.encode(
            {"iss": "Upstash", "sub": "https://x.com", "exp": int(time.time()) - 60},
            secret, algorithm="HS256",
        )
        with pytest.raises(SignatureError, match="expired"):
            _verify_qstash_signature(token, "body")

    def test_falls_back_to_next_key(self, monkeypatch):
        current_key = "old-key"
        next_key = "new-key"
        lambda_url = "https://lambda.example.com/handler"
        monkeypatch.setenv("QSTASH_CURRENT_SIGNING_KEY", current_key)
        monkeypatch.setenv("QSTASH_NEXT_SIGNING_KEY", next_key)
        monkeypatch.setenv("LAMBDA_FUNCTION_URL", lambda_url)

        token = jwt.encode(
            {"iss": "Upstash", "sub": lambda_url, "exp": int(time.time()) + 300},
            next_key, algorithm="HS256",
        )
        _verify_qstash_signature(token, "body")  # must not raise


# ---------------------------------------------------------------------------
# Tests: _response
# ---------------------------------------------------------------------------

class TestResponseBuilder:

    def test_dict_body_serialised_to_json(self):
        r = _response(200, {"key": "value"})
        assert r["statusCode"] == 200
        assert json.loads(r["body"]) == {"key": "value"}

    def test_string_body_passed_through(self):
        r = _response(403, "Forbidden")
        assert r["body"] == "Forbidden"

    def test_content_type_header_present(self):
        r = _response(200, "ok")
        assert r["headers"]["Content-Type"] == "application/json"
