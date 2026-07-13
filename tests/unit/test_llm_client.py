"""
tests/unit/test_llm_client.py
───────────────────────────────
Unit tests for app/llm/client.py.

Testing strategy: we mock at the Groq SDK boundary (client.chat.completions.create)
rather than mocking HTTP. This tests our retry/parsing logic against realistic
SDK objects without needing network access or a real API key.

We build minimal fake response objects that match the shape Groq's SDK
returns (response.choices[0].message.content, response.usage.prompt_tokens,
etc.) rather than importing Groq's actual response models — this keeps tests
fast and decoupled from SDK internals we don't control.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from groq import APIStatusError, APITimeoutError, RateLimitError

from app.llm.client import (
    LLMCallResult,
    _strip_markdown_fences,
    call_llm_json,
    get_client,
    reset_client,
)


# ---------------------------------------------------------------------------
# Fixtures — fake Groq response objects
# ---------------------------------------------------------------------------

def _make_fake_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """
    Build a minimal object matching the attribute shape of a Groq
    ChatCompletion response: response.choices[0].message.content,
    response.usage.prompt_tokens, response.usage.completion_tokens.
    """
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a clean client singleton."""
    reset_client()
    yield
    reset_client()


@pytest.fixture
def mock_groq_client(monkeypatch):
    """
    Patch get_client() to return a MagicMock standing in for the Groq client.
    Tests configure `mock_client.chat.completions.create` directly.
    """
    mock_client = MagicMock()
    monkeypatch.setattr("app.llm.client.get_client", lambda: mock_client)
    return mock_client


# ---------------------------------------------------------------------------
# Tests: get_client() — env var handling
# ---------------------------------------------------------------------------

class TestGetClient:

    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_client()

    def test_returns_client_when_api_key_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-testing")
        client = get_client()
        assert client is not None

    def test_singleton_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-testing")
        client1 = get_client()
        client2 = get_client()
        assert client1 is client2


# ---------------------------------------------------------------------------
# Tests: _strip_markdown_fences()
# ---------------------------------------------------------------------------

class TestStripMarkdownFences:

    def test_strips_json_fence(self):
        text = '```json\n{"findings": []}\n```'
        assert _strip_markdown_fences(text) == '{"findings": []}'

    def test_strips_plain_fence(self):
        text = '```\n{"findings": []}\n```'
        assert _strip_markdown_fences(text) == '{"findings": []}'

    def test_leaves_unfenced_json_untouched(self):
        text = '{"findings": []}'
        assert _strip_markdown_fences(text) == '{"findings": []}'

    def test_does_not_strip_internal_backticks(self):
        """
        A comment containing backticks (e.g. mentioning `inline code`) must
        not be mangled — only a fence wrapping the ENTIRE response is stripped.
        """
        text = '{"findings": [{"comment": "use `parameterized` queries"}]}'
        assert _strip_markdown_fences(text) == text

    def test_strips_fence_with_surrounding_whitespace(self):
        text = '  \n```json\n{"findings": []}\n```\n  '
        assert _strip_markdown_fences(text) == '{"findings": []}'


# ---------------------------------------------------------------------------
# Tests: call_llm_json() — success paths
# ---------------------------------------------------------------------------

class TestCallLlmJsonSuccess:

    def test_returns_parsed_json_on_success(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            '{"findings": [{"file_path": "a.py", "line_number": 1, '
            '"severity": "critical", "comment": "SQL injection"}]}'
        )

        result = call_llm_json("system prompt", "user content")

        assert result.success is True
        assert result.data == {
            "findings": [
                {"file_path": "a.py", "line_number": 1,
                 "severity": "critical", "comment": "SQL injection"}
            ]
        }
        assert result.error is None

    def test_strips_fence_before_parsing(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            '```json\n{"findings": []}\n```'
        )

        result = call_llm_json("system", "user")

        assert result.success is True
        assert result.data == {"findings": []}

    def test_tracks_token_usage(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            '{"findings": []}', prompt_tokens=250, completion_tokens=30,
        )

        result = call_llm_json("system", "user")

        assert result.prompt_tokens == 250
        assert result.completion_tokens == 30
        assert result.total_tokens == 280

    def test_passes_model_and_params_to_sdk(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            '{"findings": []}'
        )

        call_llm_json(
            "sys", "usr",
            model="custom-model",
            max_tokens=999,
            temperature=0.5,
        )

        call_kwargs = mock_groq_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["max_tokens"] == 999
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]


# ---------------------------------------------------------------------------
# Tests: call_llm_json() — malformed response handling
# ---------------------------------------------------------------------------

class TestCallLlmJsonMalformedResponses:

    def test_handles_invalid_json(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            "this is not json at all {{{"
        )

        result = call_llm_json("system", "user")

        assert result.success is False
        assert "JSONDecodeError" in result.error
        assert result.raw_response == "this is not json at all {{{"

    def test_handles_empty_response(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response("")

        result = call_llm_json("system", "user")

        assert result.success is False
        assert "empty" in result.error.lower()

    def test_handles_whitespace_only_response(self, mock_groq_client):
        mock_groq_client.chat.completions.create.return_value = _make_fake_response("   \n  ")

        result = call_llm_json("system", "user")

        assert result.success is False

    def test_handles_json_array_instead_of_object(self, mock_groq_client):
        """The model returns a bare JSON array instead of an object — invalid per our schema."""
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            '[{"file_path": "a.py"}]'
        )

        result = call_llm_json("system", "user")

        assert result.success is False
        assert "Expected a JSON object" in result.error

    def test_handles_no_choices_in_response(self, mock_groq_client):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=0)
        mock_groq_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[], usage=usage
        )

        result = call_llm_json("system", "user")

        assert result.success is False
        assert "no choices" in result.error.lower()

    def test_does_not_raise_on_malformed_json(self, mock_groq_client):
        """Critical: malformed LLM output must never propagate as an exception."""
        mock_groq_client.chat.completions.create.return_value = _make_fake_response(
            "garbage{{{not json"
        )
        try:
            result = call_llm_json("system", "user")
            assert isinstance(result, LLMCallResult)
        except Exception as exc:
            pytest.fail(f"call_llm_json raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tests: call_llm_json() — retry behaviour
# ---------------------------------------------------------------------------

class TestCallLlmJsonRetries:

    def test_retries_on_rate_limit_then_succeeds(self, mock_groq_client, monkeypatch):
        monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)  # skip real sleep

        fake_request = MagicMock()
        fake_429_response = MagicMock(status_code=429)
        rate_limit_error = RateLimitError(
            message="rate limited", response=fake_429_response, body=None,
        )

        mock_groq_client.chat.completions.create.side_effect = [
            rate_limit_error,
            _make_fake_response('{"findings": []}'),
        ]

        result = call_llm_json("system", "user")

        assert result.success is True
        assert mock_groq_client.chat.completions.create.call_count == 2

    def test_exhausts_retries_and_returns_failure(self, mock_groq_client, monkeypatch):
        monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)

        fake_response = MagicMock(status_code=429)
        rate_limit_error = RateLimitError(
            message="rate limited", response=fake_response, body=None,
        )
        mock_groq_client.chat.completions.create.side_effect = rate_limit_error

        result = call_llm_json("system", "user")

        assert result.success is False
        assert "Exhausted" in result.error

    def test_retries_on_timeout(self, mock_groq_client, monkeypatch):
        monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)

        timeout_error = APITimeoutError(request=MagicMock())
        mock_groq_client.chat.completions.create.side_effect = [
            timeout_error,
            _make_fake_response('{"findings": []}'),
        ]

        result = call_llm_json("system", "user")

        assert result.success is True
        assert mock_groq_client.chat.completions.create.call_count == 2

    def test_does_not_retry_on_400_bad_request(self, mock_groq_client, monkeypatch):
        """A 4xx (non-rate-limit) error indicates a malformed request — retrying won't help."""
        monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)

        fake_response = MagicMock(status_code=400)
        bad_request_error = APIStatusError(
            message="bad request", response=fake_response, body=None,
        )
        mock_groq_client.chat.completions.create.side_effect = bad_request_error

        result = call_llm_json("system", "user")

        assert result.success is False
        assert mock_groq_client.chat.completions.create.call_count == 1  # no retry

    def test_retries_on_500_server_error(self, mock_groq_client, monkeypatch):
        monkeypatch.setattr("app.llm.client.time.sleep", lambda _: None)

        fake_response = MagicMock(status_code=500)
        server_error = APIStatusError(
            message="server error", response=fake_response, body=None,
        )
        mock_groq_client.chat.completions.create.side_effect = [
            server_error,
            _make_fake_response('{"findings": []}'),
        ]

        result = call_llm_json("system", "user")

        assert result.success is True
        assert mock_groq_client.chat.completions.create.call_count == 2
