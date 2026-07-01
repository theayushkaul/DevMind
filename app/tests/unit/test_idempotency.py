"""
tests/unit/test_idempotency.py
────────────────────────────────
Unit tests for app/webhook/idempotency.py.

Strategy: mock the Redis client at the constructor level so tests never
hit a real Redis instance. We test our logic around the client
(key construction, return value interpretation, fail-open behaviour),
not Redis itself.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.webhook.idempotency import (
    KEY_PREFIX,
    KEY_TTL_SECONDS,
    IdempotencyStore,
    _build_key,
)

REPO = "ayushkaul/devmind"
PR_NUMBER = 42
SHA = "abc123def456789"


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def store(mock_redis):
    return IdempotencyStore(client=mock_redis)


class TestBuildKey:

    def test_key_format(self):
        key = _build_key(REPO, PR_NUMBER, SHA)
        assert key == f"{KEY_PREFIX}:{REPO}:{PR_NUMBER}:{SHA}"

    def test_key_includes_all_three_components(self):
        key = _build_key(REPO, PR_NUMBER, SHA)
        assert REPO in key
        assert str(PR_NUMBER) in key
        assert SHA in key

    def test_different_sha_produces_different_key(self):
        key1 = _build_key(REPO, PR_NUMBER, "sha1")
        key2 = _build_key(REPO, PR_NUMBER, "sha2")
        assert key1 != key2

    def test_different_pr_produces_different_key(self):
        key1 = _build_key(REPO, 1, SHA)
        key2 = _build_key(REPO, 2, SHA)
        assert key1 != key2

    def test_different_repo_produces_different_key(self):
        key1 = _build_key("owner/repo-a", PR_NUMBER, SHA)
        key2 = _build_key("owner/repo-b", PR_NUMBER, SHA)
        assert key1 != key2


class TestIdempotencyStore:

    def test_first_event_is_not_duplicate(self, store, mock_redis):
        """SET NX returns True (key was set) → first time seen → not a duplicate."""
        mock_redis.set.return_value = True
        assert store.is_duplicate(REPO, PR_NUMBER, SHA) is False

    def test_second_event_is_duplicate(self, store, mock_redis):
        """SET NX returns False (key already existed) → duplicate."""
        mock_redis.set.return_value = False
        assert store.is_duplicate(REPO, PR_NUMBER, SHA) is True

    def test_calls_set_with_nx_flag(self, store, mock_redis):
        """Verify we're using SET NX, not GET then SET."""
        mock_redis.set.return_value = True
        store.is_duplicate(REPO, PR_NUMBER, SHA)

        mock_redis.set.assert_called_once()
        call_kwargs = mock_redis.set.call_args.kwargs
        assert call_kwargs["nx"] is True

    def test_calls_set_with_correct_ttl(self, store, mock_redis):
        mock_redis.set.return_value = True
        store.is_duplicate(REPO, PR_NUMBER, SHA)

        call_kwargs = mock_redis.set.call_args.kwargs
        assert call_kwargs["ex"] == KEY_TTL_SECONDS

    def test_key_passed_to_redis_is_correct(self, store, mock_redis):
        mock_redis.set.return_value = True
        store.is_duplicate(REPO, PR_NUMBER, SHA)

        expected_key = _build_key(REPO, PR_NUMBER, SHA)
        actual_key = mock_redis.set.call_args.args[0]
        assert actual_key == expected_key

    def test_clear_calls_redis_delete(self, store, mock_redis):
        store.clear(REPO, PR_NUMBER, SHA)

        expected_key = _build_key(REPO, PR_NUMBER, SHA)
        mock_redis.delete.assert_called_once_with(expected_key)

    def test_from_env_raises_when_no_redis_url(self, monkeypatch):
        monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(RuntimeError, match="REDIS_URL"):
            IdempotencyStore.from_env()
