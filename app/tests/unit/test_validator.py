"""
tests/unit/test_validator.py
─────────────────────────────
Unit tests for app/webhook/validator.py.

No mocking needed — HMAC computation is pure deterministic Python.
Tests run in microseconds.
"""

import hashlib
import hmac

import pytest

from app.webhook.validator import (
    SignatureValidationError,
    _compute_digest,
    validate_signature,
)

SECRET = "test-webhook-secret"
PAYLOAD = b'{"action": "opened", "number": 42}'


def _make_valid_signature(payload: bytes, secret: str) -> str:
    digest = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


class TestValidateSignature:

    def test_valid_signature_does_not_raise(self):
        sig = _make_valid_signature(PAYLOAD, SECRET)
        validate_signature(PAYLOAD, sig, SECRET)  # must not raise

    def test_missing_header_raises(self):
        with pytest.raises(SignatureValidationError):
            validate_signature(PAYLOAD, None, SECRET)

    def test_empty_header_raises(self):
        with pytest.raises(SignatureValidationError):
            validate_signature(PAYLOAD, "", SECRET)

    def test_missing_sha256_prefix_raises(self):
        digest = hmac.new(SECRET.encode(), PAYLOAD, hashlib.sha256).hexdigest()
        with pytest.raises(SignatureValidationError):
            validate_signature(PAYLOAD, digest, SECRET)  # no "sha256=" prefix

    def test_wrong_secret_raises(self):
        sig = _make_valid_signature(PAYLOAD, "wrong-secret")
        with pytest.raises(SignatureValidationError):
            validate_signature(PAYLOAD, sig, SECRET)

    def test_tampered_payload_raises(self):
        sig = _make_valid_signature(PAYLOAD, SECRET)
        tampered = PAYLOAD + b"extra"
        with pytest.raises(SignatureValidationError):
            validate_signature(tampered, sig, SECRET)

    def test_tampered_signature_raises(self):
        sig = _make_valid_signature(PAYLOAD, SECRET)
        bad_sig = sig[:-4] + "0000"  # corrupt last 4 chars
        with pytest.raises(SignatureValidationError):
            validate_signature(PAYLOAD, bad_sig, SECRET)

    def test_different_payloads_produce_different_signatures(self):
        payload_a = b'{"action": "opened"}'
        payload_b = b'{"action": "closed"}'
        sig_a = _make_valid_signature(payload_a, SECRET)
        sig_b = _make_valid_signature(payload_b, SECRET)
        assert sig_a != sig_b

    def test_empty_payload_with_valid_signature_passes(self):
        sig = _make_valid_signature(b"", SECRET)
        validate_signature(b"", sig, SECRET)  # must not raise

    def test_error_message_does_not_leak_details(self):
        """
        The exception message must be generic — no indication of whether
        the header was missing, malformed, or incorrect.
        """
        with pytest.raises(SignatureValidationError) as exc_info:
            validate_signature(PAYLOAD, None, SECRET)
        assert "key" not in str(exc_info.value).lower()
        assert "secret" not in str(exc_info.value).lower()


class TestComputeDigest:

    def test_matches_manual_hmac_computation(self):
        expected = hmac.new(
            key=SECRET.encode("utf-8"),
            msg=PAYLOAD,
            digestmod=hashlib.sha256,
        ).hexdigest()
        assert _compute_digest(PAYLOAD, SECRET) == expected

    def test_different_secrets_produce_different_digests(self):
        d1 = _compute_digest(PAYLOAD, "secret-1")
        d2 = _compute_digest(PAYLOAD, "secret-2")
        assert d1 != d2
