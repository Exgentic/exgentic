# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for LiteLLM health check error handling."""

from __future__ import annotations

import logging
import socket
import ssl
import urllib.error
from unittest.mock import AsyncMock, patch

import pytest
from exgentic.core.types.model_settings import ModelSettings, RetryStrategy
from exgentic.integrations.litellm.health import (
    _HEALTH_MAX_RETRY_DELAY,
    _HEALTH_MIN_RETRIES,
    _HEALTH_MIN_RETRY_DELAY,
    _MODELS_PROBE_TIMEOUT,
    ErrorCategory,
    HealthCheckError,
    _describe_fetch_failure,
    _models_url,
    _probe_timeout,
    _served_model_ids,
    acheck_model_accessible,
    acheck_model_reachable,
    check_model_accessible_sync,
    check_models_endpoint,
    classify_error,
    reset_probe_memo,
    validate_model_environment,
)


@pytest.fixture(autouse=True)
def _clear_probe_memo():
    """The probe memo is process-global; keep it from leaking between tests."""
    reset_probe_memo()
    yield
    reset_probe_memo()


class MockLiteLLMError(Exception):
    """Mock exception that mimics LiteLLM exceptions with .message attribute."""

    def __init__(self, message: str):
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        """Return empty string to simulate LiteLLM exceptions that don't implement __str__."""
        return ""


def _raise_and_close(exc: BaseException):
    """Build a ``run_sync`` side effect that raises *exc* without leaking the coroutine.

    ``run_sync`` is handed a coroutine; a plain ``side_effect`` exception never
    awaits it, which surfaces later as an unraisable "coroutine was never
    awaited" RuntimeWarning in an unrelated test.
    """

    def _side_effect(coro, *args, **kwargs):
        close = getattr(coro, "close", None)
        if close is not None:
            close()
        raise exc

    return _side_effect


def _skip_cheap_layers():
    """Stub layers 1 and 2 so a test can isolate the strict completion call."""
    return (
        patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]),
        patch("exgentic.integrations.litellm.health.check_models_endpoint", return_value=None),
    )


def test_health_check_extracts_message_attribute_from_exception(caplog):
    """Test that health check extracts error details from exception.message attribute."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.utils.sync.run_sync") as mock_run_sync:
        exc = MockLiteLLMError("API key authentication failed")
        mock_run_sync.side_effect = _raise_and_close(exc)

        logger = logging.getLogger("test")

        env, endpoint = _skip_cheap_layers()
        with env, endpoint, pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("test-model", logger, strict=True)

        error_msg = str(exc_info.value)
        assert "API key authentication failed" in error_msg
        assert "test-model" in error_msg
        assert error_msg == "Model test-model is not accessible: API key authentication failed"


def test_health_check_falls_back_to_str_when_no_message_attribute(caplog):
    """Test that health check falls back to str(exc) when .message is not available."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.utils.sync.run_sync") as mock_run_sync:
        exc = ValueError("Standard error message")
        mock_run_sync.side_effect = _raise_and_close(exc)

        logger = logging.getLogger("test")

        env, endpoint = _skip_cheap_layers()
        with env, endpoint, pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("test-model", logger, strict=True)

        error_msg = str(exc_info.value)
        assert "Standard error message" in error_msg
        assert "test-model" in error_msg


def test_health_check_uses_repr_as_last_resort(caplog):
    """Test that health check uses repr(exc) when both .message and str(exc) are empty."""
    caplog.set_level(logging.ERROR)

    class EmptyError(Exception):
        """Exception that returns empty string from __str__."""

        def __str__(self) -> str:
            return ""

    with patch("exgentic.utils.sync.run_sync") as mock_run_sync:
        exc = EmptyError("hidden")
        mock_run_sync.side_effect = _raise_and_close(exc)

        logger = logging.getLogger("test")

        env, endpoint = _skip_cheap_layers()
        with env, endpoint, pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("test-model", logger, strict=True)

        error_msg = str(exc_info.value)
        assert "EmptyError" in error_msg
        assert "test-model" in error_msg


# ---------------------------------------------------------------------------
# Rate-limit detection helper
# ---------------------------------------------------------------------------


class _FakeRateLimitError(Exception):
    def __init__(self):
        self.status_code = 429
        super().__init__("rate limited")


class _FakeWrappedRateLimitError(Exception):
    def __init__(self):
        inner = Exception("inner")
        inner.status_code = 429  # type: ignore[attr-defined]
        self.original_exception = inner
        super().__init__("wrapped rate limit")


def test_is_transient_error_direct_status_code():
    assert classify_error(_FakeRateLimitError()) == ErrorCategory.TRANSIENT


def test_is_transient_error_wrapped():
    assert classify_error(_FakeWrappedRateLimitError()) == ErrorCategory.TRANSIENT


def test_is_transient_error_not_rate_limit():
    assert classify_error(ValueError("nope")) == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Async retry logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acheck_retries_on_rate_limit_then_succeeds():
    """Health check should retry on 429 and succeed when the next attempt works."""
    mock = AsyncMock(side_effect=[_FakeRateLimitError(), None])
    settings = ModelSettings(num_retries=2, retry_after=0.01)

    with patch("litellm.acompletion", mock):
        await acheck_model_accessible("m", model_settings=settings)

    assert mock.call_count == 2


@pytest.mark.asyncio
async def test_acheck_raises_after_exhausting_retries():
    """Health check should raise after all retries are exhausted."""
    exc = _FakeRateLimitError()
    mock = AsyncMock(side_effect=exc)
    # Use num_retries above the health minimum so the test controls the count.
    retries = _HEALTH_MIN_RETRIES + 1
    settings = ModelSettings(num_retries=retries, retry_after=0.01)

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(type(exc)):
                await acheck_model_accessible("m", model_settings=settings)

    # 1 initial + retries = retries + 1 calls
    assert mock.call_count == retries + 1


@pytest.mark.asyncio
async def test_acheck_does_not_retry_non_rate_limit_errors():
    """Non-rate-limit errors should propagate immediately without retry."""
    mock = AsyncMock(side_effect=ValueError("bad key"))
    settings = ModelSettings(num_retries=3, retry_after=0.01)

    with patch("litellm.acompletion", mock):
        with pytest.raises(ValueError, match="bad key"):
            await acheck_model_accessible("m", model_settings=settings)

    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_acheck_uses_constant_retry_strategy():
    """Health check should use constant delay when retry_strategy is CONSTANT."""
    mock = AsyncMock(side_effect=[_FakeRateLimitError(), _FakeRateLimitError(), None])
    # Use retry_after above the health minimum so the test controls the delay.
    delay = _HEALTH_MIN_RETRY_DELAY + 1.0
    settings = ModelSettings(
        num_retries=_HEALTH_MIN_RETRIES + 1,
        retry_after=delay,
        retry_strategy=RetryStrategy.CONSTANT,
    )

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await acheck_model_accessible("m", model_settings=settings)

    # Constant strategy: delay should always be retry_after
    assert sleep_mock.call_count == 2
    for call in sleep_mock.call_args_list:
        assert call.args[0] == pytest.approx(delay)


@pytest.mark.asyncio
async def test_acheck_defaults_to_model_settings_when_none():
    """When model_settings is None, ModelSettings() defaults are used."""
    mock = AsyncMock(side_effect=[_FakeRateLimitError(), None])

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await acheck_model_accessible("m")

    # Default ModelSettings has num_retries=5, so retry should happen
    assert mock.call_count == 2


# ---------------------------------------------------------------------------
# Transient error detection
# ---------------------------------------------------------------------------


def test_is_transient_error_timeout():
    assert classify_error(TimeoutError()) == ErrorCategory.TRANSIENT


def test_is_transient_error_connection():
    assert classify_error(ConnectionError()) == ErrorCategory.TRANSIENT


def test_is_transient_error_os_error():
    assert classify_error(OSError("network down")) == ErrorCategory.TRANSIENT


def test_is_transient_error_wrapped_timeout():
    exc = Exception("outer")
    exc.original_exception = TimeoutError()  # type: ignore[attr-defined]
    assert classify_error(exc) == ErrorCategory.TRANSIENT


def test_is_transient_error_non_transient():
    assert classify_error(ValueError("bad input")) == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Health check minimum retry guarantees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acheck_enforces_minimum_retries():
    """Health check enforces minimum retry count.

    Even if model_settings specifies fewer retries, the health check
    should retry at least _HEALTH_MIN_RETRIES times.
    """
    settings = ModelSettings(num_retries=1, retry_after=0.01)

    side_effects = [TimeoutError()] * _HEALTH_MIN_RETRIES + [None]
    mock = AsyncMock(side_effect=side_effects)

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await acheck_model_accessible("m", model_settings=settings)

    assert mock.call_count == _HEALTH_MIN_RETRIES + 1


@pytest.mark.asyncio
async def test_acheck_enforces_minimum_retry_delay():
    """Health check enforces minimum retry delay.

    Even if model_settings specifies a shorter delay, the health check
    should use at least _HEALTH_MIN_RETRY_DELAY.
    """
    settings = ModelSettings(num_retries=1, retry_after=0.001)
    mock = AsyncMock(side_effect=[TimeoutError(), None])

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await acheck_model_accessible("m", model_settings=settings)

    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args.args[0] >= _HEALTH_MIN_RETRY_DELAY


@pytest.mark.asyncio
async def test_acheck_caps_retry_delay_at_maximum():
    """Exponential backoff should be capped at _HEALTH_MAX_RETRY_DELAY."""
    settings = ModelSettings(num_retries=1, retry_after=100.0)

    side_effects = [TimeoutError()] * 3 + [None]
    mock = AsyncMock(side_effect=side_effects)

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await acheck_model_accessible("m", model_settings=settings)

    for call in sleep_mock.call_args_list:
        assert call.args[0] <= _HEALTH_MAX_RETRY_DELAY


@pytest.mark.asyncio
async def test_acheck_timeout_retries_then_succeeds():
    """Health check should retry on TimeoutError and succeed when endpoint recovers."""
    side_effects = [TimeoutError(), TimeoutError(), TimeoutError(), None]
    mock = AsyncMock(side_effect=side_effects)

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await acheck_model_accessible("m")

    assert mock.call_count == 4


@pytest.mark.asyncio
async def test_acheck_connection_error_retries():
    """Health check should retry on ConnectionError."""
    mock = AsyncMock(side_effect=[ConnectionError(), None])

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await acheck_model_accessible("m")

    assert mock.call_count == 2


# ---------------------------------------------------------------------------
# Model-reachable errors (health check should PASS)
# ---------------------------------------------------------------------------


class _FakeContentPolicyError(Exception):
    """Simulates a content policy violation (model rejected the prompt)."""

    def __init__(self):
        self.status_code = 400
        super().__init__("content policy violation")


class _FakeContentPolicyViolationError(Exception):
    """Simulates litellm.ContentPolicyViolationError by type name."""

    def __init__(self):
        self.status_code = 400
        super().__init__("content filtered")


# Give the class the name litellm uses.
_FakeContentPolicyViolationError.__name__ = "ContentPolicyViolationError"


def test_is_model_reachable_400():
    """400 Bad Request means the model backend is alive."""
    assert classify_error(_FakeContentPolicyError()) == ErrorCategory.REACHABLE


def test_is_model_reachable_content_policy_by_name():
    """ContentPolicyViolationError detected by type name."""
    assert classify_error(_FakeContentPolicyViolationError()) == ErrorCategory.REACHABLE


def test_is_model_reachable_not_reachable():
    """Timeouts and connection errors are NOT model-reachable."""
    assert classify_error(TimeoutError()) == ErrorCategory.TRANSIENT
    assert classify_error(ConnectionError()) == ErrorCategory.TRANSIENT


@pytest.mark.asyncio
async def test_acheck_passes_on_content_policy_error():
    """Health check should PASS when model returns a content policy error.

    A content policy error means the request reached the model and got
    a response — the endpoint is live.
    """
    mock = AsyncMock(side_effect=_FakeContentPolicyViolationError())

    with patch("litellm.acompletion", mock):
        await acheck_model_accessible("m")

    # Should succeed on first call, no retries.
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_acheck_passes_on_400_bad_request():
    """Health check should PASS on 400 — the model backend is reachable."""
    mock = AsyncMock(side_effect=_FakeContentPolicyError())

    with patch("litellm.acompletion", mock):
        await acheck_model_accessible("m")

    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Permanent errors (should NOT retry)
# ---------------------------------------------------------------------------


class _FakeAuthError(Exception):
    def __init__(self):
        self.status_code = 401
        super().__init__("invalid api key")


class _FakeNotFoundError(Exception):
    def __init__(self):
        self.status_code = 404
        super().__init__("model not found")


def test_is_permanent_error_401():
    assert classify_error(_FakeAuthError()) == ErrorCategory.PERMANENT


def test_is_permanent_error_404():
    assert classify_error(_FakeNotFoundError()) == ErrorCategory.PERMANENT


def test_is_permanent_error_not_permanent():
    """Transient errors should NOT be permanent."""
    assert classify_error(TimeoutError()) == ErrorCategory.TRANSIENT
    assert classify_error(_FakeRateLimitError()) == ErrorCategory.TRANSIENT


@pytest.mark.asyncio
async def test_acheck_does_not_retry_auth_error():
    """401 auth errors should fail immediately without retry."""
    mock = AsyncMock(side_effect=_FakeAuthError())

    with patch("litellm.acompletion", mock):
        with pytest.raises(_FakeAuthError):
            await acheck_model_accessible("m")

    # Should fail on first call, no retries.
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_acheck_does_not_retry_not_found_error():
    """404 not-found errors should fail immediately without retry."""
    mock = AsyncMock(side_effect=_FakeNotFoundError())

    with patch("litellm.acompletion", mock):
        with pytest.raises(_FakeNotFoundError):
            await acheck_model_accessible("m")

    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Layer 1: environment validation
# ---------------------------------------------------------------------------


def test_validate_model_environment_reports_missing_keys(monkeypatch):
    """Missing provider credentials are reported by name.

    Clears the gateway env vars so an ambient OPENAI_API_BASE/KEY in the
    developer's shell cannot engage the gateway-alias shortcut.
    """
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = {"keys_in_environment": False, "missing_keys": ["AZURE_API_KEY"]}
    with patch("litellm.validate_environment", return_value=env):
        assert validate_model_environment("azure/gpt-4.1") == ["AZURE_API_KEY"]


def test_validate_model_environment_empty_when_complete():
    """A fully configured environment reports nothing missing."""
    with patch("litellm.validate_environment", return_value={"keys_in_environment": True, "missing_keys": []}):
        assert validate_model_environment("gpt-4o") == []


def test_validate_model_environment_tolerates_unknown_model():
    """An unknown/aliased model reports nothing rather than raising."""
    with patch("litellm.validate_environment", side_effect=Exception("unknown model")):
        assert validate_model_environment("my-gateway-alias") == []


def test_sync_check_fails_fast_on_missing_credentials():
    """Layer 1 failure short-circuits before any network probe."""
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=["OPENAI_API_KEY"]):
        with patch("exgentic.integrations.litellm.health.check_models_endpoint") as probe:
            with pytest.raises(HealthCheckError) as exc_info:
                check_model_accessible_sync("gpt-4o", logger)
    assert "OPENAI_API_KEY" in str(exc_info.value)
    probe.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2: GET /v1/models reachability
# ---------------------------------------------------------------------------


def test_models_url_appends_v1_when_absent():
    assert _models_url("https://gw.example.com") == "https://gw.example.com/v1/models"


def test_models_url_reuses_existing_v1_suffix():
    assert _models_url("https://gw.example.com/v1") == "https://gw.example.com/v1/models"
    assert _models_url("https://gw.example.com/v1/") == "https://gw.example.com/v1/models"


def test_served_model_ids_parses_openai_listing():
    payload = {"data": [{"id": "gpt-4o"}, {"id": "gpt-5-mini"}, {"no_id": 1}]}
    assert _served_model_ids(payload) == {"gpt-4o", "gpt-5-mini"}


def test_served_model_ids_tolerates_unexpected_shapes():
    assert _served_model_ids(None) == set()
    assert _served_model_ids({"data": "nope"}) == set()
    assert _served_model_ids([]) == set()


def _patch_probe(status, payload=None, base="https://gw.example.com/v1"):
    """Patch base-URL resolution and the HTTP fetch for endpoint tests."""
    return (
        patch("exgentic.integrations.litellm.health._resolve_api_base", return_value=base),
        patch("exgentic.integrations.litellm.health._fetch_models", return_value=(status, payload)),
    )


def test_endpoint_check_passes_when_model_listed():
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(200, {"data": [{"id": "gpt-4o"}]})
    with base, fetch:
        check_models_endpoint("gpt-4o", logger)  # no raise


def test_endpoint_check_warns_when_model_absent_from_listing(caplog):
    """A gateway may alias or hide names — warn, don't fail."""
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(200, {"data": [{"id": "some-other-model"}]})
    with base, fetch:
        check_models_endpoint("gpt-4o", logger)  # no raise
    assert "not listed" in caplog.text


def test_endpoint_check_treats_auth_rejection_as_reachable():
    """401/403 means the endpoint is alive; auth surfaces on the real call."""
    logger = logging.getLogger("test")
    for status in (401, 403):
        base, fetch = _patch_probe(status)
        with base, fetch:
            check_models_endpoint("gpt-4o", logger)  # no raise


def test_endpoint_check_raises_on_server_error():
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(503)
    with base, fetch:
        with pytest.raises(HealthCheckError) as exc_info:
            check_models_endpoint("gpt-4o", logger)
    assert "503" in str(exc_info.value)


def test_endpoint_check_raises_when_unreachable():
    """status=None means DNS/TCP/TLS failure, not an HTTP error."""
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(None, "connection refused: [Errno 111]")
    with base, fetch:
        with pytest.raises(HealthCheckError) as exc_info:
            check_models_endpoint("gpt-4o", logger)
    message = str(exc_info.value)
    assert "did not respond" in message
    # The reported cause must be the one observed, not a guess at "unreachable".
    assert "connection refused" in message


def test_endpoint_check_tolerates_missing_models_route():
    """Some gateways omit /v1/models; a 404 still proves the endpoint answered."""
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(404)
    with base, fetch:
        check_models_endpoint("gpt-4o", logger)  # no raise


def test_endpoint_check_skipped_when_no_base_url_discoverable():
    """Guessing a base URL would cause false failures for self-hosted providers."""
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health._resolve_api_base", return_value=None):
        with patch("exgentic.integrations.litellm.health._fetch_models") as fetch:
            check_models_endpoint("some/self-hosted", logger)  # no raise
    fetch.assert_not_called()


def test_default_sync_check_issues_no_completion_call():
    """The whole point: the default path must not bill a token."""
    logger = logging.getLogger("test")
    with patch("litellm.acompletion", new=AsyncMock()) as completion:
        with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]):
            base, fetch = _patch_probe(200, {"data": [{"id": "gpt-4o"}]})
            with base, fetch:
                check_model_accessible_sync("gpt-4o", logger)
    completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_reachable_check_issues_no_completion_call():
    logger = logging.getLogger("test")
    with patch("litellm.acompletion", new=AsyncMock()) as completion:
        with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]):
            base, fetch = _patch_probe(200, {"data": [{"id": "gpt-4o"}]})
            with base, fetch:
                await acheck_model_reachable("gpt-4o", logger)
    completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_reachable_check_raises_on_missing_credentials():
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=["OPENAI_API_KEY"]):
        with pytest.raises(HealthCheckError) as exc_info:
            await acheck_model_reachable("gpt-4o", logger)
    assert "OPENAI_API_KEY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Layer 1: provider-prefixed aliases behind an OpenAI-compatible gateway
# ---------------------------------------------------------------------------


def test_gateway_alias_does_not_require_provider_credentials(monkeypatch):
    """``azure/x`` served by a gateway must not demand AZURE_API_* credentials.

    LiteLLM gateways commonly namespace aliases with provider-shaped prefixes
    (``azure/``, ``aws/``, ``gcp/``). The prefix is part of the alias, not a
    routing decision, so requiring that provider's env vars would reject models
    the gateway actually serves.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-gateway")
    env = {"keys_in_environment": False, "missing_keys": ["AZURE_API_BASE", "AZURE_API_KEY"]}
    with patch("litellm.validate_environment", return_value=env):
        assert validate_model_environment("azure/gpt-5-mini") == []


def test_provider_credentials_still_required_without_gateway(monkeypatch):
    """Without a gateway configured, a real provider route must still be validated."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = {"keys_in_environment": False, "missing_keys": ["AZURE_API_BASE", "AZURE_API_KEY"]}
    with patch("litellm.validate_environment", return_value=env):
        assert validate_model_environment("azure/gpt-4.1") == ["AZURE_API_BASE", "AZURE_API_KEY"]


def test_missing_gateway_key_is_still_reported(monkeypatch):
    """A base URL without a key is a real misconfiguration, not a gateway alias."""
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = {"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]}
    with patch("litellm.validate_environment", return_value=env):
        assert validate_model_environment("gpt-4o") == ["OPENAI_API_KEY"]


def test_gateway_shortcut_requires_both_base_and_key(monkeypatch):
    """The shortcut needs a key too; a bare base URL must not suppress validation."""
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = {"keys_in_environment": False, "missing_keys": ["AZURE_API_KEY"]}
    with patch("litellm.validate_environment", return_value=env):
        assert validate_model_environment("azure/gpt-5-mini") == ["AZURE_API_KEY"]


# ---------------------------------------------------------------------------
# Probe resilience: retry, memo, timeout, opt-out, error detail
#
# Regression cover for the per-task probe failing whole tasks on a slow first
# contact. Measured in the field: a 10.09 s stall whose immediate retry
# returned in 0.18 s, and 7 of 7 slow attempts followed by a sub-quarter-second
# success. The probe is a diagnostic and must not be the thing that fails a run.
# ---------------------------------------------------------------------------


def test_transport_failure_is_retried_and_recovers():
    """A cold first contact must not fail the task when the retry succeeds."""
    logger = logging.getLogger("test")
    attempts = [(None, "timed out after 10s"), (200, {"data": [{"id": "gpt-4o"}]})]
    with patch("exgentic.integrations.litellm.health._resolve_api_base", return_value="https://gw.example.com/v1"):
        with patch("exgentic.integrations.litellm.health._fetch_models", side_effect=attempts) as fetch:
            with patch("exgentic.integrations.litellm.health.time.sleep"):
                check_models_endpoint("gpt-4o", logger)  # no raise
    assert fetch.call_count == 2


def test_transport_failure_raises_after_exhausting_attempts():
    """A genuinely dead endpoint still fails, and says how many tries it got."""
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(None, "connection refused: [Errno 111]")
    with base, fetch, patch("exgentic.integrations.litellm.health.time.sleep"):
        with pytest.raises(HealthCheckError) as exc_info:
            check_models_endpoint("gpt-4o", logger)
    assert "2 attempt(s)" in str(exc_info.value)


def test_http_response_is_not_retried():
    """5xx is an answer, not a stall: retrying cannot change the verdict."""
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(503)
    with base, fetch as fetch_mock:
        with pytest.raises(HealthCheckError):
            check_models_endpoint("gpt-4o", logger)
    assert fetch_mock.call_count == 1


def test_probe_runs_once_per_process_across_tasks():
    """Agent instances are per task; the endpoint is not. Probe once."""
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]):
        with patch("exgentic.integrations.litellm.health.check_models_endpoint") as probe:
            for _ in range(5):
                check_model_accessible_sync("gpt-4o", logger)
    assert probe.call_count == 1


def test_probe_memo_can_be_forced():
    """An explicit re-check must still be possible."""
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]):
        with patch("exgentic.integrations.litellm.health.check_models_endpoint") as probe:
            check_model_accessible_sync("gpt-4o", logger)
            check_model_accessible_sync("gpt-4o", logger, force=True)
    assert probe.call_count == 2


def test_probe_failure_is_not_memoised():
    """A broken endpoint must be reported for every task, not just the first."""
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]):
        with patch(
            "exgentic.integrations.litellm.health.check_models_endpoint",
            side_effect=HealthCheckError("down"),
        ) as probe:
            for _ in range(3):
                with pytest.raises(HealthCheckError):
                    check_model_accessible_sync("gpt-4o", logger)
    assert probe.call_count == 3


def test_caller_timeout_is_honoured_not_clamped():
    """A caller asking for 30s used to be silently reduced to 10s."""
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=[]):
        with patch("exgentic.integrations.litellm.health.check_models_endpoint") as probe:
            check_model_accessible_sync("gpt-4o", logger, timeout=30.0)
    assert probe.call_args.kwargs["timeout"] == 30.0


def test_probe_timeout_reads_environment(monkeypatch):
    monkeypatch.setenv("EXGENTIC_MODEL_PROBE_TIMEOUT", "45")
    assert _probe_timeout() == 45.0
    # An explicit argument still wins over the environment.
    assert _probe_timeout(5.0) == 5.0


@pytest.mark.parametrize("raw", ["", "garbage", "0", "-1"])
def test_probe_timeout_falls_back_on_unusable_environment(monkeypatch, raw):
    monkeypatch.setenv("EXGENTIC_MODEL_PROBE_TIMEOUT", raw)
    assert _probe_timeout() == _MODELS_PROBE_TIMEOUT


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_probe_can_be_disabled_by_environment(monkeypatch, value):
    """Operators who verify reachability otherwise may skip the diagnostic."""
    monkeypatch.setenv("EXGENTIC_SKIP_MODEL_PROBE", value)
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health._fetch_models") as fetch:
        check_models_endpoint("gpt-4o", logger)  # no raise
    fetch.assert_not_called()


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_probe_stays_enabled_for_non_truthy_values(monkeypatch, value):
    monkeypatch.setenv("EXGENTIC_SKIP_MODEL_PROBE", value)
    logger = logging.getLogger("test")
    base, fetch = _patch_probe(200, {"data": [{"id": "gpt-4o"}]})
    with base, fetch as fetch_mock:
        check_models_endpoint("gpt-4o", logger)
    assert fetch_mock.call_count == 1


def test_missing_credentials_still_fail_before_any_probe(monkeypatch):
    """Skipping the probe must not skip layer 1, which is free and offline."""
    monkeypatch.setenv("EXGENTIC_SKIP_MODEL_PROBE", "1")
    logger = logging.getLogger("test")
    with patch("exgentic.integrations.litellm.health.validate_model_environment", return_value=["OPENAI_API_KEY"]):
        with pytest.raises(HealthCheckError):
            check_model_accessible_sync("gpt-4o", logger)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError("timed out"), "timed out"),
        (urllib.error.URLError(TimeoutError("timed out")), "timed out"),
        (urllib.error.URLError(socket.gaierror(-2, "Name or service not known")), "DNS resolution failed"),
        (urllib.error.URLError(ssl.SSLError("bad handshake")), "TLS handshake failed"),
        (urllib.error.URLError(ConnectionRefusedError(111, "refused")), "connection refused"),
        (urllib.error.URLError(OSError("network unreachable")), "connection failed"),
    ],
)
def test_fetch_failure_causes_are_distinguished(exc, expected):
    """A timeout reported as "unreachable" asserts what the probe never tested."""
    assert expected in _describe_fetch_failure(exc).format(timeout="10")


def test_timeout_message_names_the_budget_that_expired():
    """The operator needs the number to know the clamp is what bit them."""
    described = _describe_fetch_failure(TimeoutError("timed out")).format(timeout="10")
    assert "10s" in described
