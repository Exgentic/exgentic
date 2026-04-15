# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for LiteLLM health check error handling."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from exgentic.core.types.model_settings import ModelSettings, RetryStrategy
from exgentic.integrations.litellm.health import (
    _HEALTH_MAX_RETRY_DELAY,
    _HEALTH_MIN_RETRIES,
    _HEALTH_MIN_RETRY_DELAY,
    ErrorCategory,
    acheck_model_accessible,
    check_model_accessible_sync,
    classify_error,
)


class MockLiteLLMError(Exception):
    """Mock exception that mimics LiteLLM exceptions with .message attribute."""

    def __init__(self, message: str):
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        """Return empty string to simulate LiteLLM exceptions that don't implement __str__."""
        return ""


def test_health_check_extracts_message_attribute_from_exception(caplog):
    """Test that health check extracts error details from exception.message attribute."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.utils.sync.run_sync") as mock_run_sync:
        exc = MockLiteLLMError("API key authentication failed")
        mock_run_sync.side_effect = exc

        logger = logging.getLogger("test")

        with pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("test-model", logger)

        error_msg = str(exc_info.value)
        assert "API key authentication failed" in error_msg
        assert "test-model" in error_msg
        assert error_msg == "Model test-model is not accessible: API key authentication failed"


def test_health_check_falls_back_to_str_when_no_message_attribute(caplog):
    """Test that health check falls back to str(exc) when .message is not available."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.utils.sync.run_sync") as mock_run_sync:
        exc = ValueError("Standard error message")
        mock_run_sync.side_effect = exc

        logger = logging.getLogger("test")

        with pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("test-model", logger)

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
        mock_run_sync.side_effect = exc

        logger = logging.getLogger("test")

        with pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("test-model", logger)

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
