# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for LiteLLM health check error handling."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from exgentic.core.types.model_settings import ModelSettings, RetryStrategy
from exgentic.integrations.litellm.health import (
    _is_rate_limit_error,
    acheck_model_accessible,
    check_model_accessible_sync,
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


def test_is_rate_limit_error_direct_status_code():
    assert _is_rate_limit_error(_FakeRateLimitError()) is True


def test_is_rate_limit_error_wrapped():
    assert _is_rate_limit_error(_FakeWrappedRateLimitError()) is True


def test_is_rate_limit_error_not_rate_limit():
    assert _is_rate_limit_error(ValueError("nope")) is False


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
    settings = ModelSettings(num_retries=2, retry_after=0.01)

    with patch("litellm.acompletion", mock):
        with pytest.raises(type(exc)):
            await acheck_model_accessible("m", model_settings=settings)

    # 1 initial + 2 retries = 3 calls
    assert mock.call_count == 3


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
    settings = ModelSettings(num_retries=3, retry_after=0.01, retry_strategy=RetryStrategy.CONSTANT)

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            await acheck_model_accessible("m", model_settings=settings)

    # Constant strategy: delay should always be retry_after (0.01)
    assert sleep_mock.call_count == 2
    for call in sleep_mock.call_args_list:
        assert call.args[0] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_acheck_defaults_to_model_settings_when_none():
    """When model_settings is None, ModelSettings() defaults are used."""
    mock = AsyncMock(side_effect=[_FakeRateLimitError(), None])

    with patch("litellm.acompletion", mock):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await acheck_model_accessible("m")

    # Default ModelSettings has num_retries=5, so retry should happen
    assert mock.call_count == 2
