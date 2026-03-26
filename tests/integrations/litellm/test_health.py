# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for LiteLLM health check error handling."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from exgentic.integrations.litellm.health import check_model_accessible_sync


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
