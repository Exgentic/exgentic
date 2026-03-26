# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for LiteLLM model health check functionality."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from exgentic.integrations.litellm.health import check_model_accessible_sync
from litellm.exceptions import AuthenticationError, PermissionDeniedError


def test_check_model_accessible_sync_success(caplog):
    """Test successful model accessibility check."""
    caplog.set_level(logging.INFO)

    with patch("exgentic.integrations.litellm.health.acheck_model_accessible", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = None

        check_model_accessible_sync("gpt-4", logging.getLogger(__name__))

        assert "Running LiteLLM model health check (model=gpt-4)" in caplog.text
        assert "Model health check passed for gpt-4" in caplog.text


def test_check_model_accessible_sync_authentication_error(caplog):
    """Test handling of AuthenticationError with specific error message."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.integrations.litellm.health.acheck_model_accessible", new_callable=AsyncMock) as mock_check:
        mock_check.side_effect = AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4",
        )

        with pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("gpt-4", logging.getLogger(__name__))

        assert "API key authentication failed" in str(exc_info.value)
        assert "Please check your API key configuration" in str(exc_info.value)
        assert "Model health check failed for gpt-4" in caplog.text


def test_check_model_accessible_sync_permission_denied_error(caplog):
    """Test handling of PermissionDeniedError with specific error message."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.integrations.litellm.health.acheck_model_accessible", new_callable=AsyncMock) as mock_check:
        # Create a mock response object for PermissionDeniedError
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 403

        mock_check.side_effect = PermissionDeniedError(
            message="Access denied to model",
            llm_provider="openai",
            model="gpt-4",
            response=mock_response,
        )

        with pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("gpt-4", logging.getLogger(__name__))

        assert "Permission denied" in str(exc_info.value)
        assert "Your API key does not have access to this model" in str(exc_info.value)
        assert "Model health check failed for gpt-4" in caplog.text


def test_check_model_accessible_sync_generic_error(caplog):
    """Test handling of generic exceptions with fallback error message."""
    caplog.set_level(logging.ERROR)

    with patch("exgentic.integrations.litellm.health.acheck_model_accessible", new_callable=AsyncMock) as mock_check:
        mock_check.side_effect = ValueError("Some unexpected error")

        with pytest.raises(RuntimeError) as exc_info:
            check_model_accessible_sync("gpt-4", logging.getLogger(__name__))

        assert "Model gpt-4 is not accessible: Some unexpected error" in str(exc_info.value)
        assert "Model health check failed for gpt-4" in caplog.text
