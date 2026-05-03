# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests that ``litellm_params_extra`` is forwarded to ``litellm.acompletion``."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest
from exgentic.integrations.litellm.health import (
    acheck_model_accessible,
    check_model_accessible_sync,
)


@pytest.mark.asyncio
async def test_acheck_forwards_litellm_params_extra():
    extras = {
        "api_base": "https://example.invalid/v1",
        "api_key": "secret",  # pragma: allowlist secret
        "extra_headers": {"X-Backend-Auth": "secret"},  # pragma: allowlist secret
    }
    with patch("litellm.acompletion", new=AsyncMock()) as mock_completion:
        await acheck_model_accessible("openai/gpt-4o-mini", litellm_params_extra=extras)

    assert mock_completion.await_count == 1
    call_kwargs = mock_completion.await_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["api_base"] == extras["api_base"]
    assert call_kwargs["api_key"] == extras["api_key"]
    assert call_kwargs["extra_headers"] == extras["extra_headers"]


@pytest.mark.asyncio
async def test_acheck_without_extras_passes_no_extra_kwargs():
    with patch("litellm.acompletion", new=AsyncMock()) as mock_completion:
        await acheck_model_accessible("openai/gpt-4o-mini")

    assert mock_completion.await_count == 1
    call_kwargs = mock_completion.await_args.kwargs
    assert "api_base" not in call_kwargs
    assert "api_key" not in call_kwargs
    assert "extra_headers" not in call_kwargs


def test_sync_wrapper_forwards_litellm_params_extra():
    extras = {"api_base": "https://example/v1"}
    captured: dict = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
        check_model_accessible_sync(
            "openai/gpt-4o-mini",
            logger=logging.getLogger("test"),
            litellm_params_extra=extras,
        )

    assert captured["api_base"] == extras["api_base"]
