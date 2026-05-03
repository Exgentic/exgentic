# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Verify ``litellm_params_extra`` flow from agents into LiteLLM call sites.

Covers both ``LiteLLMToolCallingAgentInstance`` and (when smolagents is
installed) ``SmolagentBaseAgentInstance``: the dict must reach both the
health check and the actual completion / model-construction call site.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

EXTRAS = {
    "api_base": "https://rits.example/granite/v1",
    "api_key": "secret",  # pragma: allowlist secret
    "extra_headers": {"RITS_API_KEY": "secret"},  # pragma: allowlist secret
}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "exgentic-cache"
    litellm_cache_dir = cache_dir / "litellm"
    litellm_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EXGENTIC_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("EXGENTIC_LITELLM_CACHE_DIR", str(litellm_cache_dir))


def test_litellm_tool_calling_forwards_extras_to_health_check():
    from exgentic.agents.litellm_tool_calling import instance as mod

    with patch.object(mod, "check_model_accessible_sync") as health:
        mod.LiteLLMToolCallingAgentInstance(
            session_id="s1",
            model="hosted_vllm/granite",
            litellm_params_extra=EXTRAS,
        )

    health.assert_called_once()
    assert health.call_args.kwargs["litellm_params_extra"] == EXTRAS


def test_litellm_tool_calling_completion_includes_extras():
    from exgentic.agents.litellm_tool_calling import instance as mod

    with patch.object(mod, "check_model_accessible_sync"):
        agent = mod.LiteLLMToolCallingAgentInstance(
            session_id="s1",
            model="hosted_vllm/granite",
            litellm_params_extra=EXTRAS,
        )

    captured = {}

    def _capture(call_kwargs):
        captured.update(call_kwargs)
        return MagicMock()

    agent._completion_with_retries = _capture
    agent._completion(model="hosted_vllm/granite", messages=[], caching=False)

    assert captured["api_base"] == EXTRAS["api_base"]
    assert captured["api_key"] == EXTRAS["api_key"]
    assert captured["extra_headers"] == EXTRAS["extra_headers"]


def test_litellm_tool_calling_default_no_extras():
    from exgentic.agents.litellm_tool_calling import instance as mod

    with patch.object(mod, "check_model_accessible_sync") as health:
        mod.LiteLLMToolCallingAgentInstance(session_id="s1", model="openai/gpt-4o-mini")

    assert health.call_args.kwargs["litellm_params_extra"] is None


def test_smolagents_forwards_extras_to_health_check_and_litellm_model():
    pytest.importorskip("smolagents")
    from exgentic.agents.smolagents import base_instance as mod

    with patch.object(mod, "check_model_accessible_sync") as health:
        agent = mod.SmolagentBaseAgentInstance(
            session_id="s1",
            model_id="hosted_vllm/granite",
            litellm_params_extra=EXTRAS,
        )

    assert health.call_args.kwargs["litellm_params_extra"] == EXTRAS

    with patch.object(mod, "LiteLLMModel") as model_cls:
        agent.get_internal_model()

    kwargs = model_cls.call_args.kwargs
    assert kwargs["api_base"] == EXTRAS["api_base"]
    assert kwargs["api_key"] == EXTRAS["api_key"]
    assert kwargs["extra_headers"] == EXTRAS["extra_headers"]


def test_smolagents_default_no_extras():
    pytest.importorskip("smolagents")
    from exgentic.agents.smolagents import base_instance as mod

    with patch.object(mod, "check_model_accessible_sync") as health:
        agent = mod.SmolagentBaseAgentInstance(session_id="s1", model_id="openai/gpt-4o-mini")

    assert health.call_args.kwargs["litellm_params_extra"] is None

    with patch.object(mod, "LiteLLMModel") as model_cls:
        agent.get_internal_model()

    kwargs = model_cls.call_args.kwargs
    assert "api_base" not in kwargs
    assert "extra_headers" not in kwargs
