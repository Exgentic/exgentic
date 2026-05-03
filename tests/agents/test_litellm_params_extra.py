# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Verify ``litellm_params_extra`` flow from agent constructors to LiteLLM call sites."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

EXTRAS = {
    "api_base": "https://example.invalid/v1",
    "api_key": "secret",  # pragma: allowlist secret
    "extra_headers": {"X-Backend-Auth": "secret"},  # pragma: allowlist secret
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
            model="openai/gpt-4o-mini",
            litellm_params_extra=EXTRAS,
        )

    health.assert_called_once()
    assert health.call_args.kwargs["litellm_params_extra"] == EXTRAS


def test_litellm_tool_calling_completion_includes_extras():
    from exgentic.agents.litellm_tool_calling import instance as mod

    with patch.object(mod, "check_model_accessible_sync"):
        agent = mod.LiteLLMToolCallingAgentInstance(
            session_id="s1",
            model="openai/gpt-4o-mini",
            litellm_params_extra=EXTRAS,
        )

    captured = {}

    def _capture(call_kwargs):
        captured.update(call_kwargs)
        return MagicMock()

    agent._completion_with_retries = _capture
    agent._completion(model="openai/gpt-4o-mini", messages=[], caching=False)

    assert captured["api_base"] == EXTRAS["api_base"]
    assert captured["api_key"] == EXTRAS["api_key"]
    assert captured["extra_headers"] == EXTRAS["extra_headers"]


def test_litellm_tool_calling_default_no_extras():
    from exgentic.agents.litellm_tool_calling import instance as mod

    with patch.object(mod, "check_model_accessible_sync") as health:
        mod.LiteLLMToolCallingAgentInstance(session_id="s1", model="openai/gpt-4o-mini")

    assert health.call_args.kwargs["litellm_params_extra"] == {}


def test_smolagents_forwards_extras_to_health_check_and_litellm_model():
    pytest.importorskip("smolagents")
    from exgentic.agents.smolagents import base_instance as mod

    with patch.object(mod, "check_model_accessible_sync") as health:
        agent = mod.SmolagentBaseAgentInstance(
            session_id="s1",
            model_id="openai/gpt-4o-mini",
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

    assert health.call_args.kwargs["litellm_params_extra"] == {}

    with patch.object(mod, "LiteLLMModel") as model_cls:
        agent.get_internal_model()

    kwargs = model_cls.call_args.kwargs
    assert "api_base" not in kwargs
    assert "extra_headers" not in kwargs


def test_openai_mcp_agent_forwards_extras_to_health_check():
    pytest.importorskip("agents")
    import asyncio

    from exgentic.agents.openai import instance as mod

    async_health = MagicMock(return_value=None)

    async def _fake_acheck(*args, **kwargs):
        async_health(*args, **kwargs)

    agent = mod.OpenAIMCPAgentInstance(
        session_id="s1",
        model_id="openai/gpt-4o-mini",
        litellm_params_extra=EXTRAS,
    )

    with patch.object(mod, "acheck_model_accessible", _fake_acheck):
        asyncio.run(agent._check_model_access_once())

    assert async_health.call_args.kwargs["litellm_params_extra"] == EXTRAS


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("exgentic.agents.cli.codex.agent", "CodexAgentInstance"),
        ("exgentic.agents.cli.gemini.agent", "GeminiAgentInstance"),
        ("exgentic.agents.cli.claude.agent", "ClaudeCodeAgentInstance"),
    ],
)
def test_cli_agent_subclasses_forward_extras_to_base(module_path, class_name):
    """Each CLI agent subclass must forward litellm_params_extra to its super().__init__()."""
    import importlib

    from exgentic.agents.cli import base as base_mod

    mod = importlib.import_module(module_path)
    agent_cls = getattr(mod, class_name)

    with patch.object(base_mod, "check_model_accessible_sync") as health:
        agent = agent_cls(
            session_id="s1",
            model_id="openai/gpt-4o-mini",
            litellm_params_extra=EXTRAS,
        )

    assert agent._litellm_params_extra == EXTRAS
    assert health.call_args.kwargs["litellm_params_extra"] == EXTRAS
