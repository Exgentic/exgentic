# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("EXGENTIC_CACHE_DIR", "/tmp/exgentic-rits-test-cache")
os.environ.setdefault("EXGENTIC_LITELLM_CACHE_DIR", "/tmp/exgentic-rits-test-cache/litellm")

from exgentic.agents.litellm_tool_calling import instance as tool_instance_mod  # noqa: E402
from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance  # noqa: E402

RITS_OVERRIDES = {
    "model": "hosted_vllm/granite",
    "api_base": "https://rits.example/granite/v1",
    "api_key": "secret",  # pragma: allowlist secret
    "headers": {"RITS_API_KEY": "secret"},  # pragma: allowlist secret
}


def test_litellm_tool_calling_resolves_rits_once_and_uses_resolved_health(monkeypatch):
    resolver = MagicMock(return_value=RITS_OVERRIDES)
    health = MagicMock()
    monkeypatch.setattr(tool_instance_mod, "build_rits_overrides", resolver)
    monkeypatch.setattr(tool_instance_mod, "check_model_accessible_sync", health)

    LiteLLMToolCallingAgentInstance(session_id="s1", model="rits/granite")

    resolver.assert_called_once_with("rits/granite")
    health.assert_called_once_with(
        "hosted_vllm/granite",
        logger=health.call_args.kwargs["logger"],
        model_settings=health.call_args.kwargs["model_settings"],
        api_base="https://rits.example/granite/v1",
        api_key="secret",  # pragma: allowlist secret
        headers={"RITS_API_KEY": "secret"},  # pragma: allowlist secret
    )


def test_litellm_tool_calling_completion_applies_rits_overrides(monkeypatch):
    monkeypatch.setattr(tool_instance_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(tool_instance_mod, "check_model_accessible_sync", MagicMock())
    agent = LiteLLMToolCallingAgentInstance(session_id="s1", model="rits/granite")
    agent._completion_with_retries = MagicMock(side_effect=lambda kwargs: kwargs)

    kwargs = agent._completion(model="rits/granite", messages=[], caching=False)

    assert kwargs["model"] == "hosted_vllm/granite"
    assert kwargs["api_base"] == "https://rits.example/granite/v1"
    assert kwargs["api_key"] == "secret"  # pragma: allowlist secret
    assert kwargs["headers"] == {"RITS_API_KEY": "secret"}  # pragma: allowlist secret


def test_litellm_tool_calling_unknown_pricing_does_not_fail(monkeypatch):
    monkeypatch.setattr(tool_instance_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(tool_instance_mod, "check_model_accessible_sync", MagicMock())
    agent = LiteLLMToolCallingAgentInstance(session_id="s1", model="rits/granite")
    agent._cost_data = MagicMock()
    agent._cost_data.update_cost_from_tokens.side_effect = ValueError("No pricing info found")

    agent._register_cost(SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    agent._cost_data.update_cost_from_tokens.assert_called_once_with(10, 5)
