# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("EXGENTIC_CACHE_DIR", "/tmp/exgentic-rits-test-cache")
os.environ.setdefault("EXGENTIC_LITELLM_CACHE_DIR", "/tmp/exgentic-rits-test-cache/litellm")

import smolagents.models as _smolagents_models  # noqa: E402
import smolagents.utils as _smolagents_utils  # noqa: E402

if not hasattr(_smolagents_models, "is_rate_limit_error"):
    _smolagents_models.is_rate_limit_error = lambda exc: False  # type: ignore[attr-defined]
if not hasattr(_smolagents_utils, "Retrying"):
    class _Retrying:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    _smolagents_utils.Retrying = _Retrying  # type: ignore[attr-defined]

from exgentic.agents.litellm_tool_calling import instance as tool_instance_mod
from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance
from exgentic.agents.smolagents import base_instance as smol_base_mod
from exgentic.agents.smolagents.base_instance import SmolagentBaseAgentInstance
from exgentic.utils.cost import LiteLLMCostReport


RITS_OVERRIDES = {
    "model": "hosted_vllm/granite",
    "api_base": "https://rits.example/granite/v1",
    "api_key": "secret",
    "headers": {"RITS_API_KEY": "secret"},
}


class ConcreteSmolagentInstance(SmolagentBaseAgentInstance):
    def run_smolagent(self, tools):
        return None


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
        api_key="secret",
        headers={"RITS_API_KEY": "secret"},
    )


def test_litellm_tool_calling_completion_applies_rits_overrides(monkeypatch):
    monkeypatch.setattr(tool_instance_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(tool_instance_mod, "check_model_accessible_sync", MagicMock())
    agent = LiteLLMToolCallingAgentInstance(session_id="s1", model="rits/granite")
    agent._completion_with_retries = MagicMock(side_effect=lambda kwargs: kwargs)

    kwargs = agent._completion(model="rits/granite", messages=[], caching=False)

    assert kwargs["model"] == "hosted_vllm/granite"
    assert kwargs["api_base"] == "https://rits.example/granite/v1"
    assert kwargs["api_key"] == "secret"
    assert kwargs["headers"] == {"RITS_API_KEY": "secret"}


def test_litellm_tool_calling_unknown_pricing_does_not_fail(monkeypatch):
    monkeypatch.setattr(tool_instance_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(tool_instance_mod, "check_model_accessible_sync", MagicMock())
    agent = LiteLLMToolCallingAgentInstance(session_id="s1", model="rits/granite")
    agent._cost_data = MagicMock()
    agent._cost_data.update_cost_from_tokens.side_effect = ValueError("No pricing info found")

    agent._register_cost(SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    agent._cost_data.update_cost_from_tokens.assert_called_once_with(10, 5)


def test_smolagent_internal_model_receives_rits_kwargs(monkeypatch):
    resolver = MagicMock(return_value=RITS_OVERRIDES)
    health = MagicMock()
    model_cls = MagicMock()
    monkeypatch.setattr(smol_base_mod, "build_rits_overrides", resolver)
    monkeypatch.setattr(smol_base_mod, "check_model_accessible_sync", health)
    monkeypatch.setattr(smol_base_mod, "LiteLLMModel", model_cls)

    agent = ConcreteSmolagentInstance(session_id="s1", model_id="rits/granite")
    agent.get_internal_model()

    resolver.assert_called_once_with("rits/granite")
    health.assert_called_once()
    assert health.call_args.args[0] == "hosted_vllm/granite"
    assert health.call_args.kwargs["api_base"] == "https://rits.example/granite/v1"
    model_cls.assert_called_once()
    assert model_cls.call_args.kwargs["model_id"] == "hosted_vllm/granite"
    assert model_cls.call_args.kwargs["api_base"] == "https://rits.example/granite/v1"
    assert model_cls.call_args.kwargs["api_key"] == "secret"
    assert model_cls.call_args.kwargs["extra_headers"] == {"RITS_API_KEY": "secret"}


def test_smolagent_unknown_pricing_returns_empty_cost_report(monkeypatch):
    monkeypatch.setattr(smol_base_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(smol_base_mod, "check_model_accessible_sync", MagicMock())
    agent = ConcreteSmolagentInstance(session_id="s1", model_id="rits/granite")
    agent._agent = SimpleNamespace(
        monitor=SimpleNamespace(
            get_total_token_counts=MagicMock(
                return_value=SimpleNamespace(input_tokens=10, output_tokens=5)
            )
        )
    )
    monkeypatch.setattr(LiteLLMCostReport, "from_token_counts", MagicMock(side_effect=ValueError("unknown model")))

    report = agent.get_cost()

    assert isinstance(report, LiteLLMCostReport)
    assert report.model_name == "rits/granite"
    assert report.total_cost == 0
