# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

smolagents = pytest.importorskip("smolagents")
import smolagents.models as _smolagents_models  # noqa: E402
import smolagents.utils as _smolagents_utils  # noqa: E402

if not hasattr(_smolagents_models, "is_rate_limit_error") or not hasattr(_smolagents_utils, "Retrying"):
    pytest.skip(
        "smolagents compatibility APIs unavailable for RITS support tests",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _set_test_environment(monkeypatch, tmp_path):
    cache_dir = tmp_path / "exgentic-rits-test-cache"
    litellm_cache_dir = cache_dir / "litellm"
    litellm_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EXGENTIC_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("EXGENTIC_LITELLM_CACHE_DIR", str(litellm_cache_dir))


from exgentic.agents.smolagents import base_instance as smol_base_mod  # noqa: E402
from exgentic.agents.smolagents.base_instance import SmolagentBaseAgentInstance  # noqa: E402
from exgentic.utils.cost import LiteLLMCostReport  # noqa: E402

RITS_OVERRIDES = {
    "model": "hosted_vllm/granite",
    "api_base": "https://rits.example/granite/v1",
    "api_key": "secret",  # pragma: allowlist secret
    "headers": {"RITS_API_KEY": "secret"},  # pragma: allowlist secret
}


class ConcreteSmolagentInstance(SmolagentBaseAgentInstance):
    def run_smolagent(self, tools):
        return None


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
    assert model_cls.call_args.kwargs["api_key"] == "secret"  # pragma: allowlist secret
    assert model_cls.call_args.kwargs["extra_headers"] == {"RITS_API_KEY": "secret"}  # pragma: allowlist secret


def test_smolagent_unknown_pricing_returns_empty_cost_report(monkeypatch):
    monkeypatch.setattr(smol_base_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(smol_base_mod, "check_model_accessible_sync", MagicMock())
    agent = ConcreteSmolagentInstance(session_id="s1", model_id="rits/granite")
    agent._agent = SimpleNamespace(
        monitor=SimpleNamespace(
            get_total_token_counts=MagicMock(return_value=SimpleNamespace(input_tokens=10, output_tokens=5))
        )
    )
    monkeypatch.setattr(LiteLLMCostReport, "from_token_counts", MagicMock(side_effect=ValueError("unknown model")))

    report = agent.get_cost()

    assert isinstance(report, LiteLLMCostReport)
    assert report.model_name == "rits/granite"
    assert report.total_cost == 0


def test_smolagent_unexpected_cost_error_propagates(monkeypatch):
    monkeypatch.setattr(smol_base_mod, "build_rits_overrides", MagicMock(return_value=RITS_OVERRIDES))
    monkeypatch.setattr(smol_base_mod, "check_model_accessible_sync", MagicMock())
    agent = ConcreteSmolagentInstance(session_id="s1", model_id="rits/granite")
    agent._agent = SimpleNamespace(
        monitor=SimpleNamespace(
            get_total_token_counts=MagicMock(return_value=SimpleNamespace(input_tokens=10, output_tokens=5))
        )
    )
    monkeypatch.setattr(LiteLLMCostReport, "from_token_counts", MagicMock(side_effect=RuntimeError("bad usage")))

    with pytest.raises(RuntimeError, match="bad usage"):
        agent.get_cost()
