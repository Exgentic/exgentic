# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

import json

from framework.integrations.litellm.trace_cost import load_trace_cost


def test_load_trace_cost_uses_explicit_cost_field(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"cost": 1.25, "prompt_tokens": 1, "completion_tokens": 2}),
                "not-json",
                json.dumps({"cost": "2.5"}),
            ]
        ),
        encoding="utf-8",
    )

    def _should_not_be_called(**_kwargs):
        raise AssertionError("token fallback should not be used when cost is explicit")

    import framework.integrations.litellm.trace_cost as trace_cost_mod

    monkeypatch.setattr(trace_cost_mod, "litellm_tokens_cost", _should_not_be_called)

    assert load_trace_cost(trace_path, "openai/gpt-4o-mini") == 3.75


def test_load_trace_cost_falls_back_to_token_estimate(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt_tokens": 10, "completion_tokens": 20}),
                json.dumps({"cost": None, "prompt_tokens": 5, "completion_tokens": 5}),
            ]
        ),
        encoding="utf-8",
    )

    class _Cost:
        def __init__(self, total_cost: float) -> None:
            self.total_cost = total_cost

    def _fake_token_cost(*, model_name: str, input_tokens: int, output_tokens: int):
        assert model_name == "openai/gpt-4o-mini"
        return _Cost(total_cost=(input_tokens + output_tokens) / 1000.0)

    import framework.integrations.litellm.trace_cost as trace_cost_mod

    monkeypatch.setattr(trace_cost_mod, "litellm_tokens_cost", _fake_token_cost)

    assert load_trace_cost(trace_path, "openai/gpt-4o-mini") == 0.04
