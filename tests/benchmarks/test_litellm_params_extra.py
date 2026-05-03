# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Verify benchmark evaluators thread litellm_params_extra to their LLM call sites."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

EXTRAS = {
    "api_base": "https://example.invalid/v1",
    "api_key": "secret",  # pragma: allowlist secret
    "extra_headers": {"X-Backend-Auth": "secret"},  # pragma: allowlist secret
}


def test_browsecomp_evaluator_forwards_extras_to_litellm_completion():
    pytest.importorskip("scripts_evaluation")
    pytest.importorskip("search_agent")
    from exgentic.benchmarks.browsecompplus.browsecomp_eval import BrowseCompEvaluatorOpenai

    evaluator = BrowseCompEvaluatorOpenai(litellm_params_extra=EXTRAS)

    fake_choice = MagicMock()
    fake_choice.get.return_value.content = '{"correct": "yes", "confidence": "100"}'
    fake_response = {"choices": [fake_choice]}
    fake_response_obj = MagicMock()
    fake_response_obj.__getitem__.side_effect = fake_response.__getitem__
    fake_response_obj.usage.copy.return_value = MagicMock()

    with patch(
        "exgentic.benchmarks.browsecompplus.browsecomp_eval.litellm.completion",
        return_value=fake_response_obj,
    ) as mock_completion:
        try:
            evaluator.evaluate_response(
                agent_response="some answer",
                instance={"query": "q", "gold_answer": "a", "evidence_docs": []},
            )
        except Exception:
            # parse_judge_response or downstream metric code may fail on the stub
            # response — we only care that litellm.completion was called correctly.
            pass

    assert mock_completion.called
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["api_base"] == EXTRAS["api_base"]
    assert call_kwargs["api_key"] == EXTRAS["api_key"]
    assert call_kwargs["extra_headers"] == EXTRAS["extra_headers"]


def test_tau2_benchmark_session_kwargs_includes_user_simulator_extras():
    """TAU2Benchmark.user_simulator_litellm_params_extra is included in session kwargs."""
    from exgentic.benchmarks.tau2.tau2_benchmark import TAU2Benchmark

    bench = TAU2Benchmark(user_simulator_litellm_params_extra=EXTRAS)
    kwargs = bench._get_session_kwargs()

    assert kwargs["user_simulator_litellm_params_extra"] == EXTRAS


def test_tau2_benchmark_default_extras_is_empty_dict():
    from exgentic.benchmarks.tau2.tau2_benchmark import TAU2Benchmark

    bench = TAU2Benchmark()
    assert bench.user_simulator_litellm_params_extra == {}
    assert bench._get_session_kwargs()["user_simulator_litellm_params_extra"] == {}
