# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from framework import execute
from framework.core.types import RunConfig, SessionOutcomeStatus


def _base_config(tmp_path, *, run_id: str) -> RunConfig:
    return RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id=run_id,
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
    )


def test_invalid_action_from_agent(tmp_path):
    config = _base_config(tmp_path, run_id="run-invalid-action").model_copy(
        update={
            "agent_kwargs": {"policy": "invalid_action"},
        }
    )
    results = execute(config)
    assert results.session_results[0].status == SessionOutcomeStatus.ERROR


def test_invalid_observation_from_benchmark(tmp_path):
    config = _base_config(tmp_path, run_id="run-invalid-observation").model_copy(
        update={
            "agent_kwargs": {"policy": "good_only"},
            "benchmark_kwargs": {"tasks": ["task-1"], "invalid_observation": True},
        }
    )
    results = execute(config)
    assert results.session_results[0].status == SessionOutcomeStatus.ERROR


def test_agent_exception_marks_error(tmp_path):
    config = _base_config(tmp_path, run_id="run-agent-error").model_copy(
        update={
            "agent_kwargs": {"policy": "raise_error"},
        }
    )
    results = execute(config)
    assert results.session_results[0].status == SessionOutcomeStatus.ERROR
