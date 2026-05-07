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


def test_action_level_stop(tmp_path):
    config = _base_config(tmp_path, run_id="run-action-stop").model_copy(
        update={"agent_kwargs": {"policy": "return_none"}}
    )
    results = execute(config)
    assert results.session_results[0].status == SessionOutcomeStatus.UNFINISHED


def test_step_level_stop(tmp_path):
    config = _base_config(tmp_path, run_id="run-step-stop").model_copy(
        update={
            "benchmark_kwargs": {"tasks": ["task-1"], "stop_on_step": True},
            "agent_kwargs": {"policy": "good_only"},
        }
    )
    results = execute(config)
    assert results.session_results[0].status == SessionOutcomeStatus.UNFINISHED


def test_limit_reached(tmp_path):
    config = _base_config(tmp_path, run_id="run-limit").model_copy(
        update={
            "agent_kwargs": {"policy": "good_only"},
            "max_steps": 1,
            "max_actions": 1,
        }
    )
    results = execute(config)
    assert results.session_results[0].status == SessionOutcomeStatus.LIMIT_REACHED
