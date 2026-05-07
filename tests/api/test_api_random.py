# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from framework import execute
from framework.core.types import RunConfig


def _run(tmp_path, run_id: str):
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id=run_id,
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "random", "seed": 123},
        max_steps=5,
        max_actions=5,
    )
    return execute(config)


def test_random_policy_deterministic(tmp_path):
    first = _run(tmp_path, "run-random-1")
    second = _run(tmp_path, "run-random-2")
    assert first.session_results[0].score == second.session_results[0].score
    assert first.session_results[0].status == second.session_results[0].status
