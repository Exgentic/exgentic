# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import pytest
from exgentic import execute
from exgentic.core.types import RunConfig

from .fixtures.test_agent import TestAgent
from .fixtures.test_benchmark import TestBenchmark


def test_execute_with_instances(tmp_path):
    benchmark = TestBenchmark(tasks=["task-1"])
    agent = TestAgent(policy="good_then_finish", finish_after=2)
    results = execute(
        benchmark=benchmark,
        agent=agent,
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-instance",
    )
    assert results.total_sessions == 1


def test_instance_kwargs_rejected(tmp_path):
    benchmark = TestBenchmark(tasks=["task-1"])
    agent = TestAgent(policy="good_then_finish", finish_after=2)
    with pytest.raises(ValueError):
        execute(
            benchmark=benchmark,
            agent=agent,
            output_dir=str(tmp_path / "outputs"),
            cache_dir=str(tmp_path / "cache"),
            run_id="run-instance-bad",
            benchmark_kwargs={"tasks": ["task-1"]},
        )
    with pytest.raises(ValueError):
        execute(
            benchmark=benchmark,
            agent=agent,
            output_dir=str(tmp_path / "outputs"),
            cache_dir=str(tmp_path / "cache"),
            run_id="run-instance-bad-2",
            agent_kwargs={"policy": "good_only"},
        )


def test_config_and_args_rejected(tmp_path):
    run_config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-config-ok",
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    with pytest.raises(ValueError):
        execute(
            run_config,
            benchmark="test_benchmark",
            agent="test_agent",
        )
