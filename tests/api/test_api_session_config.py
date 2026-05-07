# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from framework import execute, status
from framework.core.types import RunConfig, SessionOutcomeStatus


def test_execute_session_config(tmp_path):
    run_config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-session",
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    session_config = run_config.to_session_config("task-1")
    results = execute(session_config)
    assert results.total_sessions == 1
    assert results.session_results[0].status == SessionOutcomeStatus.SUCCESS


def test_status_session_config(tmp_path):
    run_config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-status-session",
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    session_config = run_config.to_session_config("task-1")
    run_status = status(session_config)
    assert run_status.total_tasks == 1
    assert run_status.task_ids == ["task-1"]
