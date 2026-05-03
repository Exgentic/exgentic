# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Ordering invariant for the parallel scan in RunStatus.from_session_configs."""

from __future__ import annotations

from pathlib import Path

from exgentic.core.types import RunConfig
from exgentic.core.types.run import RunStatus
from exgentic.core.types.session import SessionExecutionStatus


def test_from_session_configs_preserves_input_order(tmp_path: Path):
    cfg = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id="ordering-run",
        num_tasks=64,  # > pool size of 32 forces real parallelism
        benchmark_kwargs={"tasks": [f"task-{i:03d}" for i in range(64)]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 1},
    )

    with cfg.get_context():
        session_configs = cfg.get_session_configs()
        run_status = RunStatus.from_session_configs(cfg, session_configs)

    assert all(s.status == SessionExecutionStatus.MISSING for s in run_status.session_statuses)
    for sc, status in zip(session_configs, run_status.session_statuses, strict=True):
        assert status.task_id == str(sc.task_id)
    assert run_status.task_ids == [str(sc.task_id) for sc in session_configs]


def test_from_session_configs_empty(tmp_path: Path):
    cfg = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id="empty-run",
        num_tasks=0,
        benchmark_kwargs={"tasks": []},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 1},
    )

    with cfg.get_context():
        run_status = RunStatus.from_session_configs(cfg, [])

    assert run_status.session_statuses == []
    assert run_status.task_ids == []
    assert run_status.total_tasks == 0
