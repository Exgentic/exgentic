# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Regression tests for the parallel scan in RunStatus.from_session_configs.

This is the central choke-point for `batch status`, `batch evaluate`'s
plan phase, and `batch aggregate`. A reordering bug would silently
mismatch task IDs against statuses; integration tests would still pass
because the *count* is right, but downstream RunPlan would zip wrong
tasks to wrong statuses.
"""

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
        # 64 tasks > the parallel pool size (32) — guarantees real parallel
        # execution rather than a single-batch shortcut.
        num_tasks=64,
        benchmark_kwargs={"tasks": [f"task-{i:03d}" for i in range(64)]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 1},
    )

    with cfg.get_context():
        session_configs = cfg.get_session_configs()
        run_status = RunStatus.from_session_configs(cfg, session_configs)

    # No session has been executed → all MISSING.
    assert all(s.status == SessionExecutionStatus.MISSING for s in run_status.session_statuses)

    # Critical invariant: status[i].task_id == session_configs[i].task_id.
    for sc, status in zip(session_configs, run_status.session_statuses):
        assert status.task_id == str(sc.task_id), (
            f"Order desync: input {sc.task_id!r} -> status.task_id {status.task_id!r}"
        )

    # And task_ids field on RunStatus matches the input order too.
    assert run_status.task_ids == [str(sc.task_id) for sc in session_configs]


def test_from_session_configs_empty(tmp_path: Path):
    """Empty input must not raise (max_workers guard)."""
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
