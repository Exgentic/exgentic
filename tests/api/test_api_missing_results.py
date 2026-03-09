# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic import aggregate, execute
from exgentic.core.types import RunConfig


def test_aggregate_logs_missing_results_and_records_ids(tmp_path):
    output_dir = tmp_path / "outputs"
    run_id = "run-missing-results"
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        run_id=run_id,
        task_ids=["task-1", "task-2"],
        agent_kwargs={"policy": "good_then_finish", "finish_after": 1},
    )
    execute(config)

    missing_session_id = config.to_session_config("task-1").get_session_id()
    kept_session_id = config.to_session_config("task-2").get_session_id()
    missing_results = output_dir / run_id / "sessions" / missing_session_id / "results.json"
    assert missing_results.exists()
    missing_results.unlink()

    results = aggregate(config)

    log_path = output_dir / run_id / "run" / "run.log"
    log_text = log_path.read_text(encoding="utf-8")
    assert "Missing session results for 1/2 planned sessions." in log_text
    assert "Missing session ids:" in log_text

    assert results.planned_sessions == 2
    assert results.total_sessions == 1
    assert set(results.planned_session_ids or []) == {
        missing_session_id,
        kept_session_id,
    }
    assert results.executed_session_ids == [kept_session_id]
