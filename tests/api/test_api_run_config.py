# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

from exgentic import (
    aggregate,
    evaluate,
    execute,
    list_agents,
    list_benchmarks,
    list_subsets,
    list_tasks,
    preview,
    results,
    status,
)
from exgentic.core.types import RunConfig, SessionOutcomeStatus


def _run_config(tmp_path, *, run_id: str, policy: str = "good_then_finish") -> RunConfig:
    return RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id=run_id,
        num_tasks=1,
        max_steps=5,
        max_actions=5,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": policy, "finish_after": 2},
    )


def test_evaluate_run_config_writes_files(tmp_path):
    config = _run_config(tmp_path, run_id="run-eval")
    run_results = evaluate(config)
    assert run_results.total_sessions == 1
    assert run_results.successful_sessions == 1
    assert run_results.benchmark_score == 1.0
    session = run_results.session_results[0]
    assert session.score == 1.0
    assert session.status == SessionOutcomeStatus.SUCCESS

    session_id = config.to_session_config("task-1").get_session_id()
    run_root = tmp_path / "outputs" / "run-eval"
    assert (run_root / "run" / "config.json").exists()
    assert (run_root / "results.json").exists()
    assert (run_root / "benchmark_results.json").exists()
    assert (run_root / "sessions" / session_id / "config.json").exists()
    assert (run_root / "sessions" / session_id / "results.json").exists()
    assert (run_root / "sessions" / session_id / "benchmark" / "results.json").exists()

    payload = json.loads((run_root / "results.json").read_text(encoding="utf-8"))
    assert payload["total_sessions"] == 1


def test_execute_then_aggregate(tmp_path):
    config = _run_config(tmp_path, run_id="run-exec")
    exec_results = execute(config)
    assert exec_results.benchmark_score is None
    agg_results = aggregate(config)
    assert agg_results.benchmark_score == 1.0


def test_preview_status_results(tmp_path):
    config = _run_config(tmp_path, run_id="run-preview")
    plan = preview(config)
    assert len(plan.to_run) == 1
    run_status = status(config)
    assert run_status.total_tasks == 1
    evaluate(config)
    loaded = results(config)
    assert loaded.total_sessions == 1


def test_listing_apis():
    benchmarks = list_benchmarks()
    agents = list_agents()
    assert any(item["slug_name"] == "test_benchmark" for item in benchmarks)
    assert any(item["slug_name"] == "test_agent" for item in agents)
    assert list_subsets("test_benchmark") == []
    tasks = list_tasks(benchmark="test_benchmark")
    assert "task-1" in tasks
