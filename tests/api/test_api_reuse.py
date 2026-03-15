# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import time

from exgentic import execute
from exgentic.core.types import RunConfig


def test_reuse_skips_completed_sessions(tmp_path):
    output_dir = tmp_path / "outputs"
    run_id = "run-reuse"
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        run_id=run_id,
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    execute(config)
    log_path = output_dir / run_id / "run" / "run.log"
    assert log_path.exists()
    session_id = config.to_session_config("task-1").get_session_id()
    results_path = output_dir / run_id / "sessions" / session_id / "results.json"
    bench_results_path = output_dir / run_id / "sessions" / session_id / "benchmark" / "results.json"
    assert results_path.exists()
    assert bench_results_path.exists()
    before_results_mtime = results_path.stat().st_mtime_ns
    before_bench_mtime = bench_results_path.stat().st_mtime_ns

    time.sleep(0.01)
    execute(config)
    after_results_mtime = results_path.stat().st_mtime_ns
    after_bench_mtime = bench_results_path.stat().st_mtime_ns

    assert after_results_mtime == before_results_mtime
    assert after_bench_mtime == before_bench_mtime
