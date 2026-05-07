# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

import json

from framework import execute
from framework.core.types import RunConfig


def test_session_files_written(tmp_path):
    output_dir = tmp_path / "outputs"
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-files",
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    results = execute(config)
    session_id = results.session_results[0].session_id
    session_root = output_dir / "run-files" / "sessions" / session_id
    assert (session_root / "config.json").exists()
    assert (session_root / "results.json").exists()
    assert (session_root / "session.json").exists()
    assert (session_root / "benchmark" / "config.json").exists()
    trajectory = session_root / "trajectory.jsonl"
    assert trajectory.exists()
    first_event = json.loads(trajectory.read_text(encoding="utf-8").splitlines()[0])
    assert first_event["run_id"] == "run-files"
