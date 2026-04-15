# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import csv
import json

from click.testing import CliRunner
from exgentic.core.types import RunConfig
from exgentic.interfaces.cli.main import cli


def _write_config(path, *, run_id: str) -> str:
    cfg = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(path.parent / "outputs"),
        cache_dir=str(path.parent / "cache"),
        run_id=run_id,
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def test_batch_status_multiple_configs(tmp_path):
    runner = CliRunner()
    c1 = _write_config(tmp_path / "a.json", run_id="run-a")
    c2 = _write_config(tmp_path / "b.json", run_id="run-b")

    result = runner.invoke(
        cli,
        [
            "batch",
            "status",
            "--config",
            c1,
            "--config",
            c2,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Batch Status" in result.output
    assert "test_benchmark" in result.output
    assert "test_agent" in result.output


def test_batch_status_config_glob_pattern(tmp_path):
    runner = CliRunner()
    p1 = tmp_path / "outputs" / "r1" / "config.json"
    p2 = tmp_path / "outputs" / "r2" / "config.json"
    p1.parent.mkdir(parents=True, exist_ok=True)
    p2.parent.mkdir(parents=True, exist_ok=True)
    _write_config(p1, run_id="run-g1")
    _write_config(p2, run_id="run-g2")

    result = runner.invoke(
        cli,
        [
            "batch",
            "status",
            "--config",
            str(tmp_path / "outputs" / "**" / "config.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "test_benchmark" in result.output


def test_batch_status_shell_expanded_values_after_single_config(tmp_path):
    runner = CliRunner()
    c1 = _write_config(tmp_path / "s1.json", run_id="run-s1")
    c2 = _write_config(tmp_path / "s2.json", run_id="run-s2")

    # Simulates shell expansion where one --config token becomes multiple values.
    result = runner.invoke(
        cli,
        [
            "batch",
            "status",
            "--config",
            c1,
            c2,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "test_benchmark" in result.output


def test_batch_extract_writes_csv(tmp_path):
    runner = CliRunner()
    config_path = _write_config(tmp_path / "extract.json", run_id="run-extract")

    results_path = tmp_path / "run-extract" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_name": "Test Benchmark",
        "agent_name": "Test Agent",
        "benchmark_score": 1.0,
        "total_sessions": 1,
    }
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    output_csv = tmp_path / "results.csv"
    result = runner.invoke(
        cli,
        [
            "batch",
            "extract",
            "--config",
            config_path,
            "--output",
            str(output_csv),
        ],
    )

    assert result.exit_code == 0, result.output
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["benchmark_name"] == "Test Benchmark"
    assert rows[0]["benchmark_score"] == "1.0"


def _write_session_results(run_root, session_id: str, payload: dict) -> None:
    sdir = run_root / "sessions" / session_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "results.json").write_text(json.dumps(payload), encoding="utf-8")


def test_batch_status_cost_is_stale_during_live_run(tmp_path):
    """Regression test for Exgentic/exgentic#190.

    `batch status` reads Score/Cost from run-level `{run_id}/results.json`,
    which is only written at end of run. During a live run, fresh session-level
    `sessions/{sid}/results.json` files exist on disk but are ignored, so the
    Total Cost shown is a stale snapshot from the previous cleanly-exited run.
    """
    runner = CliRunner()
    config_path = _write_config(tmp_path / "live.json", run_id="run-live")

    run_root = tmp_path / "outputs" / "run-live"
    run_root.mkdir(parents=True, exist_ok=True)

    # Stale run-level aggregate left behind by a previous cleanly-exited run.
    stale_payload = {
        "benchmark_name": "Test Benchmark",
        "agent_name": "Test Agent",
        "benchmark_score": 0.10,
        "total_run_cost": 41.92,
        "total_agent_cost": 38.00,
    }
    (run_root / "results.json").write_text(json.dumps(stale_payload), encoding="utf-8")

    # Fresh session-level results from the currently-in-flight run.
    # Sum: 67.0 + 67.60 = 134.60
    fresh_sessions = [
        {"session_id": "s1", "score": 1.0, "agent_cost": 60.0, "benchmark_cost": 7.0},
        {"session_id": "s2", "score": 0.5, "agent_cost": 60.67, "benchmark_cost": 6.93},
    ]
    expected_total = sum(s["agent_cost"] + s["benchmark_cost"] for s in fresh_sessions)
    for s in fresh_sessions:
        _write_session_results(run_root, s["session_id"], s)

    result = runner.invoke(cli, ["batch", "status", "--config", config_path])
    assert result.exit_code == 0, result.output

    # Expected post-fix behavior: aggregate fresh session-level costs.
    assert f"${expected_total:.2f}" in result.output, (
        f"batch status should report the fresh aggregated Total Cost "
        f"${expected_total:.2f} from session-level results.json files. "
        f"Output:\n{result.output}"
    )
    # Bug: the stale run-level value leaks into the summary panel.
    assert "$41.92" not in result.output, (
        "batch status reported the stale run-level cost $41.92 instead of "
        "aggregating fresh session-level results (#190).\n"
        f"Output:\n{result.output}"
    )


def test_batch_status_cost_missing_when_run_level_absent(tmp_path):
    """Regression test for Exgentic/exgentic#190.

    On the first live run (before any clean exit has ever written the
    run-level aggregate), `batch status` shows no Total Cost at all even
    though fresh per-session costs are on disk.
    """
    runner = CliRunner()
    config_path = _write_config(tmp_path / "nofinish.json", run_id="run-nofinish")

    run_root = tmp_path / "outputs" / "run-nofinish"
    run_root.mkdir(parents=True, exist_ok=True)

    sessions = [
        {"session_id": "s1", "score": 1.0, "agent_cost": 12.5, "benchmark_cost": 0.5},
        {"session_id": "s2", "score": 0.0, "agent_cost": 20.0, "benchmark_cost": 2.0},
    ]
    expected_total = sum(s["agent_cost"] + s["benchmark_cost"] for s in sessions)
    for s in sessions:
        _write_session_results(run_root, s["session_id"], s)

    # Sanity: no run-level results.json yet.
    assert not (run_root / "results.json").exists()

    result = runner.invoke(cli, ["batch", "status", "--config", config_path])
    assert result.exit_code == 0, result.output

    assert f"${expected_total:.2f}" in result.output, (
        f"batch status should aggregate Total Cost ${expected_total:.2f} "
        f"from session-level results.json files when the run-level "
        f"aggregate is absent. Output:\n{result.output}"
    )
