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
    assert "run-a" in result.output
    assert "run-b" in result.output


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
    assert "run-g1" in result.output
    assert "run-g2" in result.output


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
    assert "run-s1" in result.output
    assert "run-s2" in result.output


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
