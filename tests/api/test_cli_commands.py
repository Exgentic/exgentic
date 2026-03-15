# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from click.testing import CliRunner
from exgentic.core.types import RunConfig
from exgentic.interfaces.cli.main import cli


def test_cli_execute_concurrent(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "outputs"
    result = runner.invoke(
        cli,
        [
            "evaluate",
            "execute",
            "--benchmark",
            "test_benchmark",
            "--agent",
            "test_agent",
            "--task",
            "task-1",
            "--task",
            "task-2",
            "--set",
            'agent.policy="good_then_finish"',
            "--set",
            "agent.finish_after=2",
            "--max-workers",
            "2",
            "--run-id",
            "run-cli-concurrent",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    sessions_root = output_dir / "run-cli-concurrent" / "sessions"
    session_dirs = [p for p in sessions_root.iterdir() if (p / "config.json").exists()]
    assert len(session_dirs) == 2


def test_cli_execute_session_config(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "outputs"
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-cli-session",
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    session_config = config.to_session_config("task-1")
    config_path = tmp_path / "session_config.json"
    config_path.write_text(session_config.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "evaluate",
            "session",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    sessions_root = output_dir / "run-cli-session" / "sessions"
    session_dirs = [p for p in sessions_root.iterdir() if (p / "config.json").exists()]
    assert len(session_dirs) == 1


def test_cli_status_with_config(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "outputs"
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
        run_id="run-cli-status",
        num_tasks=1,
        benchmark_kwargs={"tasks": ["task-1"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )
    config_path = tmp_path / "run_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    result = runner.invoke(
        cli,
        [
            "status",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run-cli-status" in result.output
