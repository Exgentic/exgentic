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


def test_with_overrides(tmp_path):
    config = RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path),
        num_tasks=100,
        max_steps=100,
        max_actions=100,
    )

    # Overrides are applied
    updated = config.with_overrides(num_tasks=5, max_workers=1)
    assert updated.num_tasks == 5
    assert updated.max_workers == 1
    assert updated.max_steps == 100  # not overridden

    # Original is unchanged
    assert config.num_tasks == 100
    assert config.max_workers is None

    # None values are skipped, returns same object
    same = config.with_overrides(num_tasks=None, max_workers=None)
    assert same is config

    # Unknown fields are silently ignored
    also_same = config.with_overrides(num_tasks=None, bogus=42)
    assert also_same is config


def test_overridable_fields_match_run_config():
    """Ensure OVERRIDABLE_FIELDS only lists real RunConfig fields."""
    for field in RunConfig.OVERRIDABLE_FIELDS:
        assert field in RunConfig.model_fields, f"{field} is not a RunConfig field"


def test_api_functions_accept_overridable_fields():
    """Ensure Python API functions accept all OVERRIDABLE_FIELDS as parameters."""
    import inspect

    from exgentic.interfaces.lib import api

    for fn_name in ("evaluate", "execute", "aggregate", "status"):
        fn = getattr(api, fn_name)
        params = set(inspect.signature(fn).parameters.keys())
        for field in RunConfig.OVERRIDABLE_FIELDS:
            assert field in params, (
                f"api.{fn_name}() is missing '{field}' parameter. "
                f"All RunConfig.OVERRIDABLE_FIELDS must be accepted by API functions."
            )


def test_cli_commands_accept_overridable_fields():
    """Ensure CLI commands that load configs accept all OVERRIDABLE_FIELDS as options."""
    from exgentic.interfaces.cli.commands.batch import batch_evaluate_cmd, batch_execute_cmd
    from exgentic.interfaces.cli.commands.evaluate import evaluate_cmd

    for cmd in (evaluate_cmd, batch_evaluate_cmd, batch_execute_cmd):
        param_names = {p.name for p in cmd.params}
        for field in RunConfig.OVERRIDABLE_FIELDS:
            assert field in param_names, (
                f"{cmd.name} is missing --{field.replace('_', '-')} option. "
                f"All RunConfig.OVERRIDABLE_FIELDS must be accepted as CLI options."
            )


def test_has_run_options_allows_overridable_fields():
    """Ensure has_run_options does not reject overridable fields."""
    from exgentic.interfaces.cli.options import has_run_options

    # All overridable fields set, nothing else — should return False
    kwargs = {
        "benchmark": None,
        "agent": None,
        "agent_json": None,
        "agent_arg": (),
        "set_values": (),
        "subset": None,
        "tasks": (),
        "num_tasks": 5,
        "model": None,
        "debug": False,
        "overwrite": False,
        "log_level": None,
        "output_dir": None,
        "cache_dir": None,
        "run_id": None,
        "max_workers": 2,
        "max_steps": 50,
        "max_actions": 50,
    }
    assert not has_run_options(**kwargs), "has_run_options should allow overridable fields"
