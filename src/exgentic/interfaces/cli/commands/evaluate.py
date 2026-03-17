# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

import rich_click as click

from ....core.types import RunConfig, SessionConfig
from ....utils.installation_tracker import is_installed
from ....utils.settings import get_settings
from ...lib.api import (
    aggregate,
    evaluate,
    execute,
    get_agent_setup_script_path,
    get_setup_script_path,
    setup_agent,
    setup_benchmark,
)
from ..options import add_run_options, has_run_options, run_with


def _is_docker_runner(set_values: tuple[str, ...]) -> bool:
    """Check if the runner is docker (via --set or global settings)."""
    for item in set_values:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if key in ("benchmark.runner", "settings.default_runner") and val.strip("\"'") == "docker":
            return True
    return get_settings().default_runner == "docker"


def _has_setup_script(name: str, install_type: str) -> bool:
    """Check if a benchmark/agent has a setup.sh script."""
    try:
        if install_type == "benchmark":
            get_setup_script_path(name)
        else:
            get_agent_setup_script_path(name)
        return True
    except (ValueError, FileNotFoundError):
        return False


def _ensure_installed(
    benchmark: str,
    agent: str,
    set_values: tuple[str, ...],
) -> None:
    """Prompt to auto-setup benchmark/agent if not installed (skip for docker)."""
    if _is_docker_runner(set_values):
        return

    to_setup: list[tuple[str, str]] = []
    if not is_installed(benchmark, "benchmark") and _has_setup_script(benchmark, "benchmark"):
        to_setup.append(("benchmark", benchmark))
    if not is_installed(agent, "agent") and _has_setup_script(agent, "agent"):
        to_setup.append(("agent", agent))

    if not to_setup:
        return

    names = ", ".join(f"{t} '{n}'" for t, n in to_setup)
    if not click.confirm(f"{names} not set up. Install now?", default=True):
        raise click.Abort()
    for install_type, name in to_setup:
        if install_type == "benchmark":
            setup_benchmark(name)
        else:
            setup_agent(name)


def _load_config_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_config_from_session(session_config: SessionConfig) -> RunConfig:
    return RunConfig(
        benchmark=session_config.benchmark,
        agent=session_config.agent,
        subset=session_config.subset,
        task_ids=[session_config.task_id],
        output_dir=session_config.output_dir,
        cache_dir=session_config.cache_dir,
        run_id=session_config.run_id,
        model=session_config.model,
        benchmark_kwargs=session_config.benchmark_kwargs,
        agent_kwargs=session_config.agent_kwargs,
    )


@click.group("evaluate", invoke_without_command=True)
@add_run_options(required=False)
@click.option("--config", "config_path", help="RunConfig JSON file")
@click.pass_context
def evaluate_cmd(
    ctx: click.Context,
    benchmark: str | None,
    agent: str | None,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    max_steps: int | None,
    max_actions: int | None,
    model: str | None,
    debug: bool,
    overwrite: bool,
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
    config_path: str | None,
) -> None:
    """Run sessions and aggregate results."""
    if config_path:
        if ctx.invoked_subcommand is not None:
            raise click.ClickException("--config cannot be used with subcommands.")
        if has_run_options(
            benchmark=benchmark,
            agent=agent,
            agent_json=agent_json,
            agent_arg=agent_arg,
            set_values=set_values,
            subset=subset,
            tasks=tasks,
            num_tasks=num_tasks,
            max_steps=max_steps,
            max_actions=max_actions,
            model=model,
            debug=debug,
            overwrite=overwrite,
            log_level=log_level,
            output_dir=output_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            max_workers=max_workers,
        ):
            raise click.ClickException("Do not pass run options together with --config.")
        config = RunConfig.model_validate(_load_config_file(config_path))
        evaluate(config)
        return
    if ctx.invoked_subcommand is not None:
        if has_run_options(
            benchmark=benchmark,
            agent=agent,
            agent_json=agent_json,
            agent_arg=agent_arg,
            set_values=set_values,
            subset=subset,
            tasks=tasks,
            num_tasks=num_tasks,
            max_steps=max_steps,
            max_actions=max_actions,
            model=model,
            debug=debug,
            overwrite=overwrite,
            log_level=log_level,
            output_dir=output_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            max_workers=max_workers,
        ):
            raise click.ClickException(
                "Pass options after the subcommand, e.g. " "'exgentic evaluate execute --benchmark ...'."
            )
        return
    if not benchmark or not agent:
        raise click.ClickException("--benchmark and --agent are required.")

    _ensure_installed(benchmark, agent, set_values)

    run_with(
        evaluate,
        benchmark=benchmark,
        agent=agent,
        agent_json=agent_json,
        agent_arg=agent_arg,
        set_values=set_values,
        subset=subset,
        tasks=tasks,
        num_tasks=num_tasks,
        model=model,
        debug=debug,
        overwrite=overwrite,
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
    )


@evaluate_cmd.command("execute")
@add_run_options
def evaluate_execute_cmd(
    benchmark: str,
    agent: str,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    max_steps: int | None,
    max_actions: int | None,
    model: str | None,
    debug: bool,
    overwrite: bool,
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
) -> None:
    """Run sessions only."""
    _ensure_installed(benchmark, agent, set_values)
    run_with(
        execute,
        benchmark=benchmark,
        agent=agent,
        agent_json=agent_json,
        agent_arg=agent_arg,
        set_values=set_values,
        subset=subset,
        tasks=tasks,
        num_tasks=num_tasks,
        model=model,
        debug=debug,
        overwrite=overwrite,
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
    )


@evaluate_cmd.command("aggregate")
@add_run_options
def evaluate_aggregate_cmd(
    benchmark: str,
    agent: str,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    max_steps: int | None,
    max_actions: int | None,
    model: str | None,
    debug: bool,
    overwrite: bool,
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
) -> None:
    """Aggregate results from completed sessions."""
    run_with(
        aggregate,
        benchmark=benchmark,
        agent=agent,
        agent_json=agent_json,
        agent_arg=agent_arg,
        set_values=set_values,
        subset=subset,
        tasks=tasks,
        num_tasks=num_tasks,
        model=model,
        debug=debug,
        overwrite=overwrite,
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
    )


@evaluate_cmd.command("session")
@add_run_options(required=False)
@click.option(
    "--config",
    "session_config_path",
    help="SessionConfig JSON file for a single session.",
)
def evaluate_session_cmd(
    benchmark: str | None,
    agent: str | None,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    max_steps: int | None,
    max_actions: int | None,
    model: str | None,
    debug: bool,
    overwrite: bool,
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
    session_config_path: str | None,
) -> None:
    """Run a single session."""
    if session_config_path:
        if has_run_options(
            benchmark=benchmark,
            agent=agent,
            agent_json=agent_json,
            agent_arg=agent_arg,
            set_values=set_values,
            subset=subset,
            tasks=tasks,
            num_tasks=num_tasks,
            max_steps=max_steps,
            max_actions=max_actions,
            model=model,
            debug=debug,
            overwrite=overwrite,
            log_level=log_level,
            output_dir=output_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            max_workers=max_workers,
        ):
            raise click.ClickException("Do not pass run options together with --config.")
        session_config = SessionConfig.model_validate(_load_config_file(session_config_path))
        run_config = _run_config_from_session(session_config)
        execute(run_config)
        return
    if not benchmark or not agent:
        raise click.ClickException("--benchmark and --agent are required.")
    if num_tasks is not None:
        raise click.ClickException("Use --task instead of --num-tasks for sessions.")
    if len(tasks) != 1:
        raise click.ClickException("Exactly one --task is required for sessions.")
    _ensure_installed(benchmark, agent, set_values)
    run_with(
        execute,
        benchmark=benchmark,
        agent=agent,
        agent_json=agent_json,
        agent_arg=agent_arg,
        set_values=set_values,
        subset=subset,
        tasks=tasks,
        num_tasks=None,
        model=model,
        debug=debug,
        overwrite=overwrite,
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
    )


__all__ = [
    "evaluate_cmd",
    "evaluate_execute_cmd",
    "evaluate_aggregate_cmd",
    "evaluate_session_cmd",
]
