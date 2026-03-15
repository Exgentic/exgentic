# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

import rich_click as click

from ....core.types import RunConfig, SessionConfig
from ....utils.installation_tracker import is_installed
from ...lib.api import aggregate, evaluate, execute
from ..options import add_run_options, has_run_options, run_with


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

    # Check if benchmark and agent are installed
    errors = []
    if not is_installed(benchmark, "benchmark"):
        errors.append(
            f"Benchmark '{benchmark}' has not been set up. "
            f"Run 'exgentic setup --benchmark {benchmark}' to install it."
        )
    if not is_installed(agent, "agent"):
        errors.append(f"Agent '{agent}' has not been set up. " f"Run 'exgentic setup --agent {agent}' to install it.")
    if errors:
        raise click.ClickException("\n".join(errors))

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
