# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

import rich_click as click

from ....core.types import RunConfig, SessionConfig
from ...lib.api import preview, results, status
from ..options import add_run_options, has_run_options, run_query
from ..render import render_run_plan, render_run_results, render_run_status


@click.command("status")
@add_run_options(required=False, include_overwrite=False, include_max_workers=False)
@click.option(
    "--config",
    "config_path",
    help="RunConfig or SessionConfig JSON file",
)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def status_cmd(
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
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    config_path: str | None,
    output_format: str,
) -> None:
    """Show current run status."""
    if config_path:
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
            overwrite=False,
            log_level=log_level,
            output_dir=output_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            max_workers=None,
        ):
            raise click.ClickException("Do not pass run options together with --config.")
        payload = _load_config_file(config_path)
        try:
            config = RunConfig.model_validate(payload)
        except Exception:
            session_config = SessionConfig.model_validate(payload)
            config = _run_config_from_session(session_config)
        render_run_status(status(config), output_format)
        return
    if not benchmark or not agent:
        raise click.ClickException("--benchmark and --agent are required.")
    run_query(
        run_func=status,
        render_func=render_run_status,
        output_format=output_format,
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
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
    )


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


@click.command("preview")
@add_run_options(include_overwrite=False, include_max_workers=False)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def preview_cmd(
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
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    output_format: str,
) -> None:
    """Show the planned execution for a run."""
    run_query(
        run_func=preview,
        render_func=render_run_plan,
        output_format=output_format,
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
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
    )


@click.command("results")
@add_run_options(include_overwrite=False, include_max_workers=False)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def results_cmd(
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
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    output_format: str,
) -> None:
    """Load the saved run results."""
    run_query(
        run_func=results,
        render_func=render_run_results,
        output_format=output_format,
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
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
    )


__all__ = ["status_cmd", "preview_cmd", "results_cmd"]
