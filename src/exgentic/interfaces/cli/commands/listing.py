# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import rich_click as click

from ...lib.api import list_agents, list_benchmarks, list_subsets, list_tasks
from ..options import apply_debug_mode
from ..render import render_list, render_named_list


@click.group("list")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
def list_cmd(debug: bool) -> None:
    """List available resources."""
    apply_debug_mode(debug)


@list_cmd.command("benchmarks")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def list_benchmarks_cmd(debug: bool, output_format: str) -> None:
    """List available benchmarks."""
    apply_debug_mode(debug)
    items = list_benchmarks()
    render_named_list(items, output_format, title="Benchmarks")


@list_cmd.command("agents")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def list_agents_cmd(debug: bool, output_format: str) -> None:
    """List available agents."""
    apply_debug_mode(debug)
    items = list_agents()
    render_named_list(items, output_format, title="Agents")


@list_cmd.command("subsets")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option("--benchmark", required=True, help="Benchmark slug_name")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def list_subsets_cmd(debug: bool, benchmark: str, output_format: str) -> None:
    """List available subsets for a benchmark."""
    apply_debug_mode(debug)
    try:
        subsets = list_subsets(benchmark)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    render_list(subsets, output_format, title=f"Subsets ({benchmark})")


@list_cmd.command("tasks")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option("--benchmark", required=True, help="Benchmark slug_name")
@click.option("--subset", help="Benchmark subset name")
@click.option("--limit", type=int, help="Limit output to first N tasks")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text")
def list_tasks_cmd(
    debug: bool,
    benchmark: str,
    subset: str | None,
    limit: int | None,
    output_format: str,
) -> None:
    """List task ids for a benchmark."""
    apply_debug_mode(debug)
    try:
        tasks = list_tasks(benchmark=benchmark, subset=subset)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if limit is not None:
        tasks = tasks[: int(limit)]
    label = f"Tasks ({benchmark})" if subset is None else f"Tasks ({benchmark}:{subset})"
    render_list(tasks, output_format, title=label)


__all__ = [
    "list_cmd",
    "list_benchmarks_cmd",
    "list_agents_cmd",
    "list_subsets_cmd",
    "list_tasks_cmd",
]
