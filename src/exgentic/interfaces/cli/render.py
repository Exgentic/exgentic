# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import sys
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

CONSOLE = Console()


def print_json(data: Any) -> None:
    CONSOLE.print_json(json.dumps(data, ensure_ascii=False, indent=2))


def render_list(items: list[str], output_format: str, *, title: str = "Items") -> None:
    if output_format == "json":
        print_json(items)
        return
    table = Table(title=title, box=box.SIMPLE, show_header=False)
    table.add_column("Item")
    if not items:
        table.add_row("[dim]none[/dim]")
    else:
        for item in items:
            table.add_row(str(item))
    CONSOLE.print(table)


def render_named_list(
    items: list[dict[str, Any]],
    output_format: str,
    *,
    fields: tuple[str, str] = ("slug_name", "display_name"),
    title: str = "Items",
) -> None:
    if output_format == "json":
        print_json(items)
        return

    # Check if items have installation info
    has_install_info = items and "installed" in items[0]

    table = Table(title=title, box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("Slug")
    table.add_column("Name")
    if has_install_info:
        table.add_column("Installed", justify="center")
        table.add_column("Installed At")

    if not items:
        if has_install_info:
            table.add_row("[dim]none[/dim]", "[dim]none[/dim]", "[dim]-[/dim]", "[dim]-[/dim]")
        else:
            table.add_row("[dim]none[/dim]", "[dim]none[/dim]")
    else:
        for item in items:
            slug = str(item[fields[0]])
            name = str(item[fields[1]])
            if has_install_info:
                installed = item.get("installed", False)
                installed_at = item.get("installed_at", "")
                status = "[green]✓[/green]" if installed else "[dim]-[/dim]"
                # Format the timestamp to be more readable
                if installed_at:
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(installed_at.replace("Z", "+00:00"))
                        installed_at = dt.strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        pass
                table.add_row(slug, name, status, installed_at if installed_at else "[dim]-[/dim]")
            else:
                table.add_row(slug, name)
    CONSOLE.print(table)


def render_model(obj: Any, output_format: str) -> None:
    if output_format == "json":
        print_json(obj.model_dump())
        return
    CONSOLE.print(str(obj))


def render_run_status(status: Any, output_format: str) -> None:
    if output_format == "json":
        render_model(status, output_format)
        return
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan")
    meta.add_column()
    meta.add_row("Run", str(status.run_id))
    meta.add_row("Results", "yes" if status.results_exists else "no")
    meta.add_row("Benchmark Results", "yes" if status.benchmark_results_exists else "no")
    CONSOLE.print(Panel(meta, title="Run Status", border_style="cyan"))

    counts = Table(
        title="Sessions",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    counts.add_column("Total", justify="right")
    counts.add_column("Completed", justify="right")
    counts.add_column("Running", justify="right")
    counts.add_column("Incomplete", justify="right")
    counts.add_column("Missing", justify="right")
    counts.add_row(
        str(status.total_tasks),
        str(status.completed_sessions),
        str(status.running_sessions),
        str(status.incomplete_sessions),
        str(status.missing_sessions),
    )
    CONSOLE.print(counts)


def render_batch_status(rows: list[dict[str, str]]) -> None:
    total_configs = len(rows)
    total_sessions = 0
    total_completed = 0
    total_running = 0
    total_errors = 0
    total_cost = 0.0
    done_configs = 0
    for row in rows:
        sessions = row.get("sessions", "-")
        if not isinstance(sessions, str) or "/" not in sessions:
            continue
        try:
            tokens = sessions.split()
            frac = tokens[0]
            completed = int(frac.split("/")[0])
            planned = int(frac.split("/")[1])
            total_completed += completed
            total_sessions += planned
            if completed >= planned:
                done_configs += 1
            for token in tokens[1:]:
                if token.endswith("run"):
                    total_running += int(token[:-3])
                elif token.endswith("err"):
                    total_errors += int(token[:-3])
        except (ValueError, IndexError):
            continue
        try:
            cost_str = row.get("cost", "0")
            if cost_str not in ("-", ""):
                total_cost += float(cost_str)
        except (ValueError, TypeError):
            pass

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("Configs", f"{done_configs}/{total_configs} done")
    sessions_line = f"{total_completed}/{total_sessions}"
    if total_running:
        sessions_line += f"  ({total_running} running)"
    summary.add_row("Sessions", sessions_line)
    if total_errors:
        summary.add_row("Errors", str(total_errors))
    if total_cost > 0:
        summary.add_row("Total Cost", f"${total_cost:.2f}")
    CONSOLE.print(Panel(summary, title="Batch Summary", border_style="cyan"))

    table = Table(
        title="Batch Status",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Benchmark")
    table.add_column("Agent")
    table.add_column("Model")
    table.add_column("Sessions", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)

    for row in rows:
        table.add_row(
            row["#"],
            row["benchmark"],
            row["agent"],
            row.get("model", "-"),
            row.get("sessions", "-"),
            row.get("score", "-"),
            row.get("cost", "-"),
        )

    CONSOLE.print(table)


def render_run_plan(plan: Any, output_format: str) -> None:
    if output_format == "json":
        render_model(plan, output_format)
        return
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan")
    meta.add_column()
    meta.add_row("Run", str(plan.run_config.run_id))
    meta.add_row("Overwrite", "yes" if plan.overwrite_sessions else "no")
    CONSOLE.print(Panel(meta, title="Run Plan", border_style="cyan"))

    counts = Table(
        title="Sessions",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    counts.add_column("Total", justify="right")
    counts.add_column("To Run", justify="right")
    counts.add_column("Reuse", justify="right")
    counts.add_column("Running", justify="right")
    counts.add_column("Missing", justify="right")
    counts.add_column("Incomplete", justify="right")
    total_sessions = len(plan.to_run) + len(plan.reuse) + len(plan.running) + len(plan.missing) + len(plan.incomplete)
    counts.add_row(
        str(total_sessions),
        str(len(plan.to_run)),
        str(len(plan.reuse)),
        str(len(plan.running)),
        str(len(plan.missing)),
        str(len(plan.incomplete)),
    )
    CONSOLE.print(counts)


def render_run_results(results: Any, output_format: str) -> None:
    if output_format == "json":
        render_model(results, output_format)
        return
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan")
    meta.add_column()
    meta.add_row("Benchmark", str(results.benchmark_name))
    meta.add_row("Agent", str(results.agent_name))
    CONSOLE.print(Panel(meta, title="Run Results", border_style="cyan"))

    summary = Table(
        title="Summary",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )
    summary.add_column("Sessions", justify="right")
    summary.add_column("Successes", justify="right")
    summary.add_column("Final Score", justify="right")
    summary.add_column("Avg Score", justify="right")
    summary.add_row(
        str(results.total_sessions),
        str(results.successful_sessions),
        str(results.benchmark_score),
        str(results.average_score),
    )
    CONSOLE.print(summary)


def should_print_banner() -> bool:
    args = sys.argv[1:]
    if not args:
        return True
    return any(arg in ("-h", "--help") for arg in args)


def print_banner() -> None:
    title = Align.center("[bold magenta]EXGENTIC[/bold magenta]\n" "[dim]General Agent Evaluation[/dim]")
    CONSOLE.print(Panel(title, border_style="magenta", padding=(1, 8)))
