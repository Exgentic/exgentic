# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from ...core.context import get_context
from ...core.orchestrator.observer import Observer
from ...core.orchestrator.termination import RunCancelError, SessionCancelError
from ...core.types import Action, ModelSettings, SessionOutcomeStatus
from ...utils.paths import RunPaths, get_run_paths, get_session_paths
from ...utils.settings import get_settings
from .recap import RunRecapMixin
from .session_ledger import SessionLedger


class _DurationColumn(ProgressColumn):
    def render(self, task) -> Text:
        duration = task.fields.get("duration")
        if isinstance(duration, (int, float)):
            elapsed = duration
        else:
            elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(str(delta), style="progress.elapsed")


class _CountColumn(ProgressColumn):
    def render(self, task) -> Text:
        unit = task.fields.get("unit") or ""
        hide_total = bool(task.fields.get("hide_total"))
        total = task.total
        completed = int(task.completed or 0)
        if hide_total or unit == "steps":
            text = f"{completed} {unit}".strip()
        elif total is None:
            text = f"{completed} {unit}".strip()
        else:
            text = f"{completed}/{int(total)} {unit}".strip()
        return Text(text, style="progress.remaining")


class ConsoleLoggerObserver(Observer, RunRecapMixin):
    _MAX_REUSE_DURATION_SECONDS = 7 * 24 * 60 * 60
    _MAX_VISIBLE_SESSIONS = 10

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._lock = threading.Lock()
        self._ledger = SessionLedger()
        self._start_time: datetime | None = None
        self._progress: Progress | None = None
        self._run_task_id: int | None = None
        self._session_tasks: Dict[str, tuple[int, int | None]] = {}
        self._completed_session_tasks: list[int] = []
        self._run_config = None

    def on_run_start(self, run_config) -> None:
        if not self._enabled(logging.INFO):
            return
        self._run_config = run_config
        self._start_time = datetime.now()
        run_ctx = get_context()
        run_id = run_ctx.run_id
        lines = [f"[bold]Run:[/bold] [cyan]{run_id}[/cyan]"]
        overrides = {}
        if run_config.max_steps != 100:
            overrides["max_steps"] = str(run_config.max_steps)
        if run_config.max_actions != 100:
            overrides["max_actions"] = str(run_config.max_actions)
        if run_config.max_workers is not None:
            overrides["max_workers"] = str(run_config.max_workers)
        if overrides:
            for key in sorted(overrides):
                lines.append(f"[bold]{key}:[/bold] {overrides[key]}")
        body = "\n".join(lines) + "\n"
        title = Text("EXGENTIC", style="bold magenta")
        self._print(Panel(body, border_style="magenta", padding=(1, 2), title=title))
        config_panels = self._build_config_panels()
        if config_panels:
            self._print(config_panels)
        self._start_progress(run_config)

    def on_session_start(self, session, agent, observation) -> None:
        if not self._enabled(logging.INFO):
            return
        session_id = session.session_id
        session_number = self._ledger.register(session_id)
        self._start_session_progress(session_id, session_number, agent, session)

    def on_react_success(self, session, action) -> None:
        if not isinstance(action, Action):
            return
        if not self._enabled(logging.INFO):
            return
        session_id = session.session_id
        self._ledger.increment_steps(session_id)
        self._advance_session_progress(session_id)

    def on_session_success(self, session, score, agent) -> None:
        if not self._enabled(logging.INFO):
            return
        session_id = session.session_id
        session_number = self._ledger.get_number(session_id)
        limit_reached = False
        try:
            limit_reached = bool(score.session_metadata.get("limit_reached"))
        except Exception:
            limit_reached = False
        status = None
        if limit_reached and not (score.is_finished is True and bool(score.success)):
            status = SessionOutcomeStatus.LIMIT_REACHED
        outcome = self._format_outcome(
            status=status,
            success=bool(score.success),
            is_finished=score.is_finished,
        )
        if outcome == "success":
            link = self._format_path_link(session.paths.root)
            desc = f"Session {session_number} ✔ success ({link})"
            color = "green"
        else:
            link = self._format_path_link(session.paths.root)
            desc = f"Session {session_number} ⏹ {outcome} ({link})"
            color = "yellow"
        self._update_session_progress_description(session_id, f"[{color}]{desc}[/{color}]")
        self._stop_session_progress(session_id)
        self._advance_run_progress()

    def on_session_scoring(self, session) -> None:
        if not self._enabled(logging.INFO):
            return
        session_id = session.session_id
        session_number = self._ledger.get_number(session_id)
        link = self._format_path_link(session.paths.root)
        desc = f"Session {session_number} ⏳ scoring ({link})"
        self._update_session_progress_description(session_id, f"[yellow]{desc}[/yellow]")
        if self._progress is None:
            self._print(Text.from_markup(f"[yellow]{desc}[/yellow]"))
        else:
            self._progress.refresh()

    def on_session_error(self, session, error) -> None:
        session_id = session.session_id if session else None
        session_root = None
        if session is not None:
            session_root = session.paths.root
        elif session_id is not None:
            try:
                ctx = get_context()
                session_root = RunPaths.from_context(ctx).session(session_id).root
            except RuntimeError:
                pass
        session_number = self._ledger.get_number(session_id)
        if not self._enabled(logging.INFO):
            return
        if isinstance(error, (SessionCancelError, RunCancelError, KeyboardInterrupt)):
            desc = f"[yellow]Session {session_number} ⏹ cancelled"
        else:
            desc = f"[red]Session {session_number} ✖ error"
        if session_root is not None:
            link = self._format_path_link(session_root)
            desc = f"{desc} ({link})"
        if desc.startswith("[red]"):
            desc = f"{desc}[/red]"
        else:
            desc = f"{desc}[/yellow]"
        self._update_session_progress_description(session_id, desc)
        self._stop_session_progress(session_id)
        self._advance_run_progress()

    def on_session_reuse(self, session_results) -> None:
        if not self._enabled(logging.INFO):
            return
        session_id = session_results.session_id
        session_number = self._ledger.mark_reuse(session_id)
        steps = session_results.steps
        status = session_results.status
        execution_time = session_results.execution_time
        if (
            not isinstance(execution_time, (int, float))
            or execution_time < 0
            or execution_time > self._MAX_REUSE_DURATION_SECONDS
        ):
            execution_time = None
        outcome = self._format_outcome(
            status=status,
            success=session_results.success,
            is_finished=session_results.is_finished,
        )
        detail_parts = ["↺ reused", outcome]
        detail_text = " ".join(detail_parts)
        session_root = get_session_paths(session_id).root
        link = self._format_path_link(session_root)
        desc = f"[yellow]Session {session_number} {detail_text} " f"({link})[/yellow]"
        total = steps if isinstance(steps, int) and steps > 0 else 1
        self._add_completed_session_task(
            desc,
            total=total,
            duration=execution_time if isinstance(execution_time, (int, float)) else None,
        )
        self._advance_run_progress()

    def on_run_success(self, results, run_config) -> None:
        self._stop_progress()
        self._print_recap()

    def on_run_error(self, error) -> None:
        self._stop_progress()
        self._print_recap()

    def _print_recap(self) -> None:
        if not self._enabled(logging.INFO):
            return
        data = self._load_recap_data(get_run_paths().results, self._start_time)
        if data is None:
            return
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_row(
            "[bold]Sessions[/bold]",
            f"{data.total_sessions} (successes: {data.successful_sessions})",
        )
        if data.success_rate is not None:
            table.add_row(
                "[bold]Success %[/bold]",
                f"{data.success_rate:.2%}",
            )
        if data.finished_sessions is not None:
            table.add_row(
                "[bold]Finished[/bold]",
                f"{data.finished_sessions}",
            )
        table.add_row("[bold]Avg steps[/bold]", f"{data.average_steps}")
        self._print(Panel(table, border_style="magenta", title="Recap"))

        cost_table = Table(show_header=False, box=None, pad_edge=False)
        cost_table.add_row(
            "[bold]Run[/bold]",
            f"{self._format_money(data.run_cost)}",
        )
        cost_table.add_row(
            "[bold]Avg agent[/bold]",
            f"{self._format_money(data.avg_agent_cost)}",
        )
        self._print(Panel(cost_table, border_style="magenta", title="Costs"))

        results_table = Table(show_header=False, box=None, pad_edge=False)
        results_table.add_row("[bold]Results[/bold]", f"{data.results_path}")
        self._print(Panel(results_table, border_style="magenta", title="Results"))

    def _enabled(self, level: int) -> bool:
        configured = logging._nameToLevel.get(get_settings().log_level.upper(), logging.INFO)
        return level >= configured

    def _format_value(self, value) -> str:
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=True)
            except TypeError:
                return str(value)
        return str(value)

    def _build_config_panels(self):
        if self._run_config is None:
            return None
        bench_overrides = dict(self._run_config.benchmark_kwargs or {})
        agent_overrides = dict(self._run_config.agent_kwargs or {})
        model_value = self._run_config.model or agent_overrides.get("model") or "unknown"
        model_settings = agent_overrides.pop("model_settings", None)
        agent_overrides = {
            "model": str(model_value),
            **agent_overrides,
        }
        if model_settings:
            if hasattr(model_settings, "model_dump"):
                model_settings = model_settings.model_dump(exclude_none=True)
            if isinstance(model_settings, dict):
                default_settings = ModelSettings().model_dump(exclude_none=True)
                for key, value in model_settings.items():
                    if value is None:
                        continue
                    if default_settings.get(key) == value:
                        continue
                    agent_overrides[f"model.{key}"] = value

        bench_name = self._run_config.benchmark
        agent_name = self._run_config.agent
        bench_panel = self._build_config_panel(f"Benchmark: {bench_name}", bench_overrides, border_style="cyan")
        agent_panel = self._build_config_panel(f"Agent: {agent_name}", agent_overrides, border_style="green")
        console_width = self._console.width
        gap = 2
        panel_width = max(20, (console_width - gap) // 2)
        bench_panel.width = panel_width
        agent_panel.width = panel_width
        return Columns([bench_panel, agent_panel], equal=True, expand=True)

    def _build_config_panel(
        self,
        title: str,
        overrides: Dict[str, str],
        *,
        border_style: str = "magenta",
    ) -> Panel:
        table = Table(show_header=False, box=None, pad_edge=False)
        if overrides:
            for key in sorted(overrides):
                table.add_row(f"[bold]{key}[/bold]", self._format_value(overrides[key]))
        else:
            table.add_row("[dim]no overrides[/dim]", "")
        return Panel(table, border_style=border_style, title=title)

    def _format_money(self, value: float | None) -> str:
        if value is None:
            return "-"
        return f"${value:.1f}"

    def _format_score(self, value) -> str:
        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)

    def _format_outcome(
        self,
        *,
        status=None,
        success: bool | None = None,
        is_finished: bool | None = None,
    ) -> str:
        if status is not None:
            return str(status)
        if is_finished is False:
            return "unfinished"
        if is_finished is True:
            return "success" if success else "unsuccessful"
        return "unknown"

    @staticmethod
    def _format_path_link(path, *, max_len: int = 80) -> str:
        text = str(path)
        if len(text) > max_len and max_len > 3:
            text = "..." + text[-(max_len - 3) :]
        try:
            target = str(Path(path).resolve())
        except Exception:
            return text
        return f"[link={target}]{text}[/link]"

    def _print(self, renderable) -> None:
        with self._lock:
            if self._progress is not None:
                self._progress.console.print(renderable)
            else:
                self._console.print(renderable)

    def _start_progress(self, run_config) -> None:
        total = None
        if run_config is not None:
            if run_config.task_ids:
                total = len(run_config.task_ids)
                if run_config.num_tasks is not None:
                    total = min(total, int(run_config.num_tasks))
            elif run_config.num_tasks is not None:
                total = int(run_config.num_tasks)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=None),
            _CountColumn(),
            _DurationColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._run_task_id = self._progress.add_task("Run", total=total if total else None, unit="sessions")

    def _start_session_progress(self, session_id: str, session_number: int, agent, session) -> None:
        if self._progress is None:
            return
        total = agent.max_steps
        task_id = self._progress.add_task(
            f"Session {session_number} ({self._format_path_link(session.paths.root)})",
            total=total if total else None,
            unit="steps",
            hide_total=True,
        )
        self._session_tasks[session_id] = (task_id, total)

    def _update_session_progress_description(self, session_id: str, description: str) -> None:
        if self._progress is None:
            return
        entry = self._session_tasks.get(session_id)
        if entry is None:
            return
        task_id, _ = entry
        self._progress.update(task_id, description=description)

    def _add_completed_session_task(
        self,
        description: str,
        total: int,
        duration: float | None = None,
    ) -> None:
        if self._progress is None:
            return
        task_id = self._progress.add_task(
            description,
            total=total,
            unit="steps",
            duration=duration,
        )
        self._progress.update(task_id, completed=total)
        self._progress.stop_task(task_id)
        self._track_completed_session_task(task_id)

    def _advance_session_progress(self, session_id: str) -> None:
        if self._progress is None:
            return
        entry = self._session_tasks.get(session_id)
        if entry is None:
            return
        task_id, _ = entry
        self._progress.update(task_id, advance=1)

    def _stop_session_progress(self, session_id: str) -> None:
        if self._progress is None:
            return
        entry = self._session_tasks.pop(session_id, None)
        if entry is None:
            return
        task_id, total = entry
        steps = self._ledger.get_steps(session_id)
        if steps > 0:
            self._progress.update(task_id, total=steps, completed=steps)
        else:
            self._progress.update(task_id, completed=0)
        self._progress.stop_task(task_id)
        self._track_completed_session_task(task_id)

    def _track_completed_session_task(self, task_id: int) -> None:
        if self._progress is None:
            return
        self._completed_session_tasks.append(task_id)
        excess = len(self._completed_session_tasks) - self._MAX_VISIBLE_SESSIONS
        if excess <= 0:
            return
        for _ in range(excess):
            old_id = self._completed_session_tasks.pop(0)
            if old_id == self._run_task_id:
                continue
            try:
                self._progress.remove_task(old_id)
            except Exception:
                continue

    def _advance_run_progress(self) -> None:
        if self._progress is None or self._run_task_id is None:
            return
        self._progress.update(self._run_task_id, advance=1)

    def _stop_progress(self) -> None:
        if self._progress is None:
            return
        self._progress.stop()
        self._progress = None
        self._run_task_id = None
        self._session_tasks = {}
