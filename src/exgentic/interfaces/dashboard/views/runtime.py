# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from nicegui import ui

from ....core.context import run_scope
from ....core.types import RunConfig
from ....observers.handlers.dashboard_events import DashboardEventsObserver
from ....utils.paths import RunPaths
from ....utils.settings import get_settings
from ...lib.api import (
    evaluate,
    get_agent_info,
    get_benchmark_info,
    load_agent_class,
    load_benchmark_class,
)
from .data import (
    _build_history_sessions,
    _load_history_turns,
    _load_run_context,
    _open_session_from_row,
    _resolve_tab_label,
    get_display_mappings,
    load_leaderboard_data,
)
from .forms import _build_agent_form, _build_pydantic_form, _collect_values
from .panels import (
    _render_dive_panel,
    _render_overview_panel,
    _render_run_log,
)
from .state import LEADERBOARD_COLUMNS, RunState, RunViews
from .status import _status_from_outcome

_LOG = logging.getLogger("exgentic.interfaces.dashboard")
if not _LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("Dashboard | %(message)s"))
    _LOG.addHandler(_handler)
_LOG.setLevel(logging.INFO)
_LOG.propagate = False


def _init_tabs(
    state: RunState,
    scope: str,
    labels: list[str],
) -> tuple[Any, dict[str, Any], Any]:
    if scope not in state.active_tabs:
        state.active_tabs[scope] = labels[0]

    with ui.tabs() as tabs:
        tab_by_name = {label: ui.tab(label) for label in labels}

    state.tabs_controls[scope] = tabs
    state.tabs_by_scope[scope] = tab_by_name

    def _on_tab_change(e) -> None:
        value = getattr(e, "value", None)
        state.active_tabs[scope] = _resolve_tab_label(value, tab_by_name, labels[0])

    tabs.on_value_change(_on_tab_change)

    active_label = state.active_tabs.get(scope, labels[0])
    active_tab = tab_by_name.get(active_label, tab_by_name[labels[0]])
    return tabs, tab_by_name, active_tab


def process_new_events(state: RunState) -> bool:
    tracker = state.tracker
    if tracker is None:
        return False

    events_to_process = []
    event_count = 0
    max_events = 40

    while not tracker.events.empty() and event_count < max_events:
        try:
            evt = tracker.events.get_nowait()
            events_to_process.append(evt)
            event_count += 1
        except Exception:
            break

    if not events_to_process:
        return False

    sessions = state.sessions
    turns = state.turns

    for evt in events_to_process:
        state.events.append(evt)
        et = evt.get("type")

        if et == "session_started":
            sid = evt.get("session_id")
            if sid:
                sessions[sid] = {
                    "status": "running",
                    "steps": 0,
                    "success": None,
                    "is_finished": None,
                    "error_source": None,
                }

        elif et == "step":
            sid = evt.get("session_id")
            if sid and sid in sessions:
                sessions[sid]["steps"] = sessions[sid].get("steps", 0) + 1
                if "execution_time" in evt:
                    sessions[sid]["execution_time"] = evt.get("execution_time")
                if "agent_cost" in evt:
                    sessions[sid]["agent_cost"] = evt.get("agent_cost")
                if "benchmark_cost" in evt:
                    sessions[sid]["benchmark_cost"] = evt.get("benchmark_cost")
                step_no = evt.get("n")
                act_obj = evt.get("action_obj")
                if act_obj is not None:
                    turns.setdefault(sid, []).append({"type": "action", "step": step_no, "content": act_obj})
                    turns[sid] = turns[sid][-200:]

        elif et == "observation":
            sid = evt.get("session_id")
            if sid and sid in sessions:
                step_no = evt.get("step")
                obs_obj = evt.get("observation")
                if "execution_time" in evt:
                    sessions[sid]["execution_time"] = evt.get("execution_time")
                if "agent_cost" in evt:
                    sessions[sid]["agent_cost"] = evt.get("agent_cost")
                if "benchmark_cost" in evt:
                    sessions[sid]["benchmark_cost"] = evt.get("benchmark_cost")
                if obs_obj is not None:
                    turns.setdefault(sid, []).append({"type": "observation", "step": step_no, "content": obs_obj})
                    turns[sid] = turns[sid][-200:]

        elif et == "session_finished":
            sid = evt.get("session_id")
            if sid and sid in sessions:
                success = evt.get("success", False)
                is_finished = evt.get("is_finished")
                error_source = evt.get("error_source")
                sessions[sid]["status"] = _status_from_outcome(success, is_finished, error_source)
                sessions[sid]["success"] = success
                sessions[sid]["score"] = evt.get("score")
                if "steps" in evt:
                    sessions[sid]["steps"] = evt.get("steps")
                if "execution_time" in evt:
                    sessions[sid]["execution_time"] = evt.get("execution_time")
                if "agent_cost" in evt:
                    sessions[sid]["agent_cost"] = evt.get("agent_cost")
                if "benchmark_cost" in evt:
                    sessions[sid]["benchmark_cost"] = evt.get("benchmark_cost")
                if "is_finished" in evt:
                    sessions[sid]["is_finished"] = is_finished
                if "error_source" in evt:
                    sessions[sid]["error_source"] = error_source
                details = evt.get("details") or {}
                if isinstance(details, dict):
                    sessions[sid]["error"] = details.get("error")
                if error_source is not None:
                    error_msg = details.get("error") if isinstance(details, dict) else None
                    turns.setdefault(sid, []).append(
                        {
                            "type": "error",
                            "step": sessions[sid].get("steps", 0),
                            "content": error_msg or "Session ended with an error.",
                        }
                    )

        elif et == "saved":
            ctx_mgr = state.run_context_manager
            if ctx_mgr is not None:
                try:
                    ctx_mgr.__exit__(None, None, None)
                except Exception:
                    pass
                state.run_context_manager = None
            state.run_active = False
            state.refresh_needed = True
            _LOG.info("run saved; dashboard state updated")

        elif et == "run_meta":
            run_id = evt.get("run_id")
            if run_id:
                state.run_id = run_id
                _LOG.info("run_meta received: %s", run_id)

    return True


@ui.refreshable
def bench_form_panel(state: RunState) -> None:
    bench_key = state.bench_key
    if bench_key is None:
        ui.label("No benchmark selected")
        return
    try:
        bench_cls = load_benchmark_class(bench_key)
    except Exception as exc:
        ui.label(f"Failed to load benchmark '{bench_key}': {exc}")
        return
    state.bench_controls = _build_pydantic_form(bench_cls, disabled=state.run_active)


@ui.refreshable
def agent_form_panel(state: RunState) -> None:
    agent_key = state.agent_key
    if agent_key is None:
        ui.label("No agent selected")
        return
    try:
        agent_cls = load_agent_class(agent_key)
    except Exception as exc:
        ui.label(f"Failed to load agent '{agent_key}': {exc}")
        return
    state.agent_controls = _build_agent_form(agent_cls, disabled=state.run_active)


@ui.refreshable
def run_log_panel(state: RunState) -> None:
    _render_run_log(state.run_id)


@ui.refreshable
def overview_panel(state: RunState) -> None:
    fallback_benchmark = None
    if state.bench_key:
        try:
            fallback_benchmark = get_benchmark_info(state.bench_key)["display_name"]
        except Exception:
            fallback_benchmark = None
    fallback_agent = None
    if state.agent_key:
        try:
            fallback_agent = get_agent_info(state.agent_key)["display_name"]
        except Exception:
            fallback_agent = None
    context = _load_run_context(
        state.run_id,
        fallback_benchmark=fallback_benchmark,
        fallback_agent=fallback_agent,
        planned_fallback=state.planned_sessions,
        workers_fallback=state.max_workers if state.run_active else None,
    )

    def _open_session(row: dict) -> None:
        _open_session_from_row(state, "run", row, dive_panel.refresh)

    _render_overview_panel(
        state.sessions,
        run_active=state.run_active,
        run_id=state.run_id,
        results=context.results,
        planned_sessions=context.planned_sessions,
        total_workers=context.total_workers,
        run_meta=context.run_meta,
        on_open_session=_open_session,
    )


@ui.refreshable
def dive_panel(state: RunState) -> None:
    sessions = sorted(state.sessions.keys())
    if not sessions:
        ui.label("No sessions to inspect yet.")
        return

    if state.selected_session not in sessions:
        state.selected_session = sessions[0]

    def on_session_change(e) -> None:
        state.selected_session = e.value
        dive_panel.refresh()

    _render_dive_panel(
        state.sessions,
        state.turns,
        run_id=state.run_id,
        selected_session=state.selected_session,
        on_session_change=on_session_change,
    )


@ui.refreshable
def leaderboard_panel(state: RunState) -> None:
    settings = get_settings()
    rows = load_leaderboard_data(settings.output_dir)
    rows = sorted(rows, key=lambda r: r.get("run_id", ""), reverse=True)
    if not rows:
        ui.label("No runs found yet.")
        return

    def _norm(value: object) -> str:
        if value is None:
            return "unknown"
        text = str(value).strip()
        return text or "unknown"

    agents = sorted({_norm(row.get("Agent")) for row in rows})
    models = sorted({_norm(row.get("Model")) for row in rows})
    benchmarks = sorted({_norm(row.get("Benchmark")) for row in rows})
    subsets = sorted({_norm(row.get("Subset")) for row in rows})

    if not state.selected_agents:
        state.selected_agents = agents
    if not state.selected_models:
        state.selected_models = models
    if not state.selected_benchmarks:
        state.selected_benchmarks = benchmarks
    if not state.selected_subsets:
        state.selected_subsets = subsets

    def on_agents_change(e) -> None:
        state.selected_agents = e.value or []
        leaderboard_panel.refresh()

    def on_models_change(e) -> None:
        state.selected_models = e.value or []
        leaderboard_panel.refresh()

    def on_benchmarks_change(e) -> None:
        state.selected_benchmarks = e.value or []
        leaderboard_panel.refresh()

    def on_subsets_change(e) -> None:
        state.selected_subsets = e.value or []
        leaderboard_panel.refresh()

    def on_min_tasks_change(e) -> None:
        try:
            state.min_tasks = int(e.value)
        except Exception:
            state.min_tasks = 0
        leaderboard_panel.refresh()

    with ui.card().classes("w-full card p-4"):
        with ui.row().classes("w-full"):
            ui.select(
                agents,
                value=state.selected_agents,
                label="Agent",
                multiple=True,
            ).props("dense").on_value_change(on_agents_change)
            ui.select(
                models,
                value=state.selected_models,
                label="Model",
                multiple=True,
            ).props("dense").on_value_change(on_models_change)
            ui.select(
                benchmarks,
                value=state.selected_benchmarks,
                label="Benchmark",
                multiple=True,
            ).props("dense").on_value_change(on_benchmarks_change)
            ui.select(
                subsets,
                value=state.selected_subsets,
                label="Subset",
                multiple=True,
            ).props("dense").on_value_change(on_subsets_change)
            ui.number(
                label="Min Tasks",
                value=state.min_tasks,
                min=0,
                step=1,
            ).props("dense").on_value_change(on_min_tasks_change)

    def _include(row: dict) -> bool:
        if row.get("Agent", "unknown") not in state.selected_agents:
            return False
        if row.get("Model", "unknown") not in state.selected_models:
            return False
        if row.get("Benchmark", "unknown") not in state.selected_benchmarks:
            return False
        if row.get("Subset", "unknown") not in state.selected_subsets:
            return False
        try:
            return int(row.get("Num Tasks", 0)) >= state.min_tasks
        except Exception:
            return False

    filtered = [row for row in rows if _include(row)]
    with ui.card().classes("w-full card p-4"):
        ui.table(columns=LEADERBOARD_COLUMNS, rows=filtered, row_key="run_id").classes("w-full").props("flat")


@ui.refreshable
def history_panel(state: RunState) -> None:
    settings = get_settings()

    def _normalize_history_root(value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return settings.output_dir
        return os.path.abspath(os.path.expanduser(raw))

    history_root = _normalize_history_root(state.history_root)
    runs = []
    if os.path.isdir(history_root):
        for name in os.listdir(history_root):
            results_path = RunPaths(run_id=name, output_dir=history_root).results
            if os.path.isfile(results_path):
                runs.append(name)
    runs = sorted(runs, reverse=True)

    if state.selected_history_run not in runs:
        state.selected_history_run = runs[0] if runs else None
        state.selected_history_session = None
        state.active_tabs["history"] = "Overview"

    def on_history_change(e) -> None:
        state.selected_history_run = e.value
        state.selected_history_session = None
        state.active_tabs["history"] = "Overview"
        history_panel.refresh()

    def on_history_root_change(e) -> None:
        state.history_root = e.value or ""
        state.selected_history_run = None
        state.selected_history_session = None
        state.active_tabs["history"] = "Overview"
        history_panel.refresh()

    def open_history_browser() -> None:
        browser_state = {"path": history_root}

        dialog = ui.dialog()

        def refresh_entries(container) -> None:
            container.clear()
            current = browser_state["path"]
            with container:
                if not os.path.isdir(current):
                    ui.label("Directory not found.").classes("text-negative")
                    return
                parent = os.path.dirname(current.rstrip(os.sep))
                entries = [name for name in sorted(os.listdir(current)) if os.path.isdir(os.path.join(current, name))]
                if parent and parent != current:
                    entries = ["..", *entries]
                if not entries:
                    ui.label("(empty)")
                    return
                for name in entries:
                    ui.button(
                        name,
                        on_click=lambda n=name: on_entry_click(n, container),
                    ).props("flat dense").classes("justify-start w-full")

        def on_entry_click(name: str, container) -> None:
            current = browser_state["path"]
            if name == "..":
                parent = os.path.dirname(current.rstrip(os.sep))
                if parent and parent != current:
                    browser_state["path"] = parent
            else:
                browser_state["path"] = os.path.join(current, name)
            path_input.value = browser_state["path"]
            refresh_entries(container)

        def on_select() -> None:
            if not os.path.isdir(browser_state["path"]):
                ui.notify("Directory not found.")
                return
            state.history_root = browser_state["path"]
            state.selected_history_run = None
            state.selected_history_session = None
            state.active_tabs["history"] = "Overview"
            history_panel.refresh()
            dialog.close()

        with dialog, ui.card().classes("w-[520px] max-w-full"):
            ui.label("Select run directory")
            with ui.row().classes("w-full items-center gap-2"):
                path_input = (
                    ui.input(
                        label="Directory",
                        value=browser_state["path"],
                    )
                    .props("dense")
                    .style("flex: 1;")
                )

            entries_box = ui.column().classes("w-full gap-2").style("max-height: 320px; overflow-y: auto;")
            refresh_entries(entries_box)

            def on_path_change(e) -> None:
                browser_state["path"] = e.value or ""
                refresh_entries(entries_box)

            path_input.on_value_change(on_path_change)

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat dense")
                ui.button("Select", on_click=on_select).props("dense").style(
                    "background:#111111 !important; color:#ffffff !important;"
                )

        dialog.open()

    def on_history_session_change(e) -> None:
        state.selected_history_session = e.value
        state.active_tabs["history"] = "Sessions"
        history_panel.refresh()

    with ui.card().classes("w-full card p-4"):
        with ui.row().classes("w-full items-center gap-3"):
            ui.input(
                label="Run directory",
                value=state.history_root or settings.output_dir,
                placeholder=settings.output_dir,
            ).props("dense").style("min-width: 360px;").on_value_change(on_history_root_change)
            ui.button(icon="folder_open").props("dense flat").on_click(open_history_browser)
        ui.select(runs, value=state.selected_history_run, label="Select run").props("dense").style(
            "min-width: 280px;"
        ).on_value_change(on_history_change)
        if not os.path.isdir(history_root):
            ui.label(f"Directory not found: {history_root}")
        if not state.selected_history_run:
            ui.label("No runs found.")
            return

    run_id = state.selected_history_run
    context = _load_run_context(run_id, output_dir=history_root)
    sessions = _build_history_sessions(run_id, context.results, output_dir=history_root)

    session_ids = sorted(sessions.keys())
    if state.selected_history_session not in session_ids:
        state.selected_history_session = session_ids[0] if session_ids else None

    turns: dict[str, list[dict]] = {}
    if state.selected_history_session:
        turns[state.selected_history_session] = _load_history_turns(
            run_id,
            state.selected_history_session,
            output_dir=history_root,
        )

    with ui.card().classes("w-full card p-4"):
        tabs, tab_by_name, active_tab = _init_tabs(state, "history", ["Overview", "Sessions", "Log"])
        overview_tab = tab_by_name["Overview"]
        sessions_tab = tab_by_name["Sessions"]
        log_tab = tab_by_name["Log"]

        with ui.tab_panels(tabs, value=active_tab).classes("w-full"):
            with ui.tab_panel(overview_tab):

                def _open_history_session(row: dict) -> None:
                    _open_session_from_row(state, "history", row, history_panel.refresh)

                _render_overview_panel(
                    sessions,
                    run_active=False,
                    run_id=run_id,
                    results=context.results,
                    planned_sessions=context.planned_sessions,
                    total_workers=context.total_workers,
                    run_meta=context.run_meta,
                    on_open_session=_open_history_session,
                )
            with ui.tab_panel(sessions_tab):
                _render_dive_panel(
                    sessions,
                    turns,
                    run_id=run_id,
                    selected_session=state.selected_history_session,
                    on_session_change=on_history_session_change,
                    output_dir=history_root,
                )
            with ui.tab_panel(log_tab):
                _render_run_log(run_id, output_dir=history_root)


def _set_enabled(control: Any, enabled: bool) -> None:
    if hasattr(control, "enabled"):
        control.enabled = enabled
        return
    if enabled and hasattr(control, "enable"):
        control.enable()
    elif not enabled and hasattr(control, "disable"):
        control.disable()


def _set_controls_enabled(controls: dict[str, Any], enabled: bool) -> None:
    for data in controls.values():
        control = data[1]
        _set_enabled(control, enabled)


def _set_visible(control: Any, visible: bool) -> None:
    if hasattr(control, "visible"):
        control.visible = visible
        return
    if visible and hasattr(control, "show"):
        control.show()
    elif not visible and hasattr(control, "hide"):
        control.hide()


def build_run_tab(state: RunState) -> RunViews:
    bench_label_to_key, agent_label_to_key = get_display_mappings()
    bench_labels = list(bench_label_to_key.keys())
    agent_labels = list(agent_label_to_key.keys())

    if state.bench_key is None and bench_labels:
        state.bench_key = bench_label_to_key[bench_labels[0]]
    if state.agent_key is None and agent_labels:
        state.agent_key = agent_label_to_key[agent_labels[0]]

    with ui.column().classes("w-full items-center"):
        with ui.column().classes("w-full max-w-6xl gap-4"):
            with ui.element("div").classes("w-full split-grid"):
                with ui.card().classes("w-full card p-4"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("Agent")
                        agent_select = (
                            ui.select(
                                agent_labels,
                                value=agent_labels[0] if agent_labels else None,
                                label="",
                            )
                            .props("dense")
                            .style("min-width: 220px;")
                        )
                    with ui.expansion("", value=False).classes("w-full"):
                        agent_form_panel(state)
                with ui.card().classes("w-full card p-4"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        ui.label("Benchmark")
                        bench_select = (
                            ui.select(
                                bench_labels,
                                value=bench_labels[0] if bench_labels else None,
                                label="",
                            )
                            .props("dense")
                            .style("min-width: 220px;")
                        )
                    with ui.expansion("", value=False).classes("w-full"):
                        bench_form_panel(state)

            with ui.card().classes("w-full card p-4"):
                with ui.row().classes("w-full gap-4 items-end"):
                    num_tasks_input = ui.number(
                        label="Num Tasks",
                        value=state.num_tasks or 5,
                        min=0,
                        step=1,
                    ).props("dense")
                    max_workers_input = ui.number(label="Parallel Workers", value=state.max_workers, min=1).props(
                        "dense"
                    )
                    start_button = (
                        ui.button("Start Run")
                        .classes("ml-auto start-run-btn")
                        .style("background:#39ff14 !important; color:#0b0f10 !important;")
                    )

            with ui.card().classes("w-full card p-4") as run_panel_box:
                run_tabs, tab_by_name, active_tab = _init_tabs(state, "run", ["Overview", "Sessions", "Log"])
                overview_tab = tab_by_name["Overview"]
                sessions_tab = tab_by_name["Sessions"]
                log_tab = tab_by_name["Log"]

                with ui.tab_panels(run_tabs, value=active_tab).classes("w-full"):
                    with ui.tab_panel(overview_tab) as overview_panel_el:
                        overview_panel(state)
                    with ui.tab_panel(sessions_tab) as sessions_panel_el:
                        dive_panel(state)
                    with ui.tab_panel(log_tab) as log_panel_el:
                        run_log_panel(state)

    def on_bench_change(e) -> None:
        label = e.value
        state.bench_key = bench_label_to_key.get(label)
        state.bench_controls = {}
        bench_form_panel.refresh()

    def on_agent_change(e) -> None:
        label = e.value
        state.agent_key = agent_label_to_key.get(label)
        agent_form_panel.refresh()

    def on_workers_change(e) -> None:
        try:
            state.max_workers = int(e.value)
        except Exception:
            state.max_workers = 1

    def on_num_tasks_change(e) -> None:
        try:
            value = int(e.value)
        except Exception:
            state.num_tasks = None
            return
        state.num_tasks = value if value > 0 else None

    bench_select.on_value_change(on_bench_change)
    agent_select.on_value_change(on_agent_change)
    num_tasks_input.on_value_change(on_num_tasks_change)
    max_workers_input.on_value_change(on_workers_change)

    show_sessions = state.run_active or bool(state.sessions)
    _set_visible(run_panel_box, show_sessions)
    _set_visible(overview_tab, show_sessions)
    _set_visible(sessions_tab, show_sessions)
    _set_visible(log_tab, show_sessions)
    _set_visible(overview_panel_el, show_sessions)
    _set_visible(sessions_panel_el, show_sessions)
    _set_visible(log_panel_el, show_sessions)

    def start_run() -> None:
        if state.run_active:
            return
        if state.bench_key is None or state.agent_key is None:
            ui.notify("Please select a benchmark and agent.")
            return

        num_tasks_value = None
        try:
            raw = num_tasks_input.value
            if raw is not None:
                parsed = int(raw)
                if parsed > 0:
                    num_tasks_value = parsed
        except Exception:
            num_tasks_value = None
        state.num_tasks = num_tasks_value

        benchmark = None
        agent = None
        try:
            bench_cls = load_benchmark_class(state.bench_key)
            agent_cls = load_agent_class(state.agent_key)
            bench_values = _collect_values(state.bench_controls)
            agent_values = _collect_values(state.agent_controls)
            benchmark = bench_cls(**bench_values)
            agent = agent_cls(**agent_values)
            state.planned_sessions = num_tasks_value if num_tasks_value else None
        except Exception as exc:
            ui.notify(f"Config error: {exc}")
            _LOG.info("config error: %s", exc)
            return
        finally:
            if benchmark is not None:
                try:
                    benchmark.close()
                except Exception:
                    _LOG.info("benchmark preview close failed")
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    _LOG.info("agent preview close failed")

        settings = get_settings()
        output_dir = settings.output_dir

        state.events = []
        state.sessions = {}
        state.turns = {}
        state.refresh_needed = True

        ctx_mgr = run_scope(output_dir=output_dir)
        ctx = ctx_mgr.__enter__()
        dashboard_observer = DashboardEventsObserver()

        state.tracker = dashboard_observer
        state.run_id = ctx.run_id
        state.run_context_manager = ctx_mgr
        state.run_active = True
        os.environ["EXGENTIC_MAX_WORKERS"] = str(state.max_workers)
        mode = "parallel" if state.max_workers > 1 else "sequential"
        _LOG.info(
            "starting run bench=%s agent=%s mode=%s workers=%s run_id=%s",
            state.bench_key,
            state.agent_key,
            mode,
            state.max_workers,
            state.run_id,
        )

        def worker() -> None:
            try:
                _LOG.info("worker started")
                config = RunConfig(
                    benchmark=state.bench_key,
                    agent=state.agent_key,
                    benchmark_kwargs=bench_values,
                    agent_kwargs=agent_values,
                    output_dir=output_dir,
                    max_workers=state.max_workers if state.max_workers > 1 else None,
                    run_id=state.run_id,
                    num_tasks=num_tasks_value,
                )
                evaluate(
                    config,
                    observers=[dashboard_observer],
                )
                _LOG.info("worker finished")
            except Exception:
                _LOG.exception("worker crashed")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        state.thread = thread

    start_button.on("click", lambda _: start_run())

    return RunViews(
        start_button=start_button,
        bench_select=bench_select,
        agent_select=agent_select,
        num_tasks_input=num_tasks_input,
        max_workers_input=max_workers_input,
        bench_form=bench_form_panel,
        agent_form=agent_form_panel,
        overview_panel=overview_panel,
        sessions_panel=dive_panel,
        run_log_panel=run_log_panel,
        leaderboard_panel=leaderboard_panel,
        history_panel=history_panel,
        run_panel_box=run_panel_box,
        overview_tab=overview_tab,
        sessions_tab=sessions_tab,
        log_tab=log_tab,
        overview_panel_el=overview_panel_el,
        sessions_panel_el=sessions_panel_el,
        log_panel_el=log_panel_el,
    )


def build_leaderboard_tab(state: RunState) -> None:
    leaderboard_panel(state)


def build_history_tab(state: RunState) -> None:
    history_panel(state)


def refresh_ui(state: RunState, views: RunViews) -> None:
    changed = process_new_events(state)

    if state.run_active and state.thread and not state.thread.is_alive():
        state.run_active = False
        if state.run_context_manager is not None:
            try:
                state.run_context_manager.__exit__(None, None, None)
            except Exception:
                pass
            state.run_context_manager = None
        state.refresh_needed = True

    _set_enabled(views.start_button, not state.run_active)
    _set_enabled(views.bench_select, not state.run_active)
    _set_enabled(views.agent_select, not state.run_active)
    _set_enabled(views.num_tasks_input, not state.run_active)
    _set_enabled(views.max_workers_input, not state.run_active)
    _set_controls_enabled(state.bench_controls, not state.run_active)
    _set_controls_enabled(state.agent_controls, not state.run_active)

    show_sessions = state.run_active or bool(state.sessions)
    _set_visible(views.run_panel_box, show_sessions)
    _set_visible(views.overview_tab, show_sessions)
    _set_visible(views.sessions_tab, show_sessions)
    _set_visible(views.log_tab, show_sessions)
    _set_visible(views.overview_panel_el, show_sessions)
    _set_visible(views.sessions_panel_el, show_sessions)
    _set_visible(views.log_panel_el, show_sessions)

    if state.last_run_active != state.run_active:
        views.bench_form.refresh()
        views.agent_form.refresh()
        state.last_run_active = state.run_active

    if changed:
        views.overview_panel.refresh()
        views.sessions_panel.refresh()
        views.run_log_panel.refresh()

    if state.refresh_needed:
        views.leaderboard_panel.refresh()
        views.history_panel.refresh()
        views.run_log_panel.refresh()
        state.refresh_needed = False
