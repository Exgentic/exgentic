# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from nicegui import ui

from ....utils.paths import RunPaths
from ....utils.settings import get_settings
from .data import (
    _build_session_rows,
    _list_text_files,
    _load_text_file,
    load_run_config,
    load_session_file,
)
from .formatting import (
    _format_error_message,
    _format_payload,
    _format_value,
    _normalize_action_payload,
    _normalize_observation_payload,
)
from .state import SESSION_COLUMNS
from .status import _render_status_pie, _status_counts_from_sessions


def _metric_help_text(label: str) -> Optional[str]:
    return {
        "Benchmark": "Benchmark used for this run.",
        "Agent": "Agent used for this run.",
        "Models": "LLMs used in this run.",
        "Model": "Primary model used in this run.",
        "Subset": "Benchmark subset for this run.",
        "Running": "Currently running sessions / worker capacity.",
        "Completed": "Finished sessions / planned sessions.",
        "Total Sessions": "Total sessions recorded for this run.",
        "Successful": "Number of successful sessions.",
        "Benchmark Score": "Aggregate benchmark score for the run.",
        "Avg Score": "Average score across completed sessions.",
        "Avg Steps": "Average steps per session.",
        "Avg Agent Cost": "Average agent cost per session.",
        "Avg Benchmark Cost": "Average benchmark cost per session.",
        "Total Agent Cost": "Total agent cost across sessions.",
        "Total Benchmark Cost": "Total benchmark cost across sessions.",
        "Total Run Cost": "Total agent + benchmark cost.",
        "Finished": "Share of sessions marked finished.",
        "Success Rate": "Share of successful sessions.",
        "Finished Rate": "Share of sessions marked finished.",
        "Success %": "Share of successful sessions.",
        "Finished %": "Share of sessions marked finished.",
        "Status": (
            "Session status (running/success/unsuccessful/unfinished/" "agent error/benchmark error/cancelled/error)."
        ),
        "Steps": "Number of actions taken in the session.",
        "Score": "Session score from the benchmark.",
        "Exec Time (s)": "Elapsed time for the session.",
        "Agent Cost": "Cost attributed to the agent.",
        "Benchmark Cost": "Cost attributed to the benchmark.",
    }.get(label)


def _metric_info(help_text: str) -> None:
    ui.icon("info").classes("metric-info").tooltip(help_text)


def _metric_card(label: str, value: Any) -> None:
    with ui.element("div").classes("metric-card"):
        help_text = _metric_help_text(label)
        if help_text:
            _metric_info(help_text)
        ui.label(label).classes("metric-label")
        ui.label(_format_value(value)).classes("metric-value")


def _metric_card_small(label: str, value: Any) -> None:
    display_value = value if isinstance(value, str) else _format_value(value)
    with ui.element("div").classes("metric-card metric-card-sm"):
        help_text = _metric_help_text(label)
        if help_text:
            _metric_info(help_text)
        ui.label(label).classes("metric-label")
        ui.label(display_value).classes("metric-value")


def _metric_card_large(label: str, value: Any) -> None:
    display_value = value if isinstance(value, str) else _format_value(value)
    with ui.element("div").classes("metric-card metric-card-lg"):
        help_text = _metric_help_text(label)
        if help_text:
            _metric_info(help_text)
        ui.label(label).classes("metric-label")
        ui.label(display_value).classes("metric-value")


def _render_run_log(run_id: Optional[str], *, output_dir: Optional[str] = None) -> None:
    if not run_id:
        ui.label("No run log found.")
        return
    resolved_output = output_dir or get_settings().output_dir
    run_paths = RunPaths(run_id=run_id, output_dir=resolved_output)
    log_content = load_session_file(str(run_paths.tracker), "log")
    if log_content is None:
        legacy_path = run_paths.root / "tracker.log"
        if legacy_path != run_paths.tracker:
            log_content = load_session_file(str(legacy_path), "log")
    if log_content is None:
        ui.label("No run log found.")
        return
    ui.code(log_content, language="text").classes("w-full")


def _render_action_entry(payload: Any) -> None:
    data = _normalize_action_payload(payload)
    if "raw" in data:
        ui.code(_format_payload(data["raw"]), language="json").classes("w-full")
        return
    if "actions" in data:
        mode = str(data.get("mode", "parallel")).title()
        actions = data.get("actions") or []
        ui.label(f"{mode} actions ({len(actions)})").classes("metric")
        if not actions:
            ui.label("No actions provided.")
            return
        with ui.column().classes("w-full gap-3"):
            for idx, item in enumerate(actions, start=1):
                name = item.get("name") or f"Action {idx}"
                args = item.get("arguments")
                with ui.element("div").classes("trajectory-box"):
                    ui.label(name).classes("trajectory-title")
                    if args is None:
                        ui.label("No arguments.")
                    else:
                        ui.code(_format_payload(args), language="json").classes("w-full")
        return
    name = data.get("name") or "Action"
    args = data.get("arguments")
    if data.get("name") is None and data.get("arguments") is None:
        ui.label("Empty action.")
        return
    with ui.element("div").classes("trajectory-box"):
        ui.label(name).classes("trajectory-title")
        if args is None:
            ui.label("No arguments.")
        else:
            ui.code(_format_payload(args), language="json").classes("w-full")


def _render_observation_entry(payload: Any) -> None:
    data = _normalize_observation_payload(payload)
    if data is None:
        ui.label("Empty observation.")
        return
    if isinstance(data, list):
        ui.label(f"Observations ({len(data)})").classes("metric")
        if not data:
            ui.label("No observations provided.")
            return
        with ui.column().classes("w-full gap-3"):
            for item in data:
                if item is None:
                    ui.label("Empty observation.")
                    continue
                if isinstance(item, dict) and "sender" in item and "message" in item:
                    sender = item.get("sender") or "Sender"
                    message = item.get("message") or ""
                    with ui.element("div").classes("trajectory-box"):
                        ui.label(str(sender)).classes("trajectory-title")
                        ui.code(str(message), language="text").classes("w-full")
                    continue
                with ui.element("div").classes("trajectory-box"):
                    ui.code(_format_payload(item), language="json").classes("w-full")
        return
    if isinstance(data, dict) and "sender" in data and "message" in data:
        sender = data.get("sender") or "Sender"
        message = data.get("message") or ""
        with ui.element("div").classes("trajectory-box"):
            ui.label(str(sender)).classes("trajectory-title")
            ui.code(str(message), language="text").classes("w-full")
        return
    ui.code(_format_payload(data), language="json").classes("w-full")


def _render_trajectory_entry(kind: str, payload: Any) -> None:
    if kind == "action":
        _render_action_entry(payload)
        return
    if kind == "observation":
        _render_observation_entry(payload)
        return
    if kind == "error":
        ui.code(_format_payload(payload), language="json").classes("w-full")
        return
    ui.code(_format_payload(payload), language="json").classes("w-full")


def _render_overview_panel(
    sessions: dict,
    *,
    run_active: bool,
    run_id: Optional[str],
    results: Optional[dict] = None,
    planned_sessions: int | None = None,
    total_workers: int | None = None,
    run_meta: Optional[dict[str, Any]] = None,
    on_open_session=None,
) -> None:
    if run_meta:
        models = run_meta.get("models") or []
        models_text = ", ".join(models) if models else "-"
        with ui.element("div").classes("w-full metric-grid"):
            _metric_card_large("Benchmark", run_meta.get("benchmark"))
            _metric_card_large("Agent", run_meta.get("agent"))
            _metric_card_large("Models", models_text)
    if not sessions:
        if run_active:
            ui.label("Waiting for sessions...").classes("text-sm muted")
        else:
            ui.label("No sessions found.")
        return
    total = len(sessions)
    running = sum(1 for s in sessions.values() if s.get("status") == "running")
    done = total - running
    if planned_sessions is None:
        planned_sessions = total
    running_total = total_workers if total_workers is not None else "-"
    benchmark_score = results.get("benchmark_score") if isinstance(results, dict) else None
    with ui.element("div").classes("w-full metric-grid"):
        _metric_card("Running", f"{running}/{running_total}")
        _metric_card("Completed", f"{done}/{planned_sessions}")
        _metric_card("Benchmark Score", _format_value(benchmark_score))

    from .data import _build_overview_secondary_metrics

    secondary_metrics = _build_overview_secondary_metrics(sessions, results)
    with ui.element("div").classes("w-full metric-grid"):
        for label, value in secondary_metrics:
            _metric_card_small(label, value)

    with ui.card().classes("w-full card p-4"):
        ui.label("Status Breakdown").classes("section-title")
        status_counts = _status_counts_from_sessions(sessions)
        _render_status_pie(status_counts)

    with ui.card().classes("w-full card p-4"):
        rows = _build_session_rows(sessions)
        selection = "single" if on_open_session else None

        def _handle_select(e) -> None:
            selection_rows = getattr(e, "selection", None)
            if not selection_rows:
                return
            row = selection_rows[-1]
            if isinstance(row, dict):
                on_open_session(row)

        (
            ui.table(
                columns=SESSION_COLUMNS,
                rows=rows,
                row_key="session",
                selection=selection,
                on_select=_handle_select if on_open_session else None,
            )
            .classes("w-full")
            .props("flat")
        )


def _render_tabs(
    labels: list[str],
    renderers: dict[str, Any],
    *,
    default: Optional[str] = None,
) -> None:
    default_label = default or labels[0]
    with ui.tabs() as tabs:
        tab_by_name = {label: ui.tab(label) for label in labels}

    with ui.tab_panels(tabs, value=tab_by_name[default_label]).classes("w-full"):
        for label in labels:
            with ui.tab_panel(tab_by_name[label]):
                renderers[label]()


def _render_session_results_overview(meta: dict) -> None:
    with ui.element("div").classes("w-full metric-grid"):
        _metric_card("Status", meta.get("status", "-"))
        _metric_card("Steps", meta.get("steps", 0))
        _metric_card("Score", meta.get("score", "-"))
    error_msg = meta.get("error")
    if error_msg:
        ui.label("Error Details").classes("section-title")
        ui.code(_format_error_message(error_msg), language="text").classes("w-full").style("white-space: pre-wrap;")

    with ui.element("div").classes("w-full metric-grid"):
        _metric_card_small("Exec Time (s)", meta.get("execution_time"))
        _metric_card_small("Agent Cost", meta.get("agent_cost"))
        _metric_card_small("Benchmark Cost", meta.get("benchmark_cost"))


def _render_session_trajectory(turns_list: list[dict]) -> None:
    if not turns_list:
        ui.label("No trajectory data available yet.")
        return
    with ui.timeline(side="right"):
        for item in turns_list:
            kind = str(item.get("type", "event"))
            step_no = item.get("step", "?")
            content = item.get("content")
            title = kind.title()
            subtitle = f"Step {step_no}" if step_no != "?" else None
            icon = None
            if kind == "action":
                icon = "bolt"
            elif kind == "observation":
                icon = "visibility"
            elif kind == "error":
                icon = "error"
            with ui.timeline_entry(title=title, subtitle=subtitle, icon=icon):
                _render_trajectory_entry(kind, content)


def _render_session_logs(
    run_id: Optional[str],
    agent_files: list[Path],
    benchmark_files: list[Path],
) -> None:
    if not run_id:
        ui.label("No logs found.")
        return

    def _render_agent_logs() -> None:
        _render_log_files(agent_files)

    def _render_benchmark_logs() -> None:
        _render_log_files(benchmark_files)

    _render_tabs(
        ["Agent", "Benchmark"],
        {"Agent": _render_agent_logs, "Benchmark": _render_benchmark_logs},
    )


def _render_session_config(
    run_config_content: Optional[dict],
    benchmark_config_content: Optional[dict],
) -> None:
    if run_config_content is None and benchmark_config_content is None:
        ui.label("No config found.")
        return
    if run_config_content is not None:
        with ui.expansion("Run Config", value=False):
            ui.code(_format_payload(run_config_content), language="json").classes("w-full")
    if benchmark_config_content is not None:
        with ui.expansion("Benchmark Config", value=False):
            ui.code(_format_payload(benchmark_config_content), language="json").classes("w-full")


def _render_session_benchmark_results(results_content: Optional[dict]) -> None:
    if results_content is None:
        ui.label("No results found.")
    else:
        ui.code(_format_payload(results_content), language="json").classes("w-full")


def _render_session_tabs(
    *,
    meta: dict,
    session_data: Optional[dict],
    turns_list: list[dict],
    run_id: Optional[str],
    agent_files: list[Path],
    benchmark_files: list[Path],
    run_config_content: Optional[dict],
    benchmark_config_content: Optional[dict],
    results_content: Optional[dict],
) -> None:
    def _render_results_tab() -> None:
        _render_session_results_overview(meta)

    def _render_task_tab() -> None:
        _render_task_details(session_data)

    def _render_trajectory_tab() -> None:
        _render_session_trajectory(turns_list)

    def _render_logs_tab() -> None:
        _render_session_logs(run_id, agent_files, benchmark_files)

    def _render_config_tab() -> None:
        _render_session_config(run_config_content, benchmark_config_content)

    def _render_benchmark_results_tab() -> None:
        _render_session_benchmark_results(results_content)

    _render_tabs(
        ["Results", "Task", "Trajectory", "Logs", "Config", "Benchmark Results"],
        {
            "Results": _render_results_tab,
            "Task": _render_task_tab,
            "Trajectory": _render_trajectory_tab,
            "Logs": _render_logs_tab,
            "Config": _render_config_tab,
            "Benchmark Results": _render_benchmark_results_tab,
        },
        default="Results",
    )


def _render_dive_panel(
    sessions: dict,
    turns: dict,
    *,
    run_id: Optional[str],
    selected_session: Optional[str],
    on_session_change,
    output_dir: Optional[str] = None,
) -> None:
    session_ids = sorted(sessions.keys())
    if not session_ids:
        ui.label("No sessions to inspect yet.")
        return

    active_session = selected_session if selected_session in session_ids else session_ids[0]

    with ui.card().classes("w-full card p-4"):
        ui.select(session_ids, value=active_session, label="Select session").props("dense").on_value_change(
            on_session_change
        )
    meta = sessions.get(active_session, {})
    turns_list = turns.get(active_session, [])

    agent_files: list[Path] = []
    benchmark_files: list[Path] = []
    session_data = None
    run_config_content = None
    benchmark_config_content = None
    results_content = None
    if run_id:
        resolved_output = output_dir or get_settings().output_dir
        sess_paths = RunPaths(run_id=run_id, output_dir=resolved_output).session(active_session)
        base = sess_paths.benchmark_dir
        agent_files = _list_text_files(sess_paths.agent_dir)
        benchmark_files = _list_text_files(base)
        session_data = load_session_file(str(sess_paths.session_manifest), "json")
        run_config_content = load_run_config(run_id, output_dir=resolved_output)
        benchmark_config_content = load_session_file(str(base / "config.json"), "json")
        results_content = load_session_file(str(sess_paths.benchmark_results), "json")

    with ui.card().classes("w-full card p-4"):
        _render_session_tabs(
            meta=meta,
            session_data=session_data,
            turns_list=turns_list,
            run_id=run_id,
            agent_files=agent_files,
            benchmark_files=benchmark_files,
            run_config_content=run_config_content,
            benchmark_config_content=benchmark_config_content,
            results_content=results_content,
        )


def _render_log_files(files: list[Path]) -> None:
    if not files:
        ui.label("No logs found.")
        return
    for path in files:
        with ui.expansion(path.name, value=False):
            content = _load_text_file(path)
            if content is None:
                ui.label("Unable to read file.")
            else:
                language = "json" if path.suffix.lower() == ".json" else "text"
                ui.code(content, language=language).classes("w-full")


def _render_task_details(session_data: Optional[dict]) -> None:
    if not session_data:
        ui.label("No session data found.")
        return
    ordered_keys = ["task", "context", "actions"]
    seen = set()
    for key in ordered_keys:
        if key in session_data:
            seen.add(key)
            with ui.expansion(str(key), value=False):
                value = session_data.get(key)
                if isinstance(value, str):
                    ui.code(value, language="text").classes("w-full")
                else:
                    ui.code(_format_payload(value), language="json").classes("w-full")
    for key in sorted(session_data.keys()):
        if key in seen:
            continue
        with ui.expansion(str(key), value=False):
            value = session_data.get(key)
            if isinstance(value, str):
                ui.code(value, language="text").classes("w-full")
            else:
                ui.code(_format_payload(value), language="json").classes("w-full")
