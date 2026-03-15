# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from ....observers.handlers.dashboard_events import DashboardEventsObserver


@dataclass
class RunState:
    events: list[dict] = field(default_factory=list)
    sessions: dict[str, dict] = field(default_factory=dict)
    turns: dict[str, list] = field(default_factory=dict)
    bench_controls: dict[str, Any] = field(default_factory=dict)
    agent_controls: dict[str, Any] = field(default_factory=dict)
    tracker: DashboardEventsObserver | None = None
    thread: threading.Thread | None = None
    run_id: str | None = None
    run_context_manager: Any = None
    run_active: bool = False
    refresh_needed: bool = True
    last_run_active: bool = False
    bench_key: str | None = None
    agent_key: str | None = None
    num_tasks: int | None = None
    max_workers: int = 3
    selected_session: str | None = None
    selected_history_run: str | None = None
    selected_history_session: str | None = None
    history_root: str = ""
    selected_agents: list[str] = field(default_factory=list)
    selected_models: list[str] = field(default_factory=list)
    selected_benchmarks: list[str] = field(default_factory=list)
    selected_subsets: list[str] = field(default_factory=list)
    min_tasks: int = 0
    planned_sessions: int | None = None
    active_tabs: dict[str, str] = field(default_factory=dict)
    tabs_controls: dict[str, Any] = field(default_factory=dict)
    tabs_by_scope: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunContext:
    results: Optional[dict]
    config: Optional[dict]
    run_meta: dict[str, Any]
    planned_sessions: Optional[int]
    total_workers: Optional[int]


@dataclass
class RunViews:
    start_button: Any
    bench_select: Any
    agent_select: Any
    num_tasks_input: Any
    max_workers_input: Any
    bench_form: Any
    agent_form: Any
    overview_panel: Any
    sessions_panel: Any
    run_log_panel: Any
    leaderboard_panel: Any
    history_panel: Any
    run_panel_box: Any
    overview_tab: Any
    sessions_tab: Any
    log_tab: Any
    overview_panel_el: Any
    sessions_panel_el: Any
    log_panel_el: Any


SESSION_COLUMNS = [
    {"name": "session", "label": "Session", "field": "session"},
    {"name": "status", "label": "Status", "field": "status"},
    {"name": "steps", "label": "Steps", "field": "steps"},
    {"name": "score", "label": "Score", "field": "score"},
]

LEADERBOARD_COLUMNS = [
    {"name": "agent", "label": "Agent", "field": "Agent"},
    {"name": "model", "label": "Model", "field": "Model"},
    {"name": "benchmark", "label": "Benchmark", "field": "Benchmark"},
    {"name": "subset", "label": "Subset", "field": "Subset"},
    {"name": "tasks", "label": "Num Tasks", "field": "Num Tasks"},
    {"name": "score", "label": "Final Score", "field": "Final Score"},
    {"name": "run_cost", "label": "Total Run Cost", "field": "Total Run Cost"},
    {"name": "avg_agent_cost", "label": "Avg Agent Cost", "field": "Avg Agent Cost"},
]

TASK_RESULT_COLUMNS = [
    {"name": "session_id", "label": "Session", "field": "session_id"},
    {"name": "task_id", "label": "Task Id", "field": "task_id"},
    {"name": "success", "label": "Success", "field": "success"},
    {"name": "is_finished", "label": "Finished", "field": "is_finished"},
    {"name": "score", "label": "Score", "field": "score"},
    {"name": "steps", "label": "Steps", "field": "steps"},
    {"name": "agent_cost", "label": "Agent Cost", "field": "agent_cost"},
    {"name": "benchmark_cost", "label": "Benchmark Cost", "field": "benchmark_cost"},
    {"name": "execution_time", "label": "Exec Time", "field": "execution_time"},
]

ACTION_COLUMNS = [
    {"name": "name", "label": "Action", "field": "name"},
    {"name": "description", "label": "Description", "field": "description"},
    {"name": "is_message", "label": "Message", "field": "is_message"},
    {"name": "is_finish", "label": "Finish", "field": "is_finish"},
]
