# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from ....utils.paths import RunPaths
from ....utils.settings import get_settings
from ...lib.api import list_agents, list_benchmarks
from .state import RunContext
from .status import _status_from_outcome


def get_display_mappings() -> tuple[dict[str, str], dict[str, str]]:
    bench_label_to_key: dict[str, str] = {}
    for item in list_benchmarks():
        bench_label_to_key[str(item["display_name"])] = item["slug_name"]

    agent_label_to_key: dict[str, str] = {}
    for item in list_agents():
        agent_label_to_key[str(item["display_name"])] = item["slug_name"]

    return bench_label_to_key, agent_label_to_key


def _build_overview_secondary_metrics(
    sessions: dict,
    results: Optional[dict],
) -> list[tuple[str, Any]]:
    total_sessions = None
    successful_sessions = None
    percent_successful = None
    percent_finished = None
    avg_score = None
    avg_steps = None
    avg_agent_cost = None
    avg_benchmark_cost = None
    total_run_cost = None
    if isinstance(results, dict):
        total_sessions = results.get("total_sessions")
        successful_sessions = results.get("successful_sessions")
        percent_successful = results.get("percent_successful")
        percent_finished = results.get("percent_finished")
        avg_score = results.get("average_score")
        avg_steps = results.get("average_steps")
        avg_agent_cost = results.get("average_agent_cost")
        avg_benchmark_cost = results.get("average_benchmark_cost")
        total_run_cost = results.get("total_run_cost")

    if total_sessions is None:
        total_sessions = len(sessions)
    if successful_sessions is None:
        successful_sessions = sum(1 for s in sessions.values() if s.get("status") == "success")

    if avg_steps is None:
        steps = [s.get("steps") for s in sessions.values() if s.get("steps") is not None]
        if steps:
            avg_steps = sum(float(x) for x in steps) / len(steps)
    if avg_score is None:
        scores = [s.get("score") for s in sessions.values() if s.get("score") is not None]
        if scores:
            avg_score = sum(float(x) for x in scores) / len(scores)
    if avg_agent_cost is None:
        costs = [s.get("agent_cost") for s in sessions.values() if s.get("agent_cost") is not None]
        if costs:
            avg_agent_cost = sum(float(x) for x in costs) / len(costs)
    if avg_benchmark_cost is None:
        costs = [s.get("benchmark_cost") for s in sessions.values() if s.get("benchmark_cost") is not None]
        if costs:
            avg_benchmark_cost = sum(float(x) for x in costs) / len(costs)
    if total_run_cost is None:
        total_run_cost = None
        had_cost = False
        for s in sessions.values():
            agent_cost = s.get("agent_cost")
            benchmark_cost = s.get("benchmark_cost")
            if agent_cost is not None:
                total_run_cost = (total_run_cost or 0.0) + float(agent_cost)
                had_cost = True
            if benchmark_cost is not None:
                total_run_cost = (total_run_cost or 0.0) + float(benchmark_cost)
                had_cost = True
        if not had_cost:
            total_run_cost = None

    completed_sessions = [s for s in sessions.values() if s.get("status") != "running"]
    completed_total = len(completed_sessions)

    success_rate = percent_successful
    if success_rate is None and completed_total > 0:
        success_rate = sum(1 for s in completed_sessions if s.get("status") == "success") / completed_total

    finished_rate = percent_finished
    if finished_rate is None and completed_total > 0:
        finished_rate = (
            sum(1 for s in completed_sessions if s.get("status") in ("success", "unsuccessful")) / completed_total
        )

    return [
        ("Avg Score", avg_score),
        ("Success Rate", success_rate),
        ("Finished Rate", finished_rate),
        ("Avg Steps", avg_steps),
        ("Avg Agent Cost", avg_agent_cost),
        ("Avg Benchmark Cost", avg_benchmark_cost),
        ("Total Run Cost", total_run_cost),
    ]


def load_leaderboard_data(output_dir: str) -> list[dict]:
    rows = []
    try:
        for run_id in os.listdir(output_dir):
            results_path = RunPaths(run_id=run_id, output_dir=output_dir).results
            if not os.path.isfile(results_path):
                continue
            try:
                with open(results_path, encoding="utf-8-sig") as f:
                    r = json.load(f)
                rows.append(
                    {
                        "Agent": r.get("agent_name", "unknown"),
                        "Model": r.get("model_name", "unknown"),
                        "Benchmark": r.get("benchmark_name", "unknown"),
                        "Subset": r.get("subset_name", "unknown"),
                        "Num Tasks": r.get("total_sessions", 0),
                        "Final Score": r.get("average_score"),
                        "Total Run Cost": r.get("total_run_cost"),
                        "Avg Agent Cost": r.get("average_agent_cost"),
                        "run_id": run_id,
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
    except FileNotFoundError:
        pass
    return rows


def load_run_results(results_path: str) -> Optional[dict]:
    try:
        with open(results_path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_run_config(run_id: str, *, output_dir: Optional[str] = None) -> Optional[dict]:
    resolved_output = output_dir or get_settings().output_dir
    run_paths = RunPaths(run_id=run_id, output_dir=resolved_output)
    config_path = run_paths.config
    config_data = load_session_file(str(config_path), "json")
    if config_data is not None:
        return config_data
    legacy_path = run_paths.root / "config.json"
    if legacy_path == config_path:
        return None
    return load_session_file(str(legacy_path), "json")


def _short_model_name(name: Any) -> str:
    text = str(name)
    if "/" in text:
        return text.split("/")[-1]
    return text


def _resolve_run_meta(
    results: Optional[dict],
    config_data: Optional[dict],
    *,
    fallback_benchmark: Optional[str] = None,
    fallback_agent: Optional[str] = None,
) -> dict[str, Any]:
    benchmark = None
    agent = None
    models: list[str] = []

    if isinstance(results, dict):
        benchmark = results.get("benchmark_name") or benchmark
        agent = results.get("agent_name") or agent
        model_names = results.get("model_names") or []
        if isinstance(model_names, list):
            models = [_short_model_name(m) for m in model_names if m]
        if not models and results.get("model_name"):
            models = [_short_model_name(results.get("model_name"))]

    if isinstance(config_data, dict):
        bench_cfg = config_data.get("benchmark") or {}
        agent_cfg = config_data.get("agent") or {}
        if isinstance(bench_cfg, str):
            if not benchmark:
                benchmark = bench_cfg
            bench_cfg = {}
        if isinstance(agent_cfg, str):
            if not agent:
                agent = agent_cfg
            agent_cfg = {}
        if not benchmark:
            benchmark = bench_cfg.get("display_name") or bench_cfg.get("slug_name") or bench_cfg.get("class")
        if not agent:
            agent = agent_cfg.get("display_name") or agent_cfg.get("slug_name") or agent_cfg.get("class")
        if not models:
            model_names = agent_cfg.get("model_names")
            if isinstance(model_names, list):
                models = [_short_model_name(m) for m in model_names if m]
        if not models and agent_cfg.get("model_name"):
            models = [_short_model_name(agent_cfg.get("model_name"))]

    if not benchmark:
        benchmark = fallback_benchmark
    if not agent:
        agent = fallback_agent

    if models:
        models = list(dict.fromkeys(models))

    return {
        "benchmark": benchmark or "-",
        "agent": agent or "-",
        "models": models,
    }


def _resolve_planned_sessions(
    results: Optional[dict],
    config_data: Optional[dict],
    fallback: Optional[int],
) -> Optional[int]:
    if isinstance(results, dict) and results.get("planned_sessions") is not None:
        return results.get("planned_sessions")
    if isinstance(config_data, dict):
        if config_data.get("planned_sessions") is not None:
            return config_data.get("planned_sessions")
        if config_data.get("num_tasks") is not None:
            return config_data.get("num_tasks")

        bench_cfg = config_data.get("benchmark")
        if isinstance(bench_cfg, dict) and bench_cfg.get("planned_sessions") is not None:
            return bench_cfg.get("planned_sessions")

        run_cfg = config_data.get("run")
        if isinstance(run_cfg, dict):
            if run_cfg.get("planned_sessions") is not None:
                return run_cfg.get("planned_sessions")
            if run_cfg.get("num_tasks") is not None:
                return run_cfg.get("num_tasks")
    return fallback


def _resolve_total_workers(
    results: Optional[dict],
    config_data: Optional[dict],
    fallback: Optional[int],
) -> Optional[int]:
    if isinstance(results, dict) and results.get("max_workers") is not None:
        return results.get("max_workers")
    if isinstance(config_data, dict):
        if config_data.get("max_workers") is not None:
            return config_data.get("max_workers")

        run_cfg = config_data.get("run")
        if isinstance(run_cfg, dict) and run_cfg.get("max_workers") is not None:
            return run_cfg.get("max_workers")
    return fallback


def _load_run_context(
    run_id: Optional[str],
    *,
    fallback_benchmark: Optional[str] = None,
    fallback_agent: Optional[str] = None,
    planned_fallback: Optional[int] = None,
    workers_fallback: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> RunContext:
    results = None
    config_data = None
    if run_id:
        resolved_output = output_dir or get_settings().output_dir
        results = load_run_results(str(RunPaths(run_id=run_id, output_dir=resolved_output).results))
        config_data = load_run_config(run_id, output_dir=resolved_output)
    run_meta = _resolve_run_meta(
        results,
        config_data,
        fallback_benchmark=fallback_benchmark,
        fallback_agent=fallback_agent,
    )
    planned_sessions = _resolve_planned_sessions(results, config_data, planned_fallback)
    total_workers = _resolve_total_workers(results, config_data, workers_fallback)
    return RunContext(
        results=results,
        config=config_data,
        run_meta=run_meta,
        planned_sessions=planned_sessions,
        total_workers=total_workers,
    )


def _resolve_tab_label(
    value: Any,
    tab_by_name: dict[str, Any],
    default: str,
) -> str:
    if isinstance(value, str) and value in tab_by_name:
        return value
    for name, tab in tab_by_name.items():
        if value is tab:
            return name
    return default


def _open_session_from_row(
    state,
    scope: str,
    row: dict,
    refresh,
) -> None:
    session_id = row.get("session")
    if not session_id:
        return
    if scope == "history":
        state.selected_history_session = session_id
    else:
        state.selected_session = session_id
    state.active_tabs[scope] = "Sessions"
    tabs = state.tabs_controls.get(scope)
    tab_by_name = state.tabs_by_scope.get(scope) or {}
    target = tab_by_name.get("Sessions")
    if tabs is not None and target is not None:
        tabs.value = target
    refresh()


def _build_session_rows(sessions: dict) -> list[dict]:
    return [
        {
            "session": sid,
            "status": data.get("status", ""),
            "steps": data.get("steps", 0),
            "score": data.get("score"),
        }
        for sid, data in sessions.items()
    ]


def load_session_file(file_path: str, file_type: str) -> Optional[Any]:
    if not os.path.isfile(file_path):
        return None
    try:
        if file_type == "log":
            with open(file_path, encoding="utf-8-sig", errors="replace", newline="") as f:
                lines = f.readlines()[-200:]
            return "".join(lines) if lines else "(empty)"
        with open(file_path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _list_text_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files = []
    for name in sorted(os.listdir(root)):
        path = root / name
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".log", ".json"}:
            files.append(path)
    return files


def _load_text_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = load_session_file(str(path), "json")
        if data is None:
            return None
        return json.dumps(data, ensure_ascii=False, indent=2)
    if suffix in {".txt", ".log"}:
        return load_session_file(str(path), "log")
    return None


def _load_trajectory_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events


def _build_history_sessions(
    run_id: str,
    results: Optional[dict],
    *,
    output_dir: Optional[str] = None,
) -> dict[str, dict]:
    sessions: dict[str, dict] = {}
    session_results = results.get("session_results") if isinstance(results, dict) else None
    if isinstance(session_results, list):
        for item in session_results:
            if not isinstance(item, dict):
                continue
            session_id = item.get("session_id")
            if not session_id:
                continue
            success = item.get("success", False)
            is_finished = item.get("is_finished")
            details = item.get("details") or {}
            session_metadata = details.get("session_metadata") or {}
            error_source = session_metadata.get("error_source")
            error_message = session_metadata.get("error")
            sessions[session_id] = {
                "status": _status_from_outcome(success, is_finished, error_source),
                "success": bool(success),
                "steps": item.get("steps", 0),
                "score": item.get("score"),
                "execution_time": item.get("execution_time"),
                "agent_cost": item.get("agent_cost"),
                "benchmark_cost": item.get("benchmark_cost"),
                "is_finished": is_finished,
                "error_source": error_source,
                "error": error_message,
            }
    if sessions:
        return sessions

    resolved_output = output_dir or get_settings().output_dir
    run_paths = RunPaths(run_id=run_id, output_dir=resolved_output)
    sessions_root = run_paths.sessions_root
    if not os.path.isdir(sessions_root):
        return sessions
    for name in sorted(os.listdir(sessions_root)):
        sess_paths = run_paths.session(name)
        sess_results = sess_paths.results
        sess_summary = sess_paths.summary
        target_path = sess_results if sess_results.is_file() else sess_summary
        if not target_path.is_file():
            continue
        data = load_session_file(str(target_path), "json")
        if not isinstance(data, dict):
            continue
        success = data.get("success", False)
        is_finished = data.get("is_finished")
        details = data.get("details") or {}
        session_metadata = details.get("session_metadata") or {}
        error_source = session_metadata.get("error_source")
        error_message = session_metadata.get("error")
        sessions[name] = {
            "status": _status_from_outcome(success, is_finished, error_source),
            "success": bool(success),
            "steps": data.get("steps", 0),
            "score": data.get("score"),
            "execution_time": data.get("execution_time"),
            "agent_cost": data.get("agent_cost"),
            "benchmark_cost": data.get("benchmark_cost"),
            "is_finished": is_finished,
            "error_source": error_source,
            "error": error_message,
        }
    return sessions


def _load_history_turns(
    run_id: str,
    session_id: str,
    *,
    output_dir: Optional[str] = None,
) -> list[dict]:
    resolved_output = output_dir or get_settings().output_dir
    sess_paths = RunPaths(run_id=run_id, output_dir=resolved_output).session(session_id)
    events = _load_trajectory_events(sess_paths.trajectory)
    turns: list[dict] = []
    for event in events:
        kind = event.get("event") or event.get("type") or "event"
        if kind == "action":
            payload = event.get("action")
        elif kind == "observation":
            payload = event.get("observation")
        elif kind == "error":
            payload = event.get("error") or event
        else:
            payload = event
        turns.append({"type": kind, "step": event.get("step"), "content": payload})
    return turns[-200:]
