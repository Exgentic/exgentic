# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import rich_click as click

from ....core.types import RunConfig, SessionConfig
from ....core.types.session import SessionExecutionStatus, SessionOutcomeStatus
from ....core.types.session_inventory import SessionRecord, scan_sessions
from ....utils.paths import get_run_paths, get_session_paths
from ...lib.api import aggregate, execute, status
from ..options import (
    _format_exception_for_cli,
    _should_show_traceback,
    apply_debug_mode,
)
from ..render import render_batch_status


def _load_config_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_run_like_config(path: str) -> RunConfig | SessionConfig:
    """Load a RunConfig or SessionConfig from a file with full validation."""
    payload = _load_config_file(path)
    try:
        return RunConfig.model_validate(payload)
    except Exception:
        return SessionConfig.model_validate(payload)


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


def _state_from_counts(
    *,
    total: int,
    completed: int,
    running: int,
    incomplete: int,
    missing: int,
) -> str:
    if total > 0 and completed == total and running == 0 and incomplete == 0:
        return "complete"
    if total > 0 and completed == 0 and running == 0 and incomplete == 0 and missing == total:
        return "not_started"
    return "in_progress"


def _short_config_path(path: str) -> str:
    try:
        rel = os.path.relpath(path)
    except Exception:
        rel = path
    if rel.startswith(".."):
        return Path(path).name
    return rel


def _format_ago(sec: float) -> str:
    if sec < 0:
        sec = 0
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec / 60)}m"
    if sec < 86400:
        return f"{int(sec / 3600)}h"
    return f"{int(sec / 86400)}d"


def _truncate_leading(text: str, max_len: int) -> str:
    if max_len <= 3 or len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3) :]


def _format_batch_error(exc: Exception) -> str:
    if _should_show_traceback():
        details = _format_exception_for_cli(exc)
    else:
        details = str(exc)
    if not details:
        details = repr(exc)
    return details.replace("\n", "\n  ")


def _format_config_link(path: str, *, max_len: int = 30) -> str:
    display = _short_config_path(path)
    display = _truncate_leading(display, max_len)
    try:
        target = Path(path).resolve()
    except Exception:
        return display
    return f"[link=file://{target}]{display}[/link]"


def _format_float(value: Any) -> str:
    try:
        num = float(value)
    except Exception:
        return "-"
    if abs(num) >= 1000:
        return f"{num:,.0f}"
    if abs(num) >= 10:
        return f"{num:.2f}"
    return f"{num:.4g}"


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _write_session_config(session_config: SessionConfig, *, overwrite: bool) -> bool:
    session_id = session_config.get_session_id()
    sess_paths = get_session_paths(session_id)
    config_path = sess_paths.session_config
    if config_path.exists() and not overwrite:
        return False
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            session_config.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2,
        )
    return True


def _format_models(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        names = [str(item) for item in value if item is not None]
        return ", ".join(names) if names else "-"
    return str(value)


def _compute_session_id(config: dict[str, Any]) -> str:
    session_cfg = SessionConfig.model_validate(config)
    return session_cfg.get_session_id()


def _results_path_from_config_location(config_path: str, config_obj: RunConfig | SessionConfig) -> Path:
    run_id = getattr(config_obj, "run_id", None)
    if not run_id:
        raise click.ClickException(f"Missing run_id in config: {config_path}")
    output_dir = Path(config_path).parent
    return (output_dir / run_id / "results.json").resolve()


def _parse_patch_values(pairs: tuple[str, ...]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.ClickException(f"Invalid --set value (expected key=value): {pair}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException(f"Invalid --set key: {pair}")
        try:
            value = json.loads(raw)
        except Exception:
            value = raw
        updates[key] = value
    return updates


def _apply_patch(payload: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if key.startswith("agent."):
            key = f"agent_kwargs.{key[len('agent.') :]}"
        elif key.startswith("benchmark."):
            key = f"benchmark_kwargs.{key[len('benchmark.') :]}"
        if "." not in key:
            payload[key] = value
            continue
        cursor: dict[str, Any] = payload
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value


def _iter_session_files(sessions_root: Path, filename: str) -> Iterator[Path]:
    """Yield ``sessions_root/{id}/{filename}``; one-level glob (NFS-friendly)."""
    return sessions_root.glob(f"*/{filename}")


def _update_session_ids_in_dir(session_dir: Path, new_id: str) -> None:
    # Top-level only — only config.json/results.json carry session_id.
    for json_path in session_dir.glob("*.json"):
        try:
            payload = _load_config_file(str(json_path))
        except Exception:
            continue
        if not isinstance(payload, dict) or "session_id" not in payload:
            continue
        if payload.get("session_id") == new_id:
            continue
        payload["session_id"] = new_id
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")


def _extract_task_id_from_session(session_dir: Path) -> str | None:
    results_path = session_dir / "results.json"
    if not results_path.exists():
        return None
    try:
        payload = _load_config_file(str(results_path))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    task_id = payload.get("task_id") or payload.get("task_key")
    return str(task_id) if task_id is not None else None


def _recover_session_hashes(roots: list[Path], *, do_apply: bool) -> int:
    changes = 0
    for root in roots:
        sessions_root = root / "sessions"
        if not sessions_root.exists():
            raise click.ClickException(f"sessions dir not found: {sessions_root}")
        for cfg_path in _iter_session_files(sessions_root, "config.json"):
            session_dir = cfg_path.parent
            old_id = session_dir.name
            config = _load_config_file(str(cfg_path))
            new_id = _compute_session_id(config)
            if new_id == old_id:
                continue
            changes += 1
            click.echo(f"{old_id} -> {new_id} ({session_dir})")
            if not do_apply:
                continue
            if "session_id" in config:
                config["session_id"] = new_id
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                    f.write("\n")
            _update_session_ids_in_dir(session_dir, new_id)
            session_dir.rename(session_dir.with_name(new_id))
    return changes


def _load_results_summary(
    path: str,
    *,
    session_records: list[SessionRecord] | None = None,
) -> tuple[str, str, str, int | None, int | None, int | None, int | None]:
    if not path:
        return "-", "-", "-", None, None, None, None
    payload: dict = {}
    if Path(path).is_file():
        try:
            payload = _load_config_file(path) or {}
        except Exception:
            payload = {}
    score = payload.get("benchmark_score")
    if score is None:
        score = payload.get("average_score")
    cost = payload.get("total_run_cost")
    if cost is None:
        cost = payload.get("total_agent_cost")
    # Run-level results.json is only written at end of run, so during a live
    # run its Score/Cost are stale. Prefer live session-level aggregates (#190).
    if session_records is None:
        session_records = scan_sessions(Path(path).parent / "sessions")
    live_cost = 0.0
    live_scores: list[float] = []
    found = False
    for r in session_records:
        if r.results is None:
            continue
        found = True
        live_cost += float(r.results.get("agent_cost") or 0) + float(r.results.get("benchmark_cost") or 0)
        s = r.results.get("score")
        if s is not None:
            live_scores.append(float(s))
    if found:
        cost = live_cost
        if live_scores:
            score = sum(live_scores) / len(live_scores)
    models = payload.get("model_names")
    if models is None:
        models = payload.get("model_name")
    ready = None
    finished = None
    errors = None
    aggregated = None
    session_results = payload.get("session_results")
    aggregated_ids = payload.get("aggregated_session_ids")
    if isinstance(aggregated_ids, list):
        aggregated = len(aggregated_ids)
    elif isinstance(payload.get("completed_sessions"), int):
        aggregated = payload.get("completed_sessions")
    if isinstance(session_results, list):
        ready = 0
        finished = 0
        errors = 0
        for item in session_results:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").lower()
            is_finished = item.get("is_finished")
            if status in ("error", "cancelled"):
                errors += 1
            else:
                ready += 1
            if is_finished is True:
                finished += 1
    return (
        _format_float(score),
        _format_float(cost),
        _format_models(models),
        ready,
        finished,
        errors,
        aggregated,
    )


def _expand_config_inputs(
    config_values: tuple[str, ...],
    extra_args: list[str],
) -> list[str]:
    if not config_values:
        raise click.ClickException("At least one --config is required.")

    raw_values = [*config_values, *extra_args]
    expanded: list[str] = []

    for value in raw_values:
        value = str(value).strip()
        if not value:
            continue
        if glob.has_magic(value):
            matches = [str(Path(p)) for p in sorted(glob.glob(value, recursive=True)) if Path(p).is_file()]
            if not matches:
                raise click.ClickException(f"No config files matched pattern: {value}")
            expanded.extend(matches)
        else:
            path = Path(value)
            if not path.is_file():
                raise click.ClickException(f"Config file not found: {value}")
            expanded.append(str(path))

    deduped: list[str] = []
    seen: set[str] = set()
    for path in expanded:
        normalized = str(Path(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if not deduped:
        raise click.ClickException("No config files resolved from --config values.")
    return deduped


@click.group("batch")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
def batch_cmd(debug: bool) -> None:
    """Batch operations over multiple config files."""
    apply_debug_mode(debug)


@batch_cmd.command(
    "status",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.option("--num-tasks", type=int, default=None, help="Override num_tasks for display.")
@click.pass_context
def batch_status_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
    num_tasks: int | None,
) -> None:
    """Show a status table for multiple config files."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    rows: list[dict[str, str]] = []

    for i, config_path in enumerate(config_paths, start=1):
        row: dict[str, str] = {
            "#": str(i),
            "benchmark": "-",
            "agent": "-",
            "model": "-",
            "sessions": "-",
            "score": "-",
            "cost": "-",
            "last_event": "-",
        }
        try:
            cfg = _load_run_like_config(config_path)
            if num_tasks is not None and isinstance(cfg, RunConfig):
                cfg = cfg.with_overrides(num_tasks=num_tasks)
            run_status = status(cfg)
            completed = 0
            running = 0
            errors = 0
            for s in run_status.session_statuses:
                if s.status == SessionExecutionStatus.RUNNING:
                    running += 1
                elif s.status == SessionExecutionStatus.COMPLETED:
                    completed += 1
                    if s.result_status in (SessionOutcomeStatus.ERROR, SessionOutcomeStatus.CANCELLED):
                        errors += 1
            total = run_status.total_tasks
            parts = [f"{completed}/{total}"]
            if running:
                parts.append(f"{running}run")
            if errors:
                parts.append(f"{errors}err")
            sessions_str = " ".join(parts)

            sessions_dir = Path(run_status.results_path).parent / "sessions"
            session_records = scan_sessions(sessions_dir)
            score, cost, models, *_ = _load_results_summary(run_status.results_path, session_records=session_records)
            model = run_status.model_name or models
            if model != "-":
                # Strip common prefixes for readability.
                for prefix in ("openai/azure/", "openai/Azure/", "openai/"):
                    if model.startswith(prefix):
                        model = model[len(prefix) :]
                        break
            benchmark = run_status.benchmark_slug_name
            subset = run_status.subset_name
            if subset:
                benchmark = f"{benchmark}/{subset}"
            mtimes = [r.mtime for r in session_records if r.mtime is not None]
            last_event = _format_ago(time.time() - max(mtimes)) if mtimes else "-"
            row.update(
                {
                    "benchmark": benchmark,
                    "agent": run_status.agent_slug_name,
                    "model": model,
                    "sessions": sessions_str,
                    "score": score,
                    "cost": cost,
                    "last_event": last_event,
                }
            )
        except Exception:
            row["benchmark"] = _short_config_path(config_path)
        rows.append(row)

    render_batch_status(rows)


@batch_cmd.command(
    "evaluate",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.option("--num-tasks", type=int, default=None, help="Override num_tasks for all configs.")
@click.option("--max-workers", type=int, default=None, help="Concurrent sessions across all configs.")
@click.option("--max-steps", type=int, default=None, help="Override max_steps for all configs.")
@click.option("--max-actions", type=int, default=None, help="Override max_actions for all configs.")
@click.pass_context
def batch_evaluate_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
    num_tasks: int | None,
    max_workers: int | None,
    max_steps: int | None,
    max_actions: int | None,
) -> None:
    """Evaluate multiple configs with a shared worker pool.

    All sessions from all configs run in a single pool of --max-workers.
    Completed sessions are skipped.  Each config is aggregated from disk
    after all sessions finish.
    """
    from ....core.orchestrator.run import core_batch_evaluate

    apply_debug_mode(debug)
    config_paths = list(_expand_config_inputs(config_values, list(ctx.args)))
    overrides = ctx.params

    configs: list[RunConfig] = []
    for config_path in config_paths:
        cfg = _load_run_like_config(config_path)
        if isinstance(cfg, RunConfig):
            cfg = cfg.with_overrides(**overrides)
        configs.append(cfg)

    core_batch_evaluate(configs, max_workers=max_workers or 1)


@batch_cmd.command(
    "execute",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.option("--num-tasks", type=int, default=None, help="Override num_tasks for all configs.")
@click.option("--max-workers", type=int, default=None, help="Override max_workers for all configs.")
@click.option("--max-steps", type=int, default=None, help="Override max_steps for all configs.")
@click.option("--max-actions", type=int, default=None, help="Override max_actions for all configs.")
@click.pass_context
def batch_execute_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
    num_tasks: int | None,
    max_workers: int | None,
    max_steps: int | None,
    max_actions: int | None,
) -> None:
    """Execute configs sequentially (no aggregation)."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    failures: list[tuple[str, str]] = []

    for config_path in config_paths:
        click.echo(f"Running: {config_path}")
        try:
            cfg = _load_run_like_config(config_path)
            if isinstance(cfg, RunConfig):
                cfg = cfg.with_overrides(**ctx.params)
            execute(config=cfg)
        except Exception as exc:
            failures.append((config_path, str(exc)))
            click.echo(f"Error: {config_path}\n  {_format_batch_error(exc)}")

    if failures:
        raise click.ClickException("Batch execute completed with errors in " + ", ".join(path for path, _ in failures))


@batch_cmd.command(
    "prepare",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing session config files.",
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.pass_context
def batch_prepare_cmd(
    ctx: click.Context,
    debug: bool,
    overwrite: bool,
    config_values: tuple[str, ...],
) -> None:
    """Prepare session directories and configs without executing."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    failures: list[tuple[str, str]] = []

    for config_path in config_paths:
        click.echo(f"Preparing: {config_path}")
        try:
            cfg = _load_run_like_config(config_path)
            if isinstance(cfg, RunConfig):
                with cfg.get_context():
                    session_configs = cfg.get_session_configs()
                    created = 0
                    for sc in session_configs:
                        if _write_session_config(sc, overwrite=overwrite):
                            created += 1
                    click.echo(f"Prepared {created}/{len(session_configs)} sessions.")
            else:
                with cfg.get_context():
                    created = 1 if _write_session_config(cfg, overwrite=overwrite) else 0
                click.echo(f"Prepared {created}/1 sessions.")
        except Exception as exc:
            failures.append((config_path, str(exc)))
            click.echo(f"Error: {config_path}\n  {_format_batch_error(exc)}")

    if failures:
        raise click.ClickException("Batch prepare completed with errors in " + ", ".join(path for path, _ in failures))


@batch_cmd.command(
    "aggregate",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.pass_context
def batch_aggregate_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
) -> None:
    """Aggregate configs sequentially (no execution)."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    failures: list[tuple[str, str]] = []

    for config_path in config_paths:
        click.echo(f"Running: {config_path}")
        try:
            cfg = _load_run_like_config(config_path)
            aggregate(config=cfg)
        except Exception as exc:
            failures.append((config_path, str(exc)))
            click.echo(f"Error: {config_path}\n  {_format_batch_error(exc)}")

    if failures:
        raise click.ClickException(
            "Batch aggregate completed with errors in " + ", ".join(path for path, _ in failures)
        )


@batch_cmd.command(
    "patch",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.option(
    "--set",
    "set_values",
    multiple=True,
    help="key=value pairs to update (repeatable). Supports dotted paths.",
)
@click.option(
    "--apply",
    is_flag=True,
    help="Apply changes (otherwise dry-run).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without applying.",
)
@click.pass_context
def batch_patch_cmd(
    ctx: click.Context,
    config_values: tuple[str, ...],
    set_values: tuple[str, ...],
    apply: bool,
    dry_run: bool,
) -> None:
    """Patch run/session configs and recover session hashes."""
    if apply and dry_run:
        raise click.ClickException("Use only one of --apply or --dry-run.")
    if not (config_values or ctx.args):
        raise click.ClickException("Provide at least one --config.")
    if not set_values:
        raise click.ClickException("Provide at least one --set key=value.")
    do_apply = apply and not dry_run
    updates = _parse_patch_values(set_values)

    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    run_roots: list[Path] = []
    for config_path in config_paths:
        cfg = _load_run_like_config(config_path)
        with cfg.get_context():
            run_paths = get_run_paths()
            run_roots.append(Path(run_paths.root))
        payload = _load_config_file(config_path)
        _apply_patch(payload, updates)
        click.echo(f"Patch config: {config_path}")
        if do_apply:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")

    seen_roots: set[Path] = set()
    for root in run_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        run_config_path = root / "run" / "config.json"
        run_config_payload: dict[str, Any] | None = None
        if run_config_path.exists():
            run_config_payload = _load_config_file(str(run_config_path))
            _apply_patch(run_config_payload, updates)
            click.echo(f"Patch run config: {run_config_path}")
            if do_apply:
                with open(run_config_path, "w", encoding="utf-8") as f:
                    json.dump(run_config_payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")
        run_config_obj = RunConfig.model_validate(run_config_payload) if run_config_payload is not None else None

        sessions_root = root / "sessions"
        if not sessions_root.exists():
            raise click.ClickException(f"sessions dir not found: {sessions_root}")
        changes = 0
        for cfg_path in _iter_session_files(sessions_root, "config.json"):
            session_dir = cfg_path.parent
            old_id = session_dir.name
            config = _load_config_file(str(cfg_path))
            _apply_patch(config, updates)
            new_id = _compute_session_id(config)
            if new_id != old_id:
                changes += 1
                click.echo(f"{old_id} -> {new_id} ({session_dir})")
            if do_apply:
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                    f.write("\n")
        # Handle sessions without config.json by deriving from run config + results.json task id.
        if run_config_obj is not None:
            for session_dir in sessions_root.iterdir():
                if not session_dir.is_dir():
                    continue
                cfg_path = session_dir / "config.json"
                if cfg_path.exists():
                    continue
                task_id = _extract_task_id_from_session(session_dir)
                if task_id is None:
                    continue
                old_id = session_dir.name
                new_id = run_config_obj.to_session_config(task_id).get_session_id()
                if new_id != old_id:
                    changes += 1
                    target_dir = session_dir.with_name(new_id)
                    if target_dir.exists():
                        alt_name = f"{new_id}__dup__{old_id}"
                        target_dir = session_dir.with_name(alt_name)
                        click.echo(f"{old_id} -> {new_id} (collision, renaming to {alt_name})")
                    else:
                        click.echo(f"{old_id} -> {new_id} ({session_dir})")
                    if do_apply:
                        _update_session_ids_in_dir(session_dir, new_id)
                        session_dir.rename(target_dir)
        if do_apply:
            _recover_session_hashes([root], do_apply=True)

    if not do_apply:
        click.echo(f"Dry run complete. {changes} session(s) would be renamed.")
    else:
        click.echo(f"Done. {changes} session(s) renamed.")


@batch_cmd.command(
    "extract",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.option(
    "--output",
    "output_path",
    default="batch_results.csv",
    show_default=True,
    help="CSV output path (use '-' for stdout).",
)
@click.pass_context
def batch_extract_cmd(
    ctx: click.Context,
    config_values: tuple[str, ...],
    output_path: str,
) -> None:
    """Extract run results from multiple configs into a single CSV."""
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    failures: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    preferred_keys = [
        "config_path",
        "results_path",
        "run_id",
    ]

    for config_path in config_paths:
        results_path = "-"
        run_id = "-"
        try:
            cfg = _load_run_like_config(config_path)
            run_id = getattr(cfg, "run_id", None) or "-"
            results_path = _results_path_from_config_location(config_path, cfg)
            if not results_path.is_file():
                raise click.ClickException(f"Results not found: {results_path}")
            payload = _load_config_file(str(results_path))
            if not isinstance(payload, dict):
                raise click.ClickException(f"Results JSON is not an object: {results_path}")
            if "benchmark_score" not in payload:
                raise click.ClickException(f"Missing benchmark_score in results: {results_path}")
            row: dict[str, Any] = {
                "config_path": config_path,
                "results_path": str(results_path),
                "run_id": run_id,
            }
            for key, value in payload.items():
                if key in row:
                    row[f"results_{key}"] = value
                else:
                    row[key] = value
        except Exception as exc:
            failures.append((config_path, str(exc)))
            row = {
                "config_path": config_path,
                "results_path": results_path,
                "run_id": run_id,
                "error": _format_batch_error(exc),
            }
        rows.append(row)
        all_keys.update(row.keys())

    ordered_keys = [k for k in preferred_keys if k in all_keys]
    ordered_keys.extend(sorted(k for k in all_keys if k not in ordered_keys))

    if output_path == "-":
        out_stream = sys.stdout
        writer = csv.DictWriter(out_stream, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in ordered_keys})
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(row.get(k)) for k in ordered_keys})
        click.echo(f"Wrote {len(rows)} row(s) to {out_file}.")

    if failures:
        raise click.ClickException("Batch extract completed with errors in " + ", ".join(path for path, _ in failures))


# Fields to exclude when publishing to HuggingFace (internal/bulky data).
_PUBLISH_EXCLUDE_FIELDS: set[str] = {
    "config_path",
    "results_path",
    "run_id",
    "aggregated_session_ids",
    "executed_session_ids",
    "planned_session_ids",
    "skipped_session_ids",
    "skipped_session_reasons",
    "missing_result_files",
    "session_results",
    "accumulated_agent_report",
    "accumulated_benchmark_report",
    "aggregation_mode",
    "max_workers",
    "running_sessions",
    "model_names",
    "agent_slug_name",
    "benchmark_slug_name",
    "planned_sessions",
}


def _collect_results_rows(
    config_paths: list[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Load results from config paths and return clean rows for publishing.

    Uses lightweight JSON loading to find results.json without importing
    benchmark or agent modules, so it works even when those aren't installed.
    """
    failures: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []

    for config_path in config_paths:
        try:
            # Read config as raw JSON to get run_id without importing modules
            raw_config = _load_config_file(config_path)
            run_id = raw_config.get("run_id")
            if not run_id:
                raise click.ClickException(f"Missing run_id in config: {config_path}")
            output_dir = Path(config_path).parent
            results_path = (output_dir / run_id / "results.json").resolve()
            if not results_path.is_file():
                raise click.ClickException(f"Results not found: {results_path}")
            payload = _load_config_file(str(results_path))
            if not isinstance(payload, dict):
                raise click.ClickException(f"Results JSON is not an object: {results_path}")
            if "benchmark_score" not in payload:
                raise click.ClickException(f"Missing benchmark_score in results: {results_path}")
            row: dict[str, Any] = {}
            for key, value in payload.items():
                if key not in _PUBLISH_EXCLUDE_FIELDS:
                    row[key] = value
            rows.append(row)
        except Exception as exc:
            failures.append((config_path, str(exc)))

    return rows, failures


@batch_cmd.command(
    "publish",
    context_settings={"allow_extra_args": True},
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="RunConfig/SessionConfig path or glob pattern (repeatable).",
)
@click.option(
    "--repo",
    "repo_id",
    required=True,
    help="HuggingFace dataset repo ID (e.g. 'Exgentic/open-agent-leaderboard-results').",
)
@click.option(
    "--private/--public",
    "private",
    default=True,
    show_default=True,
    help="Whether the dataset should be private.",
)
@click.option(
    "--append/--overwrite",
    "append",
    default=True,
    show_default=True,
    help="Append to existing dataset or overwrite it.",
)
@click.pass_context
def batch_publish_cmd(
    ctx: click.Context,
    config_values: tuple[str, ...],
    repo_id: str,
    private: bool,
    append: bool,
) -> None:
    """Publish run results to a HuggingFace dataset."""
    try:
        from datasets import Dataset, load_dataset
    except ImportError as err:
        raise click.ClickException(
            "The 'datasets' package is required for publishing. Install it with: pip install datasets"
        ) from err

    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    rows, failures = _collect_results_rows(config_paths)

    if not rows:
        raise click.ClickException("No valid results to publish.")

    if append:
        try:
            existing_ds = load_dataset(repo_id, split="train")
            existing_rows = list(existing_ds)
            click.echo(f"Loaded {len(existing_rows)} existing row(s) from {repo_id}.")

            # Deduplicate by (benchmark, agent, model) triple
            existing_keys = set()
            for r in existing_rows:
                key = (r.get("benchmark"), r.get("agent"), r.get("model"))
                existing_keys.add(key)

            new_rows = []
            updated = 0
            for row in rows:
                key = (row.get("benchmark"), row.get("agent"), row.get("model"))
                if key in existing_keys:
                    # Replace existing row with updated one
                    existing_rows = [
                        r for r in existing_rows if (r.get("benchmark"), r.get("agent"), r.get("model")) != key
                    ]
                    updated += 1
                new_rows.append(row)

            all_rows = existing_rows + new_rows
            click.echo(f"Publishing {len(all_rows)} row(s) ({len(new_rows) - updated} new, {updated} updated).")
        except Exception:
            click.echo("No existing dataset found, creating new one.")
            all_rows = rows
    else:
        all_rows = rows

    ds = Dataset.from_list(all_rows)
    ds.push_to_hub(repo_id, private=private)
    click.echo(f"Published {len(all_rows)} row(s) to https://huggingface.co/datasets/{repo_id}")

    if failures:
        click.echo(f"Warning: {len(failures)} config(s) had errors and were skipped:")
        for path, err in failures:
            click.echo(f"  {path}: {err}")


__all__ = [
    "batch_aggregate_cmd",
    "batch_cmd",
    "batch_evaluate_cmd",
    "batch_execute_cmd",
    "batch_extract_cmd",
    "batch_patch_cmd",
    "batch_prepare_cmd",
    "batch_publish_cmd",
    "batch_status_cmd",
]
