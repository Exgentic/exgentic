# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import rich_click as click

from ....core.types import RunConfig, SessionConfig
from ....core.types.session import SessionExecutionStatus, SessionOutcomeStatus
from ....utils.paths import get_run_paths, get_session_paths
from ...lib.api import aggregate, evaluate, execute, status
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
            key = f"agent_kwargs.{key[len('agent.'):]}"
        elif key.startswith("benchmark."):
            key = f"benchmark_kwargs.{key[len('benchmark.'):]}"
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


def _update_session_ids_in_dir(session_dir: Path, new_id: str) -> None:
    for json_path in session_dir.rglob("*.json"):
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
        for cfg_path in sessions_root.rglob("config.json"):
            if cfg_path.parent.parent != sessions_root:
                continue
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
) -> tuple[str, str, str, int | None, int | None, int | None, int | None]:
    if not path or not Path(path).is_file():
        return "-", "-", "-", None, None, None, None
    try:
        payload = _load_config_file(path)
    except Exception:
        return "-", "-", "-", None, None, None, None
    score = payload.get("benchmark_score")
    if score is None:
        score = payload.get("average_score")
    cost = payload.get("total_run_cost")
    if cost is None:
        cost = payload.get("total_agent_cost")
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
@click.pass_context
def batch_status_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
) -> None:
    """Show a status table for multiple config files."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    rows: list[dict[str, str]] = []

    for i, config_path in enumerate(config_paths, start=1):
        row: dict[str, str] = {
            "#": str(i),
            "config": _format_config_link(config_path, max_len=30),
            "run_id": "-",
            "benchmark": "-",
            "agent": "-",
            "subset": "-",
            "models": "-",
            "ready": "-",
            "aggregated": "-",
            "finished": "-",
            "errors": "-",
            "score": "-",
            "cost": "-",
        }
        try:
            cfg = _load_run_like_config(config_path)
            run_status = status(cfg)
            (
                score,
                cost,
                models,
                ready,
                finished,
                errors,
                aggregated,
            ) = _load_results_summary(run_status.results_path)
            if models == "-":
                models = run_status.model_name or "-"
            models = _truncate_leading(models, 24)
            if ready is None or finished is None or errors is None:
                ready = 0
                finished = 0
                errors = 0
                if aggregated is None:
                    aggregated = 0
                for item in run_status.session_statuses:
                    if item.status != SessionExecutionStatus.COMPLETED:
                        continue
                    if item.result_status in (
                        SessionOutcomeStatus.ERROR,
                        SessionOutcomeStatus.CANCELLED,
                    ):
                        errors += 1
                    else:
                        ready += 1
                        if aggregated is not None:
                            aggregated += 1
                    if item.result_status in (
                        SessionOutcomeStatus.SUCCESS,
                        SessionOutcomeStatus.UNSUCCESSFUL,
                    ):
                        finished += 1
            row.update(
                {
                    "run_id": run_status.run_id,
                    "benchmark": run_status.benchmark_slug_name,
                    "agent": run_status.agent_slug_name,
                    "subset": run_status.subset_name or "-",
                    "models": models,
                    "ready": f"{ready}/{run_status.total_tasks}",
                    "aggregated": f"{aggregated}/{run_status.total_tasks}",
                    "finished": f"{finished}/{run_status.total_tasks}",
                    "errors": f"{errors}/{run_status.total_tasks}",
                    "score": score,
                    "cost": cost,
                }
            )
        except Exception:
            pass
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
@click.pass_context
def batch_evaluate_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
) -> None:
    """Evaluate configs sequentially."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    failures: list[tuple[str, str]] = []

    for config_path in config_paths:
        click.echo(f"Running: {config_path}")
        try:
            cfg = _load_run_like_config(config_path)
            evaluate(config=cfg)
        except Exception as exc:
            failures.append((config_path, str(exc)))
            click.echo(f"Error: {config_path}\n  {_format_batch_error(exc)}")

    if failures:
        raise click.ClickException("Batch evaluate completed with errors in " + ", ".join(path for path, _ in failures))


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
@click.pass_context
def batch_execute_cmd(
    ctx: click.Context,
    debug: bool,
    config_values: tuple[str, ...],
) -> None:
    """Execute configs sequentially (no aggregation)."""
    apply_debug_mode(debug)
    config_paths = _expand_config_inputs(config_values, list(ctx.args))
    failures: list[tuple[str, str]] = []

    for config_path in config_paths:
        click.echo(f"Running: {config_path}")
        try:
            cfg = _load_run_like_config(config_path)
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
        for cfg_path in sessions_root.rglob("config.json"):
            if cfg_path.parent.parent != sessions_root:
                continue
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


__all__ = [
    "batch_cmd",
    "batch_status_cmd",
    "batch_evaluate_cmd",
    "batch_execute_cmd",
    "batch_prepare_cmd",
    "batch_aggregate_cmd",
    "batch_patch_cmd",
    "batch_extract_cmd",
]
