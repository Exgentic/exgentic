# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import traceback
from typing import Any

import rich_click as click

from ...core.types import ModelSettings, RunConfig
from ...utils.settings import ExgenticSettings, get_settings
from ..lib.api import get_agent_info, get_benchmark_info

DEFAULT_OUTPUT_DIR = "./outputs"


def apply_debug_mode(debug: bool) -> None:
    if not debug:
        return
    settings = get_settings()
    settings.debug = True
    settings.log_level = "DEBUG"


def _should_show_traceback() -> bool:
    return bool(get_settings().debug)


def _format_exception_for_cli(exc: Exception) -> str:
    parts = [str(exc)]
    remote_tb = getattr(exc, "__remote_traceback__", None)
    if remote_tb:
        parts.append(f"Remote traceback:\n{str(remote_tb).rstrip()}")
    local_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if local_tb.strip():
        parts.append(f"Local traceback:\n{local_tb.rstrip()}")
    stdout = getattr(exc, "stdout", None)
    if stdout:
        parts.append(f"stdout:\n{str(stdout).rstrip()}")
    stderr = getattr(exc, "stderr", None)
    if stderr:
        parts.append(f"stderr:\n{str(stderr).rstrip()}")
    return "\n\n".join(parts)


def _load_json_arg(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    if raw.startswith("@"):
        path = raw[1:]
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(raw)


def _parse_kv_list(values: tuple[str, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise click.ClickException(f"Expected key=value, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except Exception:
            value = raw
        payload[key] = value
    return payload


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    if not path or any(not part for part in path):
        raise click.ClickException("Invalid --set key (empty path).")
    cur = target
    for part in path[:-1]:
        if part in cur and not isinstance(cur[part], dict):
            raise click.ClickException(f"Conflicting --set path at '{part}': not a mapping.")
        cur = cur.setdefault(part, {})
    leaf = path[-1]
    if leaf in cur and cur[leaf] != value:
        raise click.ClickException(f"Conflicting --set value for '{'.'.join(path)}'.")
    cur[leaf] = value


def _parse_set_list(values: tuple[str, ...]) -> list[tuple[str, list[str], Any]]:
    items: list[tuple[str, list[str], Any]] = []
    for item in values:
        if "=" not in item:
            raise click.ClickException(f"Expected key=value, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except Exception:
            value = raw
        if key.startswith("benchmark."):
            path = key.split(".")[1:]
            items.append(("benchmark", path, value))
        elif key.startswith("agent.model."):
            path = ["model_settings", *key.split(".")[2:]]
            items.append(("agent", path, value))
        elif key.startswith("agent."):
            path = key.split(".")[1:]
            items.append(("agent", path, value))
        elif key.startswith("settings."):
            path = key.split(".")[1:]
            items.append(("settings", path, value))
        else:
            raise click.ClickException(
                "Invalid --set key. Use benchmark.<arg>=..., agent.<arg>=..., " "or settings.<arg>=..."
            )
    return items


def _validate_set_keys_for_benchmark(benchmark: str, items: list[tuple[str, list[str], Any]]) -> None:
    try:
        info = get_benchmark_info(benchmark)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    forbidden = {"num_tasks", "subset"}
    subset_arg = info.get("subset_arg")
    if subset_arg:
        forbidden.add(subset_arg)
    allowed = set(info.get("kwargs") or [])
    allow_any = "**kwargs" in allowed
    for group, path, _ in items:
        if group != "benchmark" or not path:
            continue
        if path[0] in forbidden:
            raise click.ClickException(f"Use --subset/--num-tasks instead of --set benchmark.{path[0]}.")
        if not allow_any and path[0] not in allowed:
            raise click.ClickException(
                f"Unknown benchmark override '{path[0]}'. " f"Available: {', '.join(sorted(allowed))}"
            )


def _validate_set_keys_for_agent(agent: str, items: list[tuple[str, list[str], Any]]) -> None:
    try:
        info = get_agent_info(agent)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    allowed = set(info.get("kwargs") or [])
    allow_any = "**kwargs" in allowed
    model_fields = set(ModelSettings.model_fields.keys())
    for group, path, _ in items:
        if group != "agent" or not path:
            continue
        if path[0] == "model_settings":
            if len(path) < 2 or path[1] not in model_fields:
                raise click.ClickException(
                    f"Unknown agent model override '{'.'.join(path)}'. " f"Available: {', '.join(sorted(model_fields))}"
                )
            continue
        if not allow_any and path[0] not in allowed:
            raise click.ClickException(
                f"Unknown agent override '{path[0]}'. " f"Available: {', '.join(sorted(allowed))}"
            )


def _validate_set_keys_for_settings(
    items: list[tuple[str, list[str], Any]],
) -> None:
    allowed = set(ExgenticSettings.model_fields.keys())
    for group, path, _ in items:
        if group != "settings" or not path:
            continue
        if len(path) != 1 or path[0] not in allowed:
            raise click.ClickException(
                f"Unknown settings override '{'.'.join(path)}'. " f"Available: {', '.join(sorted(allowed))}"
            )


def _apply_settings_overrides(items: list[tuple[str, list[str], Any]]) -> None:
    settings = get_settings()
    for group, path, value in items:
        if group != "settings" or not path:
            continue
        setattr(settings, path[0], value)


def _apply_options(target, specs):
    for args, kwargs in reversed(specs):
        target = click.option(*args, **kwargs)(target)
    return target


def _run_option_specs(
    *,
    required: bool,
    include_overwrite: bool,
    include_max_workers: bool,
):
    specs = [
        (("--benchmark",), {"required": required, "help": "Benchmark slug_name"}),
        (("--agent",), {"required": required, "help": "Agent slug_name"}),
        (("--agent-json",), {"help": "Agent kwargs JSON (or @file)"}),
        (("--agent-arg",), {"multiple": True, "help": "Agent kwarg key=value"}),
        (
            ("--set", "set_values"),
            {
                "multiple": True,
                "help": "Set benchmark.*, agent.*, or settings.* values",
            },
        ),
        (("--subset",), {"help": "Benchmark subset name"}),
        (("--task", "tasks"), {"multiple": True, "help": "Task to run (repeatable)"}),
        (("--num-tasks",), {"type": int, "help": "Number of tasks to run"}),
        (("--max-steps",), {"type": int, "help": "Max steps per session"}),
        (("--max-actions",), {"type": int, "help": "Max actions per session"}),
        (("--model",), {"help": "Agent model (must be supported by the agent)"}),
        (
            ("--debug",),
            {
                "is_flag": True,
                "help": "Enable debug mode (sets settings.debug=true and log level to DEBUG)",
            },
        ),
        (
            ("--log-level",),
            {
                "type": click.Choice(
                    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    case_sensitive=False,
                ),
                "help": "Override EXGENTIC_LOG_LEVEL for this run",
            },
        ),
        (("--output-dir",), {"default": DEFAULT_OUTPUT_DIR, "show_default": True}),
        (
            ("--cache-dir",),
            {"help": "Base cache directory (overrides EXGENTIC_CACHE_DIR)"},
        ),
        (("--run-id",), {"help": "Run id (overrides context env var)"}),
    ]
    if include_overwrite:
        specs.append(
            (
                ("--overwrite",),
                {
                    "is_flag": True,
                    "help": "Overwrite existing sessions instead of skipping them",
                },
            )
        )
    if include_max_workers:
        specs.append(
            (
                ("--max-workers",),
                {
                    "type": int,
                    "help": "Parallel workers (>=2 runs sessions in parallel)",
                },
            )
        )
    return specs


def add_run_options(
    func=None,
    *,
    required: bool = True,
    include_overwrite: bool = True,
    include_max_workers: bool = True,
):
    def decorator(target):
        specs = _run_option_specs(
            required=required,
            include_overwrite=include_overwrite,
            include_max_workers=include_max_workers,
        )
        return _apply_options(target, specs)

    if func is None:
        return decorator
    return decorator(func)


def has_run_options(
    *,
    benchmark: str | None,
    agent: str | None,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    model: str | None,
    debug: bool,
    overwrite: bool,
    log_level: str | None,
    output_dir: str | None,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
    max_steps: int | None,
    max_actions: int | None,
) -> bool:
    if benchmark or agent or agent_json or subset or num_tasks or model or log_level:
        return True
    if agent_arg or set_values or tasks:
        return True
    if overwrite or run_id or max_workers is not None:
        return True
    if max_steps is not None or max_actions is not None:
        return True
    if cache_dir:
        return True
    if output_dir and output_dir != DEFAULT_OUTPUT_DIR:
        return True
    return False


def build_run_config(
    *,
    benchmark: str,
    agent: str,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    model: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
    max_steps: int | None,
    max_actions: int | None,
    overwrite: bool,
) -> RunConfig:
    bench_kwargs: dict[str, Any] = {}
    agent_kwargs = _load_json_arg(agent_json)
    agent_kwargs.update(_parse_kv_list(agent_arg))
    set_items = _parse_set_list(set_values)
    _validate_set_keys_for_settings(set_items)
    _apply_settings_overrides(set_items)
    _validate_set_keys_for_benchmark(benchmark, set_items)
    _validate_set_keys_for_agent(agent, set_items)
    for group, path, value in set_items:
        if group == "benchmark":
            _set_nested(bench_kwargs, path, value)
        elif group == "agent":
            _set_nested(agent_kwargs, path, value)
    return RunConfig(
        benchmark=benchmark,
        agent=agent,
        subset=subset,
        task_ids=list(tasks) if tasks else None,
        num_tasks=num_tasks,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        model=model,
        max_workers=max_workers,
        max_steps=(RunConfig.model_fields["max_steps"].default if max_steps is None else max_steps),
        max_actions=(RunConfig.model_fields["max_actions"].default if max_actions is None else max_actions),
        overwrite_sessions=overwrite,
        benchmark_kwargs=bench_kwargs,
        agent_kwargs=agent_kwargs,
    )


def run_with(
    run_func,
    benchmark: str,
    agent: str,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    model: str | None,
    debug: bool,
    overwrite: bool,
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_workers: int | None,
    max_steps: int | None,
    max_actions: int | None,
) -> Any:
    """Run a benchmark with an agent by slug name."""
    if log_level:
        settings = get_settings()
        level = log_level.upper()
        settings.log_level = level
        settings.debug = level == "DEBUG"
    apply_debug_mode(debug)
    try:
        config = build_run_config(
            benchmark=benchmark,
            agent=agent,
            agent_json=agent_json,
            agent_arg=agent_arg,
            set_values=set_values,
            subset=subset,
            tasks=tasks,
            num_tasks=num_tasks,
            model=model,
            output_dir=output_dir,
            cache_dir=cache_dir,
            run_id=run_id,
            max_workers=max_workers,
            max_steps=max_steps,
            max_actions=max_actions,
            overwrite=overwrite,
        )
        return run_func(config)
    except Exception as exc:
        if _should_show_traceback():
            raise click.ClickException(_format_exception_for_cli(exc)) from exc
        raise click.ClickException(str(exc)) from exc


def run_query(
    *,
    run_func,
    render_func,
    output_format: str,
    benchmark: str,
    agent: str,
    agent_json: str | None,
    agent_arg: tuple[str, ...],
    set_values: tuple[str, ...],
    subset: str | None,
    tasks: tuple[str, ...],
    num_tasks: int | None,
    model: str | None,
    debug: bool,
    log_level: str | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    max_steps: int | None,
    max_actions: int | None,
) -> None:
    result = run_with(
        run_func,
        benchmark=benchmark,
        agent=agent,
        agent_json=agent_json,
        agent_arg=agent_arg,
        set_values=set_values,
        subset=subset,
        tasks=tasks,
        num_tasks=num_tasks,
        model=model,
        debug=debug,
        overwrite=False,
        log_level=log_level,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        max_workers=None,
        max_steps=max_steps,
        max_actions=max_actions,
    )
    render_func(result, output_format)
