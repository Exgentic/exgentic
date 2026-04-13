# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import inspect
import json
from typing import Any

from ...core.agent import Agent
from ...core.benchmark import Benchmark
from ...core.orchestrator.run import (
    core_aggregate,
    core_evaluate,
    core_execute,
)
from ...core.types import RunConfig, RunPlan, RunResults, RunStatus, SessionConfig
from ..registry import (
    apply_subset_kwargs,
    get_agent_entries,
    get_benchmark_entries,
    get_benchmark_subsets,
    load_agent,
    load_benchmark,
)


def list_benchmarks() -> list[dict[str, Any]]:
    from ...environment.instance import get_manager

    mgr = get_manager()
    entries = get_benchmark_entries()
    result = []
    for slug_name, entry in entries.items():
        name = f"benchmarks/{slug_name}"
        info = mgr.get_info(name)
        installed = info is not None
        installed_at = None
        if info:
            envs = info["environments"]
            timestamps = [e["installed_at"] for e in envs.values() if "installed_at" in e]
            installed_at = min(timestamps) if timestamps else None
        result.append(
            {
                "slug_name": slug_name,
                "display_name": entry.display_name,
                "installed": installed,
                "installed_at": installed_at,
            }
        )
    return result


def list_agents() -> list[dict[str, Any]]:
    from ...environment.instance import get_manager

    mgr = get_manager()
    entries = get_agent_entries()
    result = []
    for slug_name, entry in entries.items():
        name = f"agents/{slug_name}"
        info = mgr.get_info(name)
        installed = info is not None
        installed_at = None
        if info:
            envs = info["environments"]
            timestamps = [e["installed_at"] for e in envs.values() if "installed_at" in e]
            installed_at = min(timestamps) if timestamps else None
        result.append(
            {
                "slug_name": slug_name,
                "display_name": entry.display_name,
                "installed": installed,
                "installed_at": installed_at,
            }
        )
    return result


def load_benchmark_class(benchmark: str) -> type[Benchmark]:
    entries = get_benchmark_entries()
    if benchmark not in entries:
        raise ValueError(f"Unknown benchmark slug '{benchmark}'. Available: {', '.join(sorted(entries.keys()))}")
    return load_benchmark(benchmark)


def load_agent_class(agent: str) -> type[Agent]:
    entries = get_agent_entries()
    if agent not in entries:
        raise ValueError(f"Unknown agent slug '{agent}'. Available: {', '.join(sorted(entries.keys()))}")
    return load_agent(agent)


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
        overwrite_sessions=session_config.overwrite_sessions,
    )


def _normalize_run_config(
    config: RunConfig | SessionConfig | None,
    *,
    benchmark: str | Benchmark | None,
    agent: str | Agent | None,
    subset: str | None,
    task_ids: list[str] | None,
    num_tasks: int | None,
    output_dir: str,
    cache_dir: str | None,
    run_id: str | None,
    model: str | None,
    max_workers: int | None,
    max_steps: int | None,
    max_actions: int | None,
    overwrite_sessions: bool,
    benchmark_kwargs: dict[str, Any] | None,
    agent_kwargs: dict[str, Any] | None,
) -> RunConfig:
    if config is not None:
        if (
            any(
                value is not None
                for value in (
                    benchmark,
                    agent,
                    subset,
                    task_ids,
                    cache_dir,
                    run_id,
                    model,
                    benchmark_kwargs,
                    agent_kwargs,
                )
            )
            or overwrite_sessions
            or output_dir != "./outputs"
        ):
            raise ValueError("Do not pass run parameters together with config.")
        if isinstance(config, SessionConfig):
            config = _run_config_from_session(config)
        return config.with_overrides(
            num_tasks=num_tasks,
            max_workers=max_workers,
            max_steps=max_steps,
            max_actions=max_actions,
        )
    if benchmark is None or agent is None:
        raise ValueError("benchmark and agent are required.")

    bench_slug: str
    bench_kwargs: dict[str, Any]
    if isinstance(benchmark, Benchmark):
        if benchmark_kwargs is not None or subset is not None:
            raise ValueError("Do not pass benchmark args with a benchmark instance.")
        bench_slug = benchmark.slug_name
        bench_kwargs = benchmark.model_dump()
        subset = getattr(benchmark, "subset", None)
    else:
        bench_slug = benchmark
        bench_kwargs = dict(benchmark_kwargs or {})

    agent_slug: str
    agent_cfg: dict[str, Any]
    if isinstance(agent, Agent):
        if agent_kwargs is not None or model is not None:
            raise ValueError("Do not pass agent args with an agent instance.")
        agent_slug = agent.slug_name
        agent_cfg = agent.model_dump()
    else:
        agent_slug = agent
        agent_cfg = dict(agent_kwargs or {})

    optional = {}
    if max_steps is not None:
        optional["max_steps"] = max_steps
    if max_actions is not None:
        optional["max_actions"] = max_actions

    return RunConfig(
        benchmark=bench_slug,
        agent=agent_slug,
        subset=subset,
        task_ids=task_ids,
        num_tasks=num_tasks,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        model=model,
        max_workers=max_workers,
        overwrite_sessions=overwrite_sessions,
        **optional,
        benchmark_kwargs=bench_kwargs,
        agent_kwargs=agent_cfg,
    )


def evaluate(
    config: RunConfig | SessionConfig | None = None,
    *,
    benchmark: str | Benchmark | None = None,
    agent: str | Agent | None = None,
    subset: str | None = None,
    task_ids: list[str] | None = None,
    num_tasks: int | None = None,
    output_dir: str = "./outputs",
    cache_dir: str | None = None,
    run_id: str | None = None,
    model: str | None = None,
    max_workers: int | None = None,
    max_steps: int | None = None,
    max_actions: int | None = None,
    overwrite_sessions: bool = False,
    benchmark_kwargs: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    observers: list[Any] | None = None,
    controllers: list[Any] | None = None,
) -> RunResults:
    """Evaluate sessions and aggregate results.

    Accepts either a RunConfig/SessionConfig or benchmark/agent identifiers.

    Args:
        config: RunConfig or SessionConfig. When provided, no other run args
            may be passed.
        benchmark: Benchmark slug or Benchmark instance.
        agent: Agent slug or Agent instance.
        subset: Benchmark subset name.
        task_ids: Explicit task ids to run.
        num_tasks: Number of tasks to run.
        output_dir: Output root directory.
        cache_dir: Cache directory.
        run_id: Run id override.
        model: Agent model override.
        max_workers: Parallel workers.
        max_steps: Max steps per session.
        max_actions: Max actions per session.
        overwrite_sessions: Overwrite existing session artifacts.
        benchmark_kwargs: Benchmark kwargs (when benchmark is a slug).
        agent_kwargs: Agent kwargs (when agent is a slug).
        observers: Optional observers.
        controllers: Optional controllers.

    Returns:
        RunResults: Aggregated run results.

    Raises:
        ValueError: If config is combined with other run args or if instance
            args are mixed with kwargs.
    """
    run_config = _normalize_run_config(
        config,
        benchmark=benchmark,
        agent=agent,
        subset=subset,
        task_ids=task_ids,
        num_tasks=num_tasks,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        model=model,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
        overwrite_sessions=overwrite_sessions,
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs=agent_kwargs,
    )
    return core_evaluate(
        run_config=run_config,
        observers=observers,
        controllers=controllers,
    )


def execute(
    config: RunConfig | SessionConfig | None = None,
    *,
    benchmark: str | Benchmark | None = None,
    agent: str | Agent | None = None,
    subset: str | None = None,
    task_ids: list[str] | None = None,
    num_tasks: int | None = None,
    output_dir: str = "./outputs",
    cache_dir: str | None = None,
    run_id: str | None = None,
    model: str | None = None,
    max_workers: int | None = None,
    max_steps: int | None = None,
    max_actions: int | None = None,
    overwrite_sessions: bool = False,
    benchmark_kwargs: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    observers: list[Any] | None = None,
    controllers: list[Any] | None = None,
) -> RunResults:
    """Run sessions without aggregation.

    Accepts either a RunConfig/SessionConfig or benchmark/agent identifiers.

    Args:
        config: RunConfig or SessionConfig. When provided, no other run args
            may be passed.
        benchmark: Benchmark slug or Benchmark instance.
        agent: Agent slug or Agent instance.
        subset: Benchmark subset name.
        task_ids: Explicit task ids to run.
        num_tasks: Number of tasks to run.
        output_dir: Output root directory.
        cache_dir: Cache directory.
        run_id: Run id override.
        model: Agent model override.
        max_workers: Parallel workers.
        max_steps: Max steps per session.
        max_actions: Max actions per session.
        overwrite_sessions: Overwrite existing session artifacts.
        benchmark_kwargs: Benchmark kwargs (when benchmark is a slug).
        agent_kwargs: Agent kwargs (when agent is a slug).
        observers: Optional observers.
        controllers: Optional controllers.

    Returns:
        RunResults: Run results without aggregation.

    Raises:
        ValueError: If config is combined with other run args or if instance
            args are mixed with kwargs.
    """
    run_config = _normalize_run_config(
        config,
        benchmark=benchmark,
        agent=agent,
        subset=subset,
        task_ids=task_ids,
        num_tasks=num_tasks,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        model=model,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
        overwrite_sessions=overwrite_sessions,
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs=agent_kwargs,
    )
    return core_execute(
        run_config=run_config,
        observers=observers,
        controllers=controllers,
    )


def aggregate(
    config: RunConfig | SessionConfig | None = None,
    *,
    benchmark: str | Benchmark | None = None,
    agent: str | Agent | None = None,
    subset: str | None = None,
    task_ids: list[str] | None = None,
    num_tasks: int | None = None,
    output_dir: str = "./outputs",
    cache_dir: str | None = None,
    run_id: str | None = None,
    model: str | None = None,
    max_workers: int | None = None,
    max_steps: int | None = None,
    max_actions: int | None = None,
    overwrite_sessions: bool = False,
    benchmark_kwargs: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    observers: list[Any] | None = None,
    controllers: list[Any] | None = None,
) -> RunResults:
    """Aggregate results from completed sessions.

    Accepts either a RunConfig/SessionConfig or benchmark/agent identifiers.

    Args:
        config: RunConfig or SessionConfig. When provided, no other run args
            may be passed.
        benchmark: Benchmark slug or Benchmark instance.
        agent: Agent slug or Agent instance.
        subset: Benchmark subset name.
        task_ids: Explicit task ids to run.
        num_tasks: Number of tasks to run.
        output_dir: Output root directory.
        cache_dir: Cache directory.
        run_id: Run id override.
        model: Agent model override.
        max_workers: Parallel workers.
        max_steps: Max steps per session.
        max_actions: Max actions per session.
        overwrite_sessions: Overwrite existing session artifacts.
        benchmark_kwargs: Benchmark kwargs (when benchmark is a slug).
        agent_kwargs: Agent kwargs (when agent is a slug).
        observers: Optional observers.
        controllers: Optional controllers.

    Returns:
        RunResults: Aggregated run results.

    Raises:
        ValueError: If config is combined with other run args or if instance
            args are mixed with kwargs.
    """
    run_config = _normalize_run_config(
        config,
        benchmark=benchmark,
        agent=agent,
        subset=subset,
        task_ids=task_ids,
        num_tasks=num_tasks,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        model=model,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
        overwrite_sessions=overwrite_sessions,
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs=agent_kwargs,
    )
    return core_aggregate(
        run_config=run_config,
        observers=observers,
        controllers=controllers,
    )


def status(
    config: RunConfig | SessionConfig | None = None,
    *,
    benchmark: str | Benchmark | None = None,
    agent: str | Agent | None = None,
    subset: str | None = None,
    task_ids: list[str] | None = None,
    num_tasks: int | None = None,
    output_dir: str = "./outputs",
    cache_dir: str | None = None,
    run_id: str | None = None,
    model: str | None = None,
    max_workers: int | None = None,
    max_steps: int | None = None,
    max_actions: int | None = None,
    overwrite_sessions: bool = False,
    benchmark_kwargs: dict[str, Any] | None = None,
    agent_kwargs: dict[str, Any] | None = None,
) -> RunStatus:
    run_config = _normalize_run_config(
        config,
        benchmark=benchmark,
        agent=agent,
        subset=subset,
        task_ids=task_ids,
        num_tasks=num_tasks,
        output_dir=output_dir,
        cache_dir=cache_dir,
        run_id=run_id,
        model=model,
        max_workers=max_workers,
        max_steps=max_steps,
        max_actions=max_actions,
        overwrite_sessions=overwrite_sessions,
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs=agent_kwargs,
    )
    return RunStatus.from_config(run_config)


def preview(config: RunConfig) -> RunPlan:
    status = RunStatus.from_config(config)
    return RunPlan.from_config_and_status(
        config,
        status,
    )


def results(config: RunConfig) -> RunResults:
    from ...core.context import get_context
    from ...utils.paths import RunPaths

    with config.get_context():
        results_path = RunPaths.from_context(get_context()).results
        if not results_path.exists():
            raise ValueError(f"Run results not found at {results_path}.")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        return RunResults.model_validate(payload)


def get_benchmark_info(benchmark: str) -> dict[str, Any]:
    entries = get_benchmark_entries()
    entry = entries.get(benchmark)
    if entry is None:
        raise ValueError(f"Unknown benchmark slug '{benchmark}'. Available: {', '.join(sorted(entries.keys()))}")
    bench_cls = load_benchmark_class(benchmark)
    return {
        "slug_name": entry.slug_name,
        "display_name": entry.display_name,
        "subsets": list(entry.subsets),
        "subset_arg": entry.subset_arg,
        "task_ids_arg": entry.task_ids_arg,
        "task_id_type": entry.task_id_type,
        "kwargs": _describe_init_args(bench_cls),
    }


def get_agent_info(agent: str) -> dict[str, Any]:
    entries = get_agent_entries()
    entry = entries.get(agent)
    if entry is None:
        raise ValueError(f"Unknown agent slug '{agent}'. Available: {', '.join(sorted(entries.keys()))}")
    agent_cls = load_agent_class(agent)
    return {
        "slug_name": entry.slug_name,
        "display_name": entry.display_name,
        "kwargs": _describe_init_args(agent_cls),
    }


def list_subsets(benchmark: str) -> list[str]:
    benchmark_entries = get_benchmark_entries()
    if benchmark not in benchmark_entries:
        raise ValueError(
            f"Unknown benchmark slug '{benchmark}'. Available: {', '.join(sorted(benchmark_entries.keys()))}"
        )
    return get_benchmark_subsets(benchmark)


def list_tasks(
    *,
    benchmark: str,
    subset: str | None = None,
    benchmark_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    from ...core.types.run import _load_cached_task_ids, _save_cached_task_ids, _task_ids_cache_key

    benchmark_entries = get_benchmark_entries()
    if benchmark not in benchmark_entries:
        raise ValueError(
            f"Unknown benchmark slug '{benchmark}'. Available: {', '.join(sorted(benchmark_entries.keys()))}"
        )
    bench_kwargs = dict(benchmark_kwargs or {})
    if subset is not None:
        bench_kwargs = apply_subset_kwargs(benchmark, subset, bench_kwargs)
    bench_cls = load_benchmark_class(benchmark)
    benchmark_obj: Benchmark = bench_cls(**bench_kwargs)
    cache_key = _task_ids_cache_key(benchmark, benchmark_obj.subset_name, bench_kwargs or None)
    cached = _load_cached_task_ids(benchmark, cache_key)
    if cached is not None:
        benchmark_obj.close()
        return cached
    evaluator = benchmark_obj.get_evaluator()
    try:
        try:
            tasks = evaluator.list_tasks()
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
    finally:
        try:
            evaluator.close()
        except Exception:
            pass
        benchmark_obj.close()
    _save_cached_task_ids(benchmark, cache_key, tasks)
    return tasks


def needs_setup(name: str, kind: str) -> bool:
    """Return True if a benchmark/agent has a setup.sh or requirements.txt."""
    from ...environment.helpers import find_package_file

    entries = get_benchmark_entries() if kind == "benchmark" else get_agent_entries()
    entry = entries.get(name)
    if entry is None:
        return False
    return (
        find_package_file(entry.module, "setup.sh") is not None
        or find_package_file(entry.module, "requirements.txt") is not None
    )


def _describe_init_args(cls: type) -> list[str]:
    model_fields = getattr(cls, "model_fields", None)
    if model_fields:
        names = set(model_fields.keys())
        for field in model_fields.values():
            alias = getattr(field, "alias", None)
            if alias and alias not in names:
                names.add(alias)
        return sorted(names)
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return []
    args = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == param.VAR_KEYWORD:
            args.append("**kwargs")
            continue
        args.append(name)
    return args


__all__ = [
    "aggregate",
    "evaluate",
    "execute",
    "list_agents",
    "list_benchmarks",
    "list_subsets",
    "list_tasks",
    "preview",
    "results",
    "status",
]
