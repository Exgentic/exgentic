# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from ...core.agent import Agent
from ...core.benchmark import Benchmark
from ...core.orchestrator.run import (
    core_aggregate,
    core_evaluate,
    core_execute,
)
from ...core.types import RunConfig, RunPlan, RunResults, RunStatus, SessionConfig
from ...utils.installation_tracker import (
    get_all_installations,
    record_installation,
)
from ..registry import (
    apply_subset_kwargs,
    get_agent_entries,
    get_benchmark_entries,
    get_benchmark_subsets,
    load_agent,
    load_benchmark,
)


def list_benchmarks() -> list[dict[str, Any]]:
    entries = get_benchmark_entries()
    installations = get_all_installations("benchmark")
    return [
        {
            "slug_name": slug,
            "display_name": entry.display_name,
            "installed": slug in installations,
            "installed_at": installations.get(slug, {}).get("installed_at"),
        }
        for slug, entry in sorted(entries.items())
    ]


def list_agents() -> list[dict[str, Any]]:
    entries = get_agent_entries()
    installations = get_all_installations("agent")
    return [
        {
            "slug_name": slug,
            "display_name": entry.display_name,
            "installed": slug in installations,
            "installed_at": installations.get(slug, {}).get("installed_at"),
        }
        for slug, entry in sorted(entries.items())
    ]


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
    max_steps: int,
    max_actions: int,
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
                    num_tasks,
                    cache_dir,
                    run_id,
                    model,
                    max_workers,
                    benchmark_kwargs,
                    agent_kwargs,
                )
            )
            or overwrite_sessions
            or output_dir != "./outputs"
            or max_steps != 100
            or max_actions != 100
        ):
            raise ValueError("Do not pass run parameters together with config.")
        if isinstance(config, SessionConfig):
            return _run_config_from_session(config)
        return config
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
        max_steps=max_steps,
        max_actions=max_actions,
        overwrite_sessions=overwrite_sessions,
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
    max_steps: int = 100,
    max_actions: int = 100,
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
    max_steps: int = 100,
    max_actions: int = 100,
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
    max_steps: int = 100,
    max_actions: int = 100,
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
    max_steps: int = 100,
    max_actions: int = 100,
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
    benchmark_entries = get_benchmark_entries()
    if benchmark not in benchmark_entries:
        raise ValueError(
            f"Unknown benchmark slug '{benchmark}'. Available: {', '.join(sorted(benchmark_entries.keys()))}"
        )
    bench_kwargs = dict(benchmark_kwargs or {})
    if subset is not None:
        bench_kwargs = apply_subset_kwargs(benchmark, subset, bench_kwargs)
    from ...adapters.runners import with_runner

    bench_cls = load_benchmark_class(benchmark)
    benchmark_obj: Benchmark = bench_cls(**bench_kwargs)
    evaluator = with_runner(
        benchmark_obj.get_evaluator_class(),
        runner=benchmark_obj.resolve_runner(),
        **benchmark_obj.get_evaluator_kwargs(),
        **benchmark_obj.runner_kwargs(),
    )
    try:
        try:
            return evaluator.list_tasks()
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
    finally:
        try:
            evaluator.close()
        except Exception:
            pass
        benchmark_obj.close()


def _find_package_file(module_path: str, filename: str) -> Path | None:
    """Locate *filename* in the package directory for *module_path*.

    Walks up from the deepest package (e.g. ``exgentic.agents.cli.claude``)
    toward the root until it finds the file.  This lets CLI sub-agents
    share a single ``requirements.txt`` at ``agents/cli/``.
    """
    parts = module_path.split(".")
    # Start from the deepest package directory, walk up to top-level package.
    for depth in range(len(parts) - 1, 1, -1):
        package = ".".join(parts[:depth])
        try:
            candidate = resources.files(package) / filename
        except Exception:
            continue
        if candidate.is_file():
            return Path(str(candidate))
    return None


def _install_requirements(slug: str, kind: str, *, venv_python: str | None = None) -> None:
    """Install dependencies from the requirements.txt in the benchmark/agent folder.

    Args:
        slug: The slug name of the benchmark or agent.
        kind: Either "benchmark" or "agent".
        venv_python: If provided, install into this Python interpreter's env
                     instead of the host Python.
    """
    entries = get_benchmark_entries() if kind == "benchmark" else get_agent_entries()
    entry = entries.get(slug)
    if entry is None:
        return
    req_path = _find_package_file(entry.module, "requirements.txt")
    if req_path is None:
        return
    # Skip if requirements.txt is empty or only has comments
    lines = [
        line.strip() for line in req_path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    # Skip Git LFS smudge filters during install — pip only needs the
    # Python source, not large data files tracked by LFS.  Benchmarks
    # that need LFS data should fetch it in their setup.sh instead.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    python_target = venv_python or sys.executable
    subprocess.run(
        ["uv", "pip", "install", "--python", python_target, "-r", str(req_path)],
        check=True,
        env=env,
    )

    # Only refresh host import paths when installing into the host env.
    if venv_python is None:
        import importlib
        import site

        importlib.invalidate_caches()
        for site_dir in site.getsitepackages():
            if site_dir not in sys.path:
                sys.path.insert(0, site_dir)


def _setup_env() -> dict[str, str]:
    """Build environment for setup subprocesses with cache dir."""
    from ...utils.settings import get_settings

    env = os.environ.copy()
    env["EXGENTIC_CACHE_DIR"] = str(Path(get_settings().cache_dir).resolve())
    return env


def needs_setup(name: str, kind: str) -> bool:
    """Return True if a benchmark/agent has a setup.sh or requirements.txt."""
    entries = get_benchmark_entries() if kind == "benchmark" else get_agent_entries()
    entry = entries.get(name)
    if entry is None:
        return False
    return (
        _find_package_file(entry.module, "setup.sh") is not None
        or _find_package_file(entry.module, "requirements.txt") is not None
    )


def get_setup_script_path(benchmark: str) -> str:
    entries = get_benchmark_entries()
    entry = entries.get(benchmark)
    if entry is None:
        raise ValueError(f"Unknown benchmark slug '{benchmark}'. Available: {', '.join(sorted(entries.keys()))}")
    path = _find_package_file(entry.module, "setup.sh")
    if path is None:
        raise ValueError(f"setup.sh not found for benchmark '{benchmark}'.")
    return str(path)


def setup_benchmark(benchmark: str, force: bool = False, runner: str | None = None) -> None:
    """Install requirements, call Benchmark.setup(), run setup.sh if present.

    Args:
        benchmark: Benchmark slug name
        force: Force reinstall even if already installed
        runner: Runner type. When ``"venv"``, creates an isolated venv and
                installs requirements there instead of the host Python.
    """
    from ...utils.installation_tracker import get_installations_dir, is_installed

    # Check if already installed
    if not force and is_installed(benchmark, "benchmark"):
        print(f"Benchmark '{benchmark}' is already installed. Use --force to reinstall.")
        return

    # Remove installation marker before starting
    if is_installed(benchmark, "benchmark"):
        install_file = get_installations_dir() / "benchmark" / f"{benchmark}.json"
        install_file.unlink(missing_ok=True)

    venv_python: str | None = None
    venv_dir: str | None = None

    if runner == "venv":
        import shutil

        from ...utils.cache import benchmark_cache_dir

        venv_path = benchmark_cache_dir(benchmark) / "venv"
        venv_dir = str(venv_path)

        uv_bin = shutil.which("uv")
        if uv_bin is None:
            raise RuntimeError("uv CLI not found on PATH — required for venv runner")

        if not (venv_path / "bin" / "python").exists():
            venv_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [uv_bin, "venv", str(venv_path), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
                check=True,
                capture_output=True,
                text=True,
            )
            # Install exgentic into the venv.
            from ...adapters.runners._utils import find_project_root

            root = find_project_root()
            subprocess.run(
                [uv_bin, "pip", "install", "--python", str(venv_path / "bin" / "python"), "--no-cache", str(root)],
                check=True,
                capture_output=True,
                text=True,
            )

        venv_python = str(venv_path / "bin" / "python")

    _install_requirements(benchmark, "benchmark", venv_python=venv_python)
    bench_cls = load_benchmark_class(benchmark)
    bench_cls.setup()
    _run_setup_script_if_exists(benchmark, "benchmark", venv_dir=venv_dir)
    record_installation(benchmark, "benchmark")


def get_agent_setup_script_path(agent: str) -> str:
    entries = get_agent_entries()
    entry = entries.get(agent)
    if entry is None:
        raise ValueError(f"Unknown agent slug '{agent}'. Available: {', '.join(sorted(entries.keys()))}")
    path = _find_package_file(entry.module, "setup.sh")
    if path is None:
        raise ValueError(f"setup.sh not found for agent '{agent}'.")
    return str(path)


def setup_agent(agent: str, force: bool = False, runner: str | None = None) -> None:
    """Install requirements, call Agent.setup(), run setup.sh if present.

    Args:
        agent: Agent slug name
        force: Force reinstall even if already installed
        runner: Runner type. When ``"venv"``, creates an isolated venv and
                installs requirements there instead of the host Python.
    """
    from ...utils.installation_tracker import get_installations_dir, is_installed

    # Check if already installed
    if not force and is_installed(agent, "agent"):
        print(f"Agent '{agent}' is already installed. Use --force to reinstall.")
        return

    # Remove installation marker before starting
    if is_installed(agent, "agent"):
        install_file = get_installations_dir() / "agent" / f"{agent}.json"
        install_file.unlink(missing_ok=True)

    venv_python: str | None = None
    venv_dir: str | None = None

    if runner == "venv":
        import shutil

        from ...utils.cache import benchmark_cache_dir

        venv_path = benchmark_cache_dir(agent) / "venv"
        venv_dir = str(venv_path)

        uv_bin = shutil.which("uv")
        if uv_bin is None:
            raise RuntimeError("uv CLI not found on PATH — required for venv runner")

        if not (venv_path / "bin" / "python").exists():
            venv_path.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [uv_bin, "venv", str(venv_path), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
                check=True,
                capture_output=True,
                text=True,
            )
            # Install exgentic into the venv.
            from ...adapters.runners._utils import find_project_root

            root = find_project_root()
            subprocess.run(
                [uv_bin, "pip", "install", "--python", str(venv_path / "bin" / "python"), "--no-cache", str(root)],
                check=True,
                capture_output=True,
                text=True,
            )

        venv_python = str(venv_path / "bin" / "python")

    _install_requirements(agent, "agent", venv_python=venv_python)
    agent_cls = load_agent_class(agent)
    agent_cls.setup()
    _run_setup_script_if_exists(agent, "agent", venv_dir=venv_dir)
    record_installation(agent, "agent")


def _run_setup_script_if_exists(slug: str, kind: str, *, venv_dir: str | None = None) -> None:
    """Run setup.sh for a benchmark or agent if it exists.

    Args:
        slug: The slug name.
        kind: Either "benchmark" or "agent".
        venv_dir: If provided, run setup.sh with this venv activated.
    """
    try:
        if kind == "benchmark":
            setup_path = get_setup_script_path(slug)
        else:
            setup_path = get_agent_setup_script_path(slug)
        env = _setup_env()
        if venv_dir is not None:
            env["VIRTUAL_ENV"] = venv_dir
            venv_bin = str(Path(venv_dir) / "bin")
            env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        subprocess.run(["bash", setup_path], check=True, env=env)
    except ValueError:
        pass  # No setup.sh — fine


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
    "get_agent_setup_script_path",
    "get_setup_script_path",
    "list_agents",
    "list_benchmarks",
    "list_subsets",
    "list_tasks",
    "preview",
    "results",
    "setup_agent",
    "setup_benchmark",
    "status",
]
