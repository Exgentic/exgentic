# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import logging

from ...interfaces.registry import load_benchmark
from ...observers.logging import get_logger
from ...utils.container_reaper import install_cleanup_handlers, reap_orphaned_containers
from ...utils.paths import get_run_paths
from ..types import (
    RunConfig,
    RunPlan,
    RunResults,
    RunStatus,
    SessionConfig,
    SessionExecutionStatus,
    SessionIndex,
)
from .controller import Controller
from .execution import _run_task_with_lock, execute_sessions, load_reused_results
from .observer import Observer
from .tracker import Tracker

_log = logging.getLogger(__name__)


def _build_session_indexes(run_config: RunConfig, task_ids: list[str]):
    return [run_config.to_session_config(task_id).to_index() for task_id in task_ids]


def _log_missing_session_results(
    *,
    run_config: RunConfig,
    task_ids: list[str],
    log,
) -> None:
    if not task_ids:
        return
    missing: list[str] = []
    run_paths = get_run_paths()
    for task_id in task_ids:
        session_id = run_config.to_session_config(task_id).get_session_id()
        if not run_paths.session(session_id).results.exists():
            missing.append(session_id)
    if not missing:
        return
    count = len(missing)
    total = len(task_ids)
    log.warning("Missing session results for %d/%d planned sessions.", count, total)
    preview = ", ".join(missing[:10])
    if count > 10:
        preview = f"{preview}, ..."
    log.warning("Missing session ids: %s", preview)


def core_run(
    *,
    run_config: RunConfig,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
    execute: bool,
    aggregate: bool,
) -> RunResults:
    if execute:
        try:
            reap_orphaned_containers(logger=_log)
        except Exception:
            _log.debug("Orphan container reap skipped", exc_info=True)
        install_cleanup_handlers(logger=_log)

    with run_config.get_context() as ctx:
        if run_config.run_id is None or run_config.cache_dir is None:
            updates = {}
            if run_config.run_id is None:
                updates["run_id"] = ctx.run_id
            if run_config.cache_dir is None:
                updates["cache_dir"] = ctx.cache_dir
            run_config = run_config.model_copy(update=updates)
        if execute:
            tracker = Tracker(
                observers=observers,
                controllers=controllers,
                max_steps=run_config.max_steps,
                max_actions=run_config.max_actions,
            )
        else:
            tracker = Tracker(observers=observers, controllers=controllers)
        run_paths = get_run_paths()
        log = get_logger(
            f"tracker.{run_paths.run_id}",
            str(run_paths.tracker),
        )
        status = RunStatus.from_config(run_config)
        if run_config.task_ids is None and status.task_ids:
            run_config = run_config.model_copy(update={"task_ids": status.task_ids})
        tracker.on_run_start(run_config)
        if execute:
            plan = RunPlan.from_config_and_status(run_config, status)
            reused_results = load_reused_results(plan.reuse, log)
            log.info(
                "Session selection: total=%d to_run=%d skipped=%d",
                len(status.task_ids),
                len(plan.to_run),
                len(plan.reuse),
            )
            had_error = execute_sessions(
                session_configs=plan.to_run,
                tracker=tracker,
                reused_results=reused_results,
                max_workers=run_config.max_workers,
                log=log,
            )
            if had_error:
                return tracker.results()
        else:
            reused_results = load_reused_results(
                [run_config.to_session_config(task_id) for task_id in status.task_ids],
                log,
            )
            for item in reused_results:
                tracker.on_session_reuse(item)

        results = None
        if aggregate:
            if execute:
                status = RunStatus.from_config(run_config)
            if status.task_ids:
                _log_missing_session_results(run_config=run_config, task_ids=status.task_ids, log=log)
            # Aggregate only completed sessions.
            completed = [item for item in status.session_statuses if item.status == SessionExecutionStatus.COMPLETED]
            if completed:
                session_indexes = [SessionIndex(task_id=item.task_id, session_id=item.session_id) for item in completed]
            else:
                session_indexes = []
            skipped = len(status.session_statuses) - len(session_indexes)
            if skipped:
                log.warning(
                    "Skipping %d non-completed sessions during aggregation.",
                    skipped,
                )
            if not session_indexes:
                log.warning("No completed sessions available for aggregation.")
            # Create evaluator for aggregation.
            bench_cls = load_benchmark(run_config.benchmark)
            benchmark = bench_cls(**(run_config.benchmark_kwargs or {}))
            evaluator = benchmark.get_evaluator()
            try:
                results = evaluator.aggregate_sessions(session_indexes)
            finally:
                try:
                    evaluator.close()
                except Exception:
                    pass
                benchmark.close()
        tracker.on_run_success(results, run_config)
        return tracker.results()


def core_execute(
    *,
    run_config: RunConfig,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
) -> RunResults:
    return core_run(
        run_config=run_config,
        observers=observers,
        controllers=controllers,
        execute=True,
        aggregate=False,
    )


def core_evaluate(
    *,
    run_config: RunConfig,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
) -> RunResults:
    return core_run(
        run_config=run_config,
        observers=observers,
        controllers=controllers,
        execute=True,
        aggregate=True,
    )


def core_aggregate(
    *,
    run_config: RunConfig,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
) -> RunResults:
    return core_run(
        run_config=run_config,
        observers=observers,
        controllers=controllers,
        execute=False,
        aggregate=True,
    )


# ---------------------------------------------------------------------------
# Batch evaluation with a global worker pool
# ---------------------------------------------------------------------------


def _plan_config(run_config: RunConfig) -> tuple[RunConfig, Tracker, list[SessionConfig]]:
    """Plan sessions for a config.  Returns (resolved config, tracker, sessions to run)."""
    with run_config.get_context() as ctx:
        updates = {}
        if run_config.run_id is None:
            updates["run_id"] = ctx.run_id
        if run_config.cache_dir is None:
            updates["cache_dir"] = ctx.cache_dir
        if updates:
            run_config = run_config.model_copy(update=updates)

        # Quiet tracker — no console output during batch planning.
        tracker = Tracker(
            use_defaults=False,
            max_steps=run_config.max_steps,
            max_actions=run_config.max_actions,
        )
        status = RunStatus.from_config(run_config)
        if run_config.task_ids is None and status.task_ids:
            run_config = run_config.model_copy(update={"task_ids": status.task_ids})

        tracker.on_run_start(run_config)
        plan = RunPlan.from_config_and_status(run_config, status)

    _log.info(
        "%s/%s: %d to run, %d to reuse",
        run_config.benchmark,
        run_config.agent,
        len(plan.to_run),
        len(plan.reuse),
    )
    return run_config, tracker, plan.to_run


def core_batch_evaluate(
    run_configs: list[RunConfig],
    *,
    max_workers: int = 4,
) -> list[RunResults]:
    """Evaluate multiple configs with a single global worker pool.

    1. **Plan** — resolve sessions per config (sequential, fast).
    2. **Execute** — run ALL sessions in one ``ThreadPoolExecutor``.
    3. **Aggregate** — score each config from disk (sequential).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from contextvars import copy_context

    try:
        reap_orphaned_containers(logger=_log)
    except Exception:
        _log.debug("Orphan container reap skipped", exc_info=True)
    install_cleanup_handlers(logger=_log)

    # Phase 1: plan.
    configs: list[RunConfig] = []
    work: list[tuple[SessionConfig, RunConfig, Tracker]] = []

    for cfg in run_configs:
        cfg, tracker, sessions = _plan_config(cfg)
        configs.append(cfg)
        for sc in sessions:
            work.append((sc, cfg, tracker))

    _log.info("Batch: %d sessions, %d configs, %d workers", len(work), len(configs), max_workers)

    # Phase 2: execute.
    if work:

        def _run(item: tuple[SessionConfig, RunConfig, Tracker]) -> None:
            sc, cfg, tracker = item
            with cfg.get_context():
                _run_task_with_lock(session_config=sc, tracker=tracker, log=_log)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(copy_context().run, _run, item): item for item in work}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    sc, _, _ = futures[future]
                    _log.exception("Session %s failed", sc.get_session_id())

    # Phase 3: aggregate from disk.
    results: list[RunResults] = []
    for cfg in configs:
        try:
            results.append(core_aggregate(run_config=cfg))
        except Exception:
            _log.exception("Aggregation failed for %s", cfg.run_id)
    return results
