# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from ...interfaces.registry import load_benchmark
from ...observers.logging import get_logger
from ...utils.paths import get_run_paths
from ..types import (
    RunConfig,
    RunPlan,
    RunResults,
    RunStatus,
    SessionExecutionStatus,
    SessionIndex,
)
from .controller import Controller
from .execution import execute_sessions, load_reused_results
from .observer import Observer
from .tracker import Tracker


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
