# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import shutil
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from pathlib import Path

from filelock import FileLock, Timeout

from ...adapters.runners import with_runner
from ...interfaces.registry import load_agent, load_benchmark
from ...observers.logging import get_disabled_logger
from ...utils.paths import get_run_paths, get_session_paths
from ..types import (
    SessionConfig,
    SessionExecutionStatus,
    SessionIndex,
    SessionOutcomeStatus,
    SessionResults,
    SessionStatus,
)
from .session import run_session
from .termination import RunCancelError
from .tracker import Tracker

_BENCHMARK_CACHE: dict[str, type] = {}
_AGENT_CACHE: dict[str, type] = {}


def _get_benchmark_class(slug: str):
    cls = _BENCHMARK_CACHE.get(slug)
    if cls is None:
        cls = load_benchmark(slug)
        _BENCHMARK_CACHE[slug] = cls
    return cls


def _get_agent_class(slug: str):
    cls = _AGENT_CACHE.get(slug)
    if cls is None:
        cls = load_agent(slug)
        _AGENT_CACHE[slug] = cls
    return cls


def _try_reuse_completed(
    *,
    status: SessionStatus,
    session_config: SessionConfig,
    sess_paths,
    tracker: Tracker,
    log,
) -> bool:
    if status.status != SessionExecutionStatus.COMPLETED:
        return False
    if session_config.overwrite_sessions:
        return False
    results_path = sess_paths.results
    if not results_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        results = SessionResults.model_validate(payload)
        tracker.on_session_reuse(results)
        log.info(
            "Skipping completed session %s (task=%s)",
            session_config.get_session_id(),
            session_config.task_id,
        )
        return True
    except Exception:
        log.exception(
            "Failed to load results for session %s (task=%s); rerunning.",
            session_config.get_session_id(),
            session_config.task_id,
        )
        return False


def _cleanup_session_dir(
    *,
    status: SessionStatus,
    session_config: SessionConfig,
    sess_paths,
    log,
) -> None:
    if session_config.overwrite_sessions and sess_paths.root.exists():
        shutil.rmtree(sess_paths.root)
        log.info(
            "Overwriting existing session %s (task=%s)",
            session_config.get_session_id(),
            session_config.task_id,
        )
        return
    if status.status == SessionExecutionStatus.INCOMPLETE and sess_paths.root.exists():
        shutil.rmtree(sess_paths.root)
        log.info(
            "Overwriting incomplete session %s (task=%s)",
            session_config.get_session_id(),
            session_config.task_id,
        )


def _write_session_config(
    *,
    session_config: SessionConfig,
    sess_paths,
) -> None:
    config_path = sess_paths.session_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            session_config.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2,
        )


def _status_from_paths_without_lock(
    *,
    session_config: SessionConfig,
    sess_paths,
) -> SessionStatus:
    """Derive session status from filesystem without consulting lock state."""
    results_exists = sess_paths.results.exists()
    session_dir_exists = sess_paths.root.exists()
    result_status = None

    if results_exists:
        result_status = SessionStatus._extract_result_status(sess_paths.results)
        if result_status in (
            SessionOutcomeStatus.ERROR,
            SessionOutcomeStatus.CANCELLED,
            SessionOutcomeStatus.LIMIT_REACHED,
        ):
            status = SessionExecutionStatus.INCOMPLETE
        else:
            status = SessionExecutionStatus.COMPLETED
    elif session_dir_exists:
        status = SessionExecutionStatus.INCOMPLETE
    else:
        status = SessionExecutionStatus.MISSING

    return SessionStatus(
        task_id=str(session_config.task_id),
        session_id=session_config.get_session_id(),
        results_path=str(sess_paths.results),
        session_dir=str(sess_paths.root),
        status=status,
        result_status=result_status,
    )


def run_session_config(
    *,
    session_config: SessionConfig,
    tracker: Tracker,
) -> None:
    bench_cls = _get_benchmark_class(session_config.benchmark)
    agent_cls = _get_agent_class(session_config.agent)
    benchmark = bench_cls(**(session_config.benchmark_kwargs or {}))
    agent = agent_cls(**(session_config.agent_kwargs or {}))

    # Create evaluator to obtain session kwargs.
    evaluator = with_runner(
        benchmark.get_evaluator_class(),
        runner=benchmark.resolve_runner(),
        **benchmark.get_evaluator_kwargs(),
        **benchmark.runner_kwargs(),
    )

    session_id = session_config.get_session_id()
    index = SessionIndex(
        task_id=str(session_config.task_id),
        session_id=session_id,
    )

    try:
        session_kwargs = evaluator.get_session_kwargs(index)
        # Create session via with_runner for isolation.
        session = with_runner(
            benchmark.get_session_class(),
            runner=benchmark.resolve_runner(),
            **session_kwargs,
            **benchmark.runner_kwargs(),
        )
        # run_session handles session.close() internally.
        run_session(session_config, session, agent, tracker=tracker)
    finally:
        try:
            evaluator.close()
        except Exception:
            pass
        try:
            benchmark.close()
        finally:
            agent.close()


def _run_task_with_lock(
    *,
    session_config: SessionConfig,
    tracker: Tracker,
    log,
) -> None:
    session_id = session_config.get_session_id()
    sess_paths = get_session_paths(session_id)
    sess_paths.root.mkdir(parents=True, exist_ok=True)
    status = SessionStatus.from_config(session_config)
    lock = FileLock(str(sess_paths.lock))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        log.info(
            "Skipping running session %s (task=%s)",
            session_id,
            session_config.task_id,
        )
        return
    try:
        # If status was sampled while another process held the lock, refresh it
        # after acquiring the lock so cleanup/reuse decisions stay accurate.
        if status.status == SessionExecutionStatus.RUNNING:
            status = _status_from_paths_without_lock(session_config=session_config, sess_paths=sess_paths)
        if _try_reuse_completed(
            status=status,
            session_config=session_config,
            sess_paths=sess_paths,
            tracker=tracker,
            log=log,
        ):
            return
        _cleanup_session_dir(
            status=status,
            session_config=session_config,
            sess_paths=sess_paths,
            log=log,
        )
        _write_session_config(session_config=session_config, sess_paths=sess_paths)
        run_session_config(session_config=session_config, tracker=tracker)
    finally:
        if lock.is_locked:
            lock.release()


def _execute_sessions_serial(
    *,
    session_configs: list[SessionConfig],
    tracker: Tracker,
    log,
) -> bool:
    had_error = False
    try:
        for session_config in session_configs:
            _run_task_with_lock(
                session_config=session_config,
                tracker=tracker,
                log=log,
            )
    except (KeyboardInterrupt, RunCancelError) as exc:
        tracker.on_run_error(exc)
        had_error = True
    return had_error


def _execute_sessions_parallel(
    *,
    session_configs: list[SessionConfig],
    tracker: Tracker,
    max_workers: int,
    log,
) -> bool:
    had_error = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {}
        pending_tasks = deque(session_configs)
        try:
            while futures or pending_tasks:
                while len(futures) < max_workers:
                    if not pending_tasks:
                        break
                    session_config = pending_tasks.popleft()
                    ctx = copy_context()
                    future = executor.submit(
                        ctx.run,
                        _run_task_with_lock,
                        session_config=session_config,
                        tracker=tracker,
                        log=log,
                    )
                    futures[future] = session_config

                if futures:
                    done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for done in done_set:
                        session_config = futures.pop(done, None)
                        try:
                            done.result()
                        except RunCancelError as exc:
                            tracker.on_run_error(exc)
                            had_error = True
                            raise
                        except Exception as exc:
                            session_id = session_config.get_session_id() if session_config is not None else "unknown"
                            log.exception(
                                "Session task failed task=%s session=%s",
                                session_config.task_id if session_config else "unknown",
                                session_id,
                            )
                            tracker.on_run_error(exc)
                            had_error = True
                            for future in futures:
                                future.cancel()
                            raise
        except (KeyboardInterrupt, RunCancelError) as exc:
            tracker.on_run_error(exc)
            had_error = True
            for future in futures:
                future.cancel()
    return had_error


def execute_sessions(
    *,
    session_configs: list[SessionConfig],
    tracker: Tracker,
    reused_results: list[SessionResults] | None = None,
    max_workers: int | None = None,
    log=None,
) -> bool:
    if log is None:
        log = get_disabled_logger(__name__)
    if reused_results:
        for item in reused_results:
            tracker.on_session_reuse(item)

    if max_workers and max_workers > 1:
        return _execute_sessions_parallel(
            session_configs=session_configs,
            tracker=tracker,
            max_workers=max_workers,
            log=log,
        )
    return _execute_sessions_serial(
        session_configs=session_configs,
        tracker=tracker,
        log=log,
    )


def load_session_results(
    results_path: Path,
    session_id: str,
    log,
) -> SessionResults | None:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        return SessionResults.model_validate(payload)
    except Exception:
        log.exception(
            "Failed to load session results for %s at %s",
            session_id,
            results_path,
        )
        return None


def load_reused_results(
    session_configs: list[SessionConfig],
    log,
) -> list[SessionResults]:
    run_paths = get_run_paths()
    reused: list[SessionResults] = []
    for session_config in session_configs:
        session_id = session_config.get_session_id()
        results_path = run_paths.session(session_id).results
        if not results_path.exists():
            continue
        results = load_session_results(results_path, session_id, log)
        if results is not None:
            reused.append(results)
            log.info(
                "Skipping existing session %s (task=%s)",
                session_id,
                session_config.task_id,
            )
    return reused
