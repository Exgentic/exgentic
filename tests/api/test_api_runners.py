# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""End-to-end evaluate flow through every runner.

Exercises the full evaluate() pipeline — evaluator creation, session execution,
scoring, close(), and aggregation — through each transport layer.

This catches issues like:
- ContextVar propagation across uvicorn thread-pool workers (service runner)
- Path resolution mismatches between host and runner
- close() forwarding through proxied objects
- Result file persistence across runner boundaries
- Volume mount and env var consistency (docker runner)
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from exgentic import evaluate, execute
from exgentic.core.types import RunConfig

_RUNNERS = ["direct", "thread", "process", "service"]

# Detect Docker availability for conditional tests.
_docker_available = shutil.which("docker") is not None
if _docker_available:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=5)
    except Exception:
        _docker_available = False

if _docker_available and sys.version_info[:2] == (3, 12):
    _RUNNERS.append("docker")


@pytest.fixture(params=_RUNNERS)
def runner(request):
    return request.param


@pytest.fixture(scope="module")
def _docker_tmpdir():
    """Module-scoped temp dir for Docker tests under $HOME.

    Rancher Desktop / Docker Desktop on macOS only share ``/Users/`` by
    default.  pytest's ``tmp_path`` lives under ``/var/folders/`` which
    is NOT shared, so Docker volume mounts silently fail.

    Using a module-scoped parent dir avoids cleaning up temp dirs between
    tests, which would break stale logging FileHandlers held by the
    evaluate framework.
    """
    if platform.system() != "Darwin":
        yield None
        return
    d = Path(tempfile.mkdtemp(prefix=".exgentic_test_", dir=Path.home()))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def run_tmp(runner, tmp_path, _docker_tmpdir):
    """Temp path that works with Docker volume mounts on macOS."""
    if runner != "docker" or _docker_tmpdir is None:
        return tmp_path
    d = Path(tempfile.mkdtemp(dir=_docker_tmpdir))
    return d


def _run_config(tmp_path, runner: str, *, num_tasks: int = 2, policy: str = "good_then_finish") -> RunConfig:
    tasks = [f"task-{i}" for i in range(1, num_tasks + 1)]
    return RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        output_dir=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        run_id=f"run-{runner}",
        num_tasks=num_tasks,
        max_steps=10,
        max_actions=10,
        benchmark_kwargs={"tasks": tasks, "runner": runner},
        agent_kwargs={"policy": policy, "finish_after": 2},
    )


def test_evaluate_full_flow(run_tmp, runner):
    """Full evaluate (sessions + aggregation) through each runner."""
    config = _run_config(run_tmp, runner)
    results = evaluate(config)

    assert results.total_sessions == 2
    assert results.successful_sessions == 2
    assert results.benchmark_score == 1.0

    for session in results.session_results:
        assert session.score == 1.0
        assert session.success is True


def test_result_files_written(run_tmp, runner):
    """Verify all expected output files are created through each runner."""
    config = _run_config(run_tmp, runner, num_tasks=1)
    evaluate(config)

    run_dir = run_tmp / "outputs" / f"run-{runner}"
    session_id = config.to_session_config("task-1").get_session_id()
    session_dir = run_dir / "sessions" / session_id

    # Run-level files
    assert (run_dir / "results.json").exists()
    assert (run_dir / "benchmark_results.json").exists()

    # Session-level files
    assert (session_dir / "results.json").exists()
    assert (session_dir / "benchmark" / "results.json").exists()

    # Verify content is valid JSON with expected fields
    payload = json.loads((run_dir / "results.json").read_text())
    assert payload["total_sessions"] == 1
    assert payload["benchmark_score"] == 1.0


def test_execute_then_aggregate(run_tmp, runner):
    """Verify execute-only + aggregate-only works through each runner."""
    from exgentic import aggregate

    config = _run_config(run_tmp, runner, num_tasks=1)

    exec_results = execute(config)
    assert exec_results.benchmark_score is None
    assert exec_results.total_sessions == 1

    agg_results = aggregate(config)
    assert agg_results.benchmark_score == 1.0


def test_unsuccessful_session(run_tmp, runner):
    """Agent that finishes immediately scores 0 through each runner."""
    config = _run_config(run_tmp, runner, num_tasks=1, policy="finish_immediately")
    results = evaluate(config)

    assert results.total_sessions == 1
    assert results.successful_sessions == 0
    assert results.session_results[0].score == 0.0
    assert results.session_results[0].success is False


def test_parallel_workers(run_tmp, runner):
    """Evaluate with max_workers>1 to test thread-safety of pickling."""
    config = _run_config(run_tmp, runner, num_tasks=4)
    config = config.model_copy(update={"max_workers": 2})
    results = evaluate(config)

    assert results.total_sessions == 4
    assert results.successful_sessions == 4
    assert results.benchmark_score == 1.0
