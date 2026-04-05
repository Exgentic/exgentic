# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from pathlib import Path

from exgentic.core.context import (
    Context,
    Role,
    _derive_runtime_path,
    _load_runtime,
    save_service_runtime,
    set_context,
)


def test_derive_runtime_path_session_agent():
    ctx = Context(
        run_id="r1",
        output_dir="/tmp/out",
        cache_dir="/tmp/c",
        session_id="s1",
        role=Role.AGENT,
    )
    assert _derive_runtime_path(ctx) == Path("/tmp/out/r1/sessions/s1/agent/runtime.json")


def test_derive_runtime_path_session_benchmark_override():
    ctx = Context(
        run_id="r1",
        output_dir="/tmp/out",
        cache_dir="/tmp/c",
        session_id="s1",
        role=Role.FRAMEWORK,
    )
    assert _derive_runtime_path(ctx, role=Role.BENCHMARK) == Path("/tmp/out/r1/sessions/s1/benchmark/runtime.json")


def test_derive_runtime_path_run_only():
    ctx = Context(run_id="r1", output_dir="/tmp/out", cache_dir="/tmp/c", role=Role.BENCHMARK)
    assert _derive_runtime_path(ctx) == Path("/tmp/out/r1/benchmark/runtime.json")


def test_derive_runtime_path_none():
    ctx = Context(run_id="r1", output_dir="", cache_dir="/tmp/c")
    assert _derive_runtime_path(ctx) is None


def test_save_and_load_service_runtime(tmp_path):
    ctx = Context(
        run_id="run-1",
        output_dir=str(tmp_path),
        cache_dir="/tmp/cache",
        session_id="sess-1",
        task_id="task-1",
        role=Role.FRAMEWORK,
    )
    set_context(ctx)

    path = save_service_runtime(Role.AGENT)
    expected = tmp_path / "run-1" / "sessions" / "sess-1" / "agent" / "runtime.json"
    assert path == expected
    assert path.exists()

    loaded = _load_runtime(path)
    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.session_id == "sess-1"
    assert loaded.role == Role.AGENT


def test_load_runtime_missing():
    assert _load_runtime(Path("/nonexistent/runtime.json")) is None
