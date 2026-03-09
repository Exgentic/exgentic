# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

from exgentic.core.context import run_scope, try_get_context
from exgentic.core.types import RunConfig
from exgentic.integrations.litellm.cache import build_litellm_cache
from exgentic.integrations.litellm.config import configure_litellm
from exgentic.integrations.litellm.trace_logger import (
    FILE_ENV,
    AsyncTraceLogger,
    SyncTraceLogger,
    TraceLogger,
)
from exgentic.utils.settings import ExgenticSettings, resolve_cache_path


def test_resolve_cache_path_uses_base_dir_for_relative_paths() -> None:
    assert resolve_cache_path(".exgentic_cache", ".litellm_cache") == ".exgentic_cache/.litellm_cache"


def test_resolve_cache_path_keeps_absolute_paths() -> None:
    assert resolve_cache_path(".exgentic_cache", "/tmp/litellm") == "/tmp/litellm"


def test_build_litellm_cache_resolves_relative_path_under_cache_dir(tmp_path) -> None:
    base_dir = tmp_path / "cache-root"
    settings = ExgenticSettings(
        cache_dir=str(base_dir),
        litellm_cache_dir=".litellm_cache",
    )
    cache = build_litellm_cache(settings)
    assert cache.cache.disk_cache.directory == str(base_dir / ".litellm_cache")


def test_run_config_to_session_config_preserves_cache_dir() -> None:
    run_config = RunConfig(
        benchmark="tau2",
        agent="tool_calling",
        cache_dir="/tmp/exgentic-cache",
    )
    session_config = run_config.to_session_config("task-1")
    assert session_config.cache_dir == "/tmp/exgentic-cache"


def test_run_scope_sets_and_restores_cache_env() -> None:
    before = try_get_context()
    with run_scope(
        run_id="cache-test",
        output_dir="./outputs",
        cache_dir="./cache",
    ):
        ctx = try_get_context()
        assert ctx is not None
        assert ctx.cache_dir == "./cache"
    after = try_get_context()
    if before is None:
        assert after is None
    else:
        assert after == before


def test_configure_litellm_always_registers_trace_logger_callbacks() -> None:
    import litellm

    original_callbacks = litellm.callbacks
    original_success = litellm.success_callback
    original_failure = litellm.failure_callback
    original_async_success = litellm._async_success_callback
    original_async_failure = litellm._async_failure_callback
    try:
        litellm.callbacks = []
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []

        settings = ExgenticSettings(litellm_caching=False)
        configure_litellm(config=settings.to_litellm_config(), cache_only=False)
        configure_litellm(config=settings.to_litellm_config(), cache_only=False)

        assert any(isinstance(cb, SyncTraceLogger) for cb in litellm.success_callback)
        assert any(isinstance(cb, AsyncTraceLogger) for cb in litellm.success_callback)
        assert any(isinstance(cb, SyncTraceLogger) for cb in litellm.failure_callback)
        assert any(isinstance(cb, AsyncTraceLogger) for cb in litellm.failure_callback)
        assert sum(isinstance(cb, SyncTraceLogger) for cb in litellm.success_callback) == 1
        assert sum(isinstance(cb, AsyncTraceLogger) for cb in litellm.success_callback) == 1
        assert sum(isinstance(cb, SyncTraceLogger) for cb in litellm.failure_callback) == 1
        assert sum(isinstance(cb, AsyncTraceLogger) for cb in litellm.failure_callback) == 1
        assert any(isinstance(cb, SyncTraceLogger) for cb in litellm._async_success_callback)
        assert any(isinstance(cb, AsyncTraceLogger) for cb in litellm._async_success_callback)
        assert any(isinstance(cb, SyncTraceLogger) for cb in litellm._async_failure_callback)
        assert any(isinstance(cb, AsyncTraceLogger) for cb in litellm._async_failure_callback)
    finally:
        litellm.callbacks = original_callbacks
        litellm.success_callback = original_success
        litellm.failure_callback = original_failure
        litellm._async_success_callback = original_async_success
        litellm._async_failure_callback = original_async_failure


def test_trace_logger_callback_registered_and_writes_by_default(tmp_path, monkeypatch) -> None:
    import litellm

    original_callbacks = litellm.callbacks
    original_success = litellm.success_callback
    original_failure = litellm.failure_callback
    original_async_success = litellm._async_success_callback
    original_async_failure = litellm._async_failure_callback
    try:
        litellm.callbacks = []
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []

        log_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv(FILE_ENV, str(log_path))

        settings = ExgenticSettings(litellm_caching=False)
        configure_litellm(config=settings.to_litellm_config(), cache_only=False)

        registered = [cb for cb in litellm.success_callback if isinstance(cb, TraceLogger)]
        assert len(registered) == 2

        kwargs = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_cost": 0.0,
        }
        response = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        registered[0].log_success_event(kwargs, response, None, None)

        assert log_path.exists()
        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["status"] == "success"
        assert record["model"] == "openai/gpt-4o-mini"
    finally:
        litellm.callbacks = original_callbacks
        litellm.success_callback = original_success
        litellm.failure_callback = original_failure
        litellm._async_success_callback = original_async_success
        litellm._async_failure_callback = original_async_failure
