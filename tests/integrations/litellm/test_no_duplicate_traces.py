# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Test that trace callbacks don't create duplicate entries."""

from __future__ import annotations

import json
from pathlib import Path


def test_trace_logger_writes_single_entry_on_success(tmp_path: Path) -> None:
    """Verify that a successful LLM call writes exactly one trace entry."""
    from exgentic.integrations.litellm.trace_logger import TraceLogger

    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger(file_path=str(trace_path))

    # Mock kwargs and response
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "litellm_trace_id": "test-trace-123",
    }
    response_obj = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
    }

    # Call log_success_event once
    logger.log_success_event(kwargs, response_obj, start_time=None, end_time=None)

    # Verify exactly one entry was written
    assert trace_path.exists()
    lines = trace_path.read_text().strip().split("\n")
    assert len(lines) == 1, f"Expected 1 trace entry, got {len(lines)}"

    # Verify the entry is valid JSON with correct status
    entry = json.loads(lines[0])
    assert entry["status"] == "success"
    assert entry["model"] == "gpt-4o-mini"
    assert entry["trace_id"] == "test-trace-123"


def test_trace_logger_writes_single_entry_on_failure(tmp_path: Path) -> None:
    """Verify that a failed LLM call writes exactly one trace entry."""
    from exgentic.integrations.litellm.trace_logger import TraceLogger

    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger(file_path=str(trace_path))

    # Mock kwargs and error response
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "litellm_trace_id": "test-trace-456",
        "exception": {"type": "RateLimitError", "message": "Rate limit exceeded"},
    }
    response_obj = {
        "error": {
            "type": "RateLimitError",
            "message": "Rate limit exceeded",
        },
        "usage": {},
    }

    # Call log_failure_event once
    logger.log_failure_event(kwargs, response_obj, start_time=None, end_time=None)

    # Verify exactly one entry was written
    assert trace_path.exists()
    lines = trace_path.read_text().strip().split("\n")
    assert len(lines) == 1, f"Expected 1 trace entry, got {len(lines)}"

    # Verify the entry is valid JSON with correct status
    entry = json.loads(lines[0])
    assert entry["status"] == "failure"
    assert entry["model"] == "gpt-4o-mini"
    assert entry["trace_id"] == "test-trace-456"


def test_configure_litellm_does_not_duplicate_callbacks() -> None:
    """Verify that calling configure_litellm multiple times doesn't duplicate callbacks."""
    import litellm
    from exgentic.integrations.litellm.config import configure_litellm
    from exgentic.integrations.litellm.trace_logger import AsyncTraceLogger, SyncTraceLogger
    from exgentic.utils.settings import get_settings

    # Store original callbacks
    original_success = litellm.success_callback.copy()
    original_failure = litellm.failure_callback.copy()
    original_async_success = litellm._async_success_callback.copy()
    original_async_failure = litellm._async_failure_callback.copy()

    try:
        # Clear callbacks
        litellm.success_callback.clear()
        litellm.failure_callback.clear()
        litellm._async_success_callback.clear()
        litellm._async_failure_callback.clear()

        settings = get_settings()

        # Call configure_litellm multiple times
        configure_litellm(config=settings.to_litellm_config(), cache_only=False)
        configure_litellm(config=settings.to_litellm_config(), cache_only=False)
        configure_litellm(config=settings.to_litellm_config(), cache_only=False)

        # Count instances of each callback type
        sync_success_count = sum(isinstance(cb, SyncTraceLogger) for cb in litellm.success_callback)
        async_success_count = sum(isinstance(cb, AsyncTraceLogger) for cb in litellm.success_callback)
        sync_failure_count = sum(isinstance(cb, SyncTraceLogger) for cb in litellm.failure_callback)
        async_failure_count = sum(isinstance(cb, AsyncTraceLogger) for cb in litellm.failure_callback)

        # Each callback type should appear exactly once
        assert sync_success_count == 1, f"Expected 1 SyncTraceLogger in success_callback, got {sync_success_count}"
        assert async_success_count == 1, f"Expected 1 AsyncTraceLogger in success_callback, got {async_success_count}"
        assert sync_failure_count == 1, f"Expected 1 SyncTraceLogger in failure_callback, got {sync_failure_count}"
        assert async_failure_count == 1, f"Expected 1 AsyncTraceLogger in failure_callback, got {async_failure_count}"

    finally:
        # Restore original callbacks
        litellm.success_callback = original_success
        litellm.failure_callback = original_failure
        litellm._async_success_callback = original_async_success
        litellm._async_failure_callback = original_async_failure
