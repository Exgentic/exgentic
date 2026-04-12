# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

from exgentic.core.context import Context, Role
from exgentic.integrations.litellm.trace_logger import (
    FILE_ENV,
    TraceLogger,
)


def _sample_payload() -> tuple[dict, dict]:
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_cost": 0.0,
    }
    response = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    return kwargs, response


def test_trace_logger_writes(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    logger.log_success_event(kwargs, response, None, None)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "success"
    assert record["model"] == "openai/gpt-4o-mini"


def test_trace_logger_writes_without_toggle(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    logger.log_success_event(kwargs, response, None, None)

    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_trace_logger_uses_kwargs_context_for_log_path(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["context"] = Context(
        run_id="run_abc",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        session_id="sess_123",
        role=Role.AGENT,
    )
    logger.log_success_event(kwargs, response, None, None)

    expected = tmp_path / "run_abc" / "sessions" / "sess_123" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    assert not env_log_path.exists()


def test_trace_logger_resolves_alias_to_backend_model(tmp_path, monkeypatch) -> None:
    """Trace should log the actual backend model, not the proxy alias.

    Reproduces https://github.com/Exgentic/exgentic/issues/52
    """
    log_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(log_path))

    logger = TraceLogger()
    kwargs = {
        "model": "claude-3-5-sonnet-20241022",  # alias from CLI client
        "litellm_params": {"model": "openai/aws/claude-opus-4-5"},  # actual backend
        "messages": [{"role": "user", "content": "Hi"}],
        "response_cost": 0.0,
    }
    response = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    logger.log_success_event(kwargs, response, None, None)

    record = json.loads(log_path.read_text().strip())
    assert record["model"] == "openai/aws/claude-opus-4-5"


def test_trace_logger_get_context_falls_back_to_try_get_context(monkeypatch) -> None:
    fallback = Context(
        run_id="run_fallback",
        output_dir="/tmp/out",
        cache_dir="/tmp/cache",
        session_id="sess_fallback",
        role=Role.AGENT,
    )
    monkeypatch.setattr("exgentic.core.context.try_get_context", lambda: fallback)

    logger = TraceLogger()
    assert logger.get_context({}) == fallback
