# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import asyncio
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


def test_trace_logger_reads_context_from_metadata(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["litellm_metadata"] = {
        "context": Context(
            run_id="run_meta",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            session_id="sess_meta",
            role=Role.AGENT,
        )
    }
    logger.log_success_event(kwargs, response, None, None)

    expected = tmp_path / "run_meta" / "sessions" / "sess_meta" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    assert not env_log_path.exists()


def test_trace_logger_reads_context_from_nested_metadata(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["litellm_params"] = {
        "litellm_metadata": {
            "context": Context(
                run_id="run_nested",
                output_dir=str(tmp_path),
                cache_dir=str(tmp_path / "cache"),
                session_id="sess_nested",
                role=Role.AGENT,
            )
        }
    }
    logger.log_success_event(kwargs, response, None, None)

    expected = tmp_path / "run_nested" / "sessions" / "sess_nested" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    assert not env_log_path.exists()


def test_trace_logger_async_writes_with_kwargs_context(tmp_path, monkeypatch) -> None:
    env_log_path = tmp_path / "env_trace.jsonl"
    monkeypatch.setenv(FILE_ENV, str(env_log_path))

    logger = TraceLogger()
    kwargs, response = _sample_payload()
    kwargs["litellm_metadata"] = {
        "context": Context(
            run_id="run_async",
            output_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
            session_id="sess_async",
            role=Role.AGENT,
        )
    }

    asyncio.run(logger.async_log_success_event(kwargs, response, None, None))

    expected = tmp_path / "run_async" / "sessions" / "sess_async" / "agent" / "litellm" / "trace.jsonl"
    assert expected.exists()
    lines = expected.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["status"] == "success"
    assert not env_log_path.exists()


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
