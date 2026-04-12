# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Custom LiteLLM logger that writes token/cost usage and OTEL spans for LLM calls."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from ...utils.settings import get_settings

logger = logging.getLogger(__name__)

FILE_ENV = "EXGENTIC_LLM_LOG_FILE"
DEFAULT_FILE = "trace.jsonl"


_PROVIDER_MAP = {
    "openai": "openai",
    "azure": "azure.ai.openai",
    "anthropic": "anthropic",
    "bedrock": "aws.bedrock",
    "vertex_ai": "gcp.vertex_ai",
    "gemini": "gcp.gemini",
    "cohere": "cohere",
    "groq": "groq",
    "mistral": "mistral_ai",
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "watsonx": "ibm.watsonx.ai",
    "xai": "x_ai",
}


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _json_or_none(value: Any) -> str | None:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return None


_OPTIONAL_PARAM_ATTRS = {
    "max_tokens": "gen_ai.request.max_tokens",
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
}


def _set_json_attr(span, key: str, value: Any) -> None:
    """Set a span attribute as JSON if the value is truthy and serializable."""
    if value:
        serialized = _json_or_none(value)
        if serialized:
            span.set_attribute(key, serialized)


class TraceLogger(CustomLogger):
    def __init__(self, file_path: str | None = None) -> None:
        super().__init__()
        self._file_path = file_path
        self._tracer = None
        self._otel_logger = None

    # -- Context resolution --------------------------------------------------

    @staticmethod
    def _ensure_context() -> None:
        from ...core.context import try_get_context, try_init_context

        if try_get_context() is None:
            try_init_context()

    def get_context(self, kwargs: dict[str, Any]):
        from ...core.context import Context, try_get_context

        self._ensure_context()
        context = kwargs.get("context")
        return context if isinstance(context, Context) else try_get_context()

    @staticmethod
    def _context_log_path(ctx) -> str:
        base = Path(ctx.output_dir) / ctx.run_id
        if ctx.session_id:
            return str(base / "sessions" / ctx.session_id / ctx.role.value / "litellm" / "trace.jsonl")
        return str(base / "run" / "litellm" / "trace.jsonl")

    def _resolve_log_path(self, kwargs: dict[str, Any]) -> str:
        from ...core.context import Context

        if self._file_path:
            return self._file_path

        self._ensure_context()
        context = kwargs.get("context")
        if isinstance(context, Context):
            return self._context_log_path(context)

        if file_path := os.environ.get(FILE_ENV):
            return file_path

        ctx = self.get_context(kwargs)
        return self._context_log_path(ctx) if ctx else DEFAULT_FILE

    # -- OTEL initialization -------------------------------------------------

    def _init_otel(self, kwargs) -> None:
        import threading

        from ...utils.otel import get_session_logger, init_tracing_from_env

        ctx = self.get_context(kwargs)
        if ctx is None or ctx.session_id is None or ctx.otel_context is None:
            logger.debug("No OTEL context for TraceLogger, skipping init. context=%s", ctx)
            return

        from opentelemetry import trace as trace_api

        is_new_provider = type(trace_api.get_tracer_provider()).__name__ == "ProxyTracerProvider"
        self._tracer = init_tracing_from_env()

        if is_new_provider:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            from ...utils.otel import PerSessionFileExporter

            provider = trace_api.get_tracer_provider()
            if hasattr(provider, "add_span_processor"):
                base = Path(ctx.output_dir) / ctx.run_id
                provider.add_span_processor(SimpleSpanProcessor(PerSessionFileExporter(base)))

        session_root = Path(ctx.output_dir) / ctx.run_id / "sessions" / ctx.session_id
        self._otel_logger = get_session_logger(
            session_root,
            f"{__name__} | pid={os.getpid()} tid={threading.get_native_id()}",
        )

    def _get_parent_context(self, kwargs) -> Any:
        from opentelemetry import context, trace
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        ctx = self.get_context(kwargs)
        trace_id_hex = ctx.otel_context.trace_id
        span_id_hex = ctx.otel_context.span_id

        if not trace_id_hex or not span_id_hex:
            return context.get_current()

        self._otel_logger.log_context_read(trace_id_hex, span_id_hex)
        span_context = SpanContext(
            trace_id=int(trace_id_hex, 16),
            span_id=int(span_id_hex, 16),
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )
        return trace.set_span_in_context(NonRecordingSpan(span_context))

    # -- Event logging -------------------------------------------------------

    def _log_event(self, kwargs, response_obj, status, start_time=None, end_time=None):
        self._write_row(kwargs, response_obj, status)
        self._write_otel(kwargs, response_obj, status, start_time=start_time, end_time=end_time)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event(kwargs, response_obj, "success", start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event(kwargs, response_obj, "success", start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event(kwargs, response_obj, "failure", start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event(kwargs, response_obj, "failure", start_time, end_time)

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        self._log_event(kwargs, response_obj, "stream")

    # -- JSONL writing -------------------------------------------------------

    def _write_row(self, kwargs: dict[str, Any], response_obj: dict[str, Any], status: str) -> None:
        file_path = self._resolve_log_path(kwargs)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        usage = self._extract_usage(response_obj)
        litellm_params = kwargs.get("litellm_params") or {}
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": status,
            "model": litellm_params.get("model") or kwargs.get("model"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": kwargs.get("response_cost"),
            "request": self._capture_request(kwargs),
            "response": self._capture_response(response_obj),
            "trace_id": kwargs.get("litellm_trace_id") or kwargs.get("litellm_call_id"),
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # -- OTEL span writing ---------------------------------------------------

    def _write_otel(self, kwargs, response_obj, status, start_time=None, end_time=None) -> None:
        if not get_settings().otel_enabled:
            return
        if self._tracer is None:
            self._init_otel(kwargs)
        if self._tracer is None:
            return
        try:
            self._emit_llm_span(kwargs, response_obj, status, start_time, end_time)
        except Exception:
            logger.warning("TraceLogger._write_otel failed", exc_info=True)

    def _emit_llm_span(self, kwargs, response_obj, status, start_time, end_time) -> None:
        from opentelemetry.trace import SpanKind, Status, StatusCode

        optional_params = kwargs.get("optional_params") or {}
        litellm_params = kwargs.get("litellm_params") or {}
        ctx = self.get_context(kwargs)

        operation = "chat" if kwargs.get("messages") else "text_completion"
        model = litellm_params.get("model") or kwargs.get("model", "unknown")
        span_name = f"{operation} {model}"

        # Create span
        parent_ctx = self._get_parent_context(kwargs)
        span_kwargs: dict[str, Any] = {"context": parent_ctx, "kind": SpanKind.CLIENT}
        if start_time:
            span_kwargs["start_time"] = int(start_time.timestamp() * 1_000_000_000)
        span = self._tracer.start_span(span_name, **span_kwargs)

        span_ctx = span.get_span_context()
        span_id = format(span_ctx.span_id, "016x")
        self._otel_logger.log_span_start(
            span_name=span_name,
            span_id=span_id,
            trace_id=format(span_ctx.trace_id, "032x"),
            parent_span_id=ctx.otel_context.span_id if ctx and ctx.otel_context else None,
            start_time=start_time,
        )

        # Status
        if status == "success":
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))
            self._set_error_type(span, kwargs, response_obj)

        # Attributes
        provider = litellm_params.get("custom_llm_provider") or "unknown"
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": _PROVIDER_MAP.get(provider.lower(), provider),
            "gen_ai.request.model": model,
            "gen_ai.conversation.id": ctx.session_id,
            "exgentic.session.id": ctx.session_id,
        }
        self._collect_request_attrs(attrs, optional_params)
        self._collect_response_attrs(attrs, response_obj)

        for k, v in attrs.items():
            span.set_attribute(k, v)

        if get_settings().otel_record_content:
            self._set_content_attributes(span, kwargs, response_obj)

        # End span
        end_ns = int(end_time.timestamp() * 1_000_000_000) if end_time else None
        span.end(end_time=end_ns)

        self._otel_logger.log_span_end(span_name=span_name, span_id=span_id, status=status, end_time=end_time)

    @staticmethod
    def _set_error_type(span, kwargs, response_obj) -> None:
        error_info = kwargs.get("exception") or _safe_get(response_obj, "error")
        if not error_info:
            return
        if isinstance(error_info, dict):
            error_type = error_info.get("type") or error_info.get("code") or "unknown_error"
        elif isinstance(error_info, Exception):
            error_type = type(error_info).__name__
        else:
            error_type = str(error_info)
        span.set_attribute("error.type", error_type)

    @staticmethod
    def _collect_request_attrs(attrs: dict[str, Any], optional_params: dict) -> None:
        for param_key, attr_key in _OPTIONAL_PARAM_ATTRS.items():
            v = optional_params.get(param_key)
            if v is not None:
                attrs[attr_key] = v

        top_k = optional_params.get("top_k")
        if top_k is not None:
            attrs["gen_ai.request.top_k"] = float(top_k)

        n = optional_params.get("n")
        if n is not None and n != 1:
            attrs["gen_ai.request.choice.count"] = n

        seed = optional_params.get("seed")
        if seed is not None:
            attrs["gen_ai.request.seed"] = seed

        stop = optional_params.get("stop")
        if stop is not None:
            attrs["gen_ai.request.stop_sequences"] = stop if isinstance(stop, list) else [stop]

    @staticmethod
    def _collect_response_attrs(attrs: dict[str, Any], response_obj) -> None:
        if not response_obj:
            return

        resp_id = _safe_get(response_obj, "id")
        if resp_id:
            attrs["gen_ai.response.id"] = resp_id

        resp_model = _safe_get(response_obj, "model")
        if resp_model:
            attrs["gen_ai.response.model"] = resp_model

        usage = _safe_get(response_obj, "usage")
        if usage:
            input_tokens = _safe_get(usage, "prompt_tokens")
            if input_tokens is not None:
                attrs["gen_ai.usage.input_tokens"] = input_tokens
            output_tokens = _safe_get(usage, "completion_tokens")
            if output_tokens is not None:
                attrs["gen_ai.usage.output_tokens"] = output_tokens

        choices = _safe_get(response_obj, "choices", [])
        if choices:
            reasons = [r for c in choices if (r := _safe_get(c, "finish_reason"))]
            if reasons:
                attrs["gen_ai.response.finish_reasons"] = reasons

    @staticmethod
    def _set_content_attributes(span, kwargs: dict[str, Any], response_obj: dict[str, Any]) -> None:
        _set_json_attr(span, "gen_ai.tool.definitions", kwargs.get("tools"))
        _set_json_attr(span, "gen_ai.input.messages", kwargs.get("messages"))

        if not response_obj:
            return
        choices = _safe_get(response_obj, "choices", [])
        if not choices:
            return

        output_messages = []
        for choice in choices:
            message = _safe_get(choice, "message")
            if not message:
                continue

            output_msg: dict[str, Any] = {"role": _safe_get(message, "role", "assistant"), "parts": []}
            content = _safe_get(message, "content")
            if content:
                output_msg["parts"].append({"type": "text", "content": content})

            for tc in _safe_get(message, "tool_calls") or []:
                func = _safe_get(tc, "function", {})
                output_msg["parts"].append(
                    {
                        "type": "tool_call",
                        "id": _safe_get(tc, "id"),
                        "name": _safe_get(func, "name"),
                        "arguments": _safe_get(func, "arguments"),
                    }
                )

            finish_reason = _safe_get(choice, "finish_reason")
            if finish_reason:
                output_msg["finish_reason"] = finish_reason
            output_messages.append(output_msg)

        _set_json_attr(span, "gen_ai.output.messages", output_messages)

    # -- Extraction helpers --------------------------------------------------

    @staticmethod
    def _extract_usage(response_obj: Any) -> dict[str, Any]:
        if isinstance(response_obj, dict):
            return response_obj.get("usage") or {}
        usage_attr = getattr(response_obj, "usage", None)
        if isinstance(usage_attr, dict):
            return usage_attr
        if hasattr(usage_attr, "__dict__"):
            return usage_attr.__dict__
        return {}

    @staticmethod
    def _capture_request(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            k: kwargs[k]
            for k in (
                "messages",
                "prompt",
                "tools",
                "tool_choice",
                "functions",
                "function_call",
                "temperature",
                "model",
            )
            if k in kwargs
        }

    @staticmethod
    def _capture_response(response_obj: Any) -> dict[str, Any]:
        if isinstance(response_obj, dict):
            resp = dict(response_obj)
            resp.pop("usage", None)
            return resp
        out: dict[str, Any] = {}
        for attr in ("choices", "id", "object", "created", "model", "error"):
            if hasattr(response_obj, attr):
                out[attr] = getattr(response_obj, attr)
        return out or {"repr": repr(response_obj)}


class SyncTraceLogger(TraceLogger):
    def __call__(self, *args, **kwargs):
        return None


class AsyncTraceLogger(TraceLogger):
    async def __call__(self, *args, **kwargs):
        return None


trace_logger = TraceLogger()
sync_trace_logger = SyncTraceLogger()
async_trace_logger = AsyncTraceLogger()
