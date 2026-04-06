# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Custom LiteLLM logger that writes token/cost usage and request/response trace to a JSONL file.

Environment variables:
    EXGENTIC_LLM_LOG_FILE: optional override of the output JSONL path (default: trace.jsonl in CWD).
    EXGENTIC_OTEL_ENABLED: enable OpenTelemetry span creation for LLM calls.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from ...utils.settings import get_settings

# Environment variable constants
FILE_ENV = "EXGENTIC_LLM_LOG_FILE"
DEFAULT_FILE = "trace.jsonl"


def _otel_enabled() -> bool:
    """Check if OTEL is enabled via settings."""
    return bool(get_settings().otel_enabled)


def _otel_record_content() -> bool:
    """Check if OTEL content recording is enabled via settings."""
    return get_settings().otel_record_content


class TraceLogger(CustomLogger):
    def __init__(self, file_path: str | None = None) -> None:
        super().__init__()
        self._file_path = file_path
        self._tracer = None
        self._otel_logger = None
        self._context = None

    @staticmethod
    def _ensure_context() -> None:
        from ...core.context import try_get_context, try_init_context

        if try_get_context() is not None:
            return

        try_init_context()

    def _init_otel(self, kwargs) -> None:
        import threading

        from ...utils.otel import get_session_logger, init_tracing_from_env

        ctx = self.get_context(kwargs)
        if ctx is None or ctx.session_id is None or ctx.otel_context is None:
            # no otel context means tracing cannot be initialized
            warnings.warn(
                f"No OTEL context found for TraceLogger, skipping OTEL initialization. context={ctx}",
                stacklevel=2,
            )
            return

        # Initialize the TracerProvider (use simple processor for subprocess).
        # Check if this is a fresh provider (subprocess) so we can add the
        # per-session file exporter for local span capture.
        from opentelemetry import trace as trace_api

        is_new_provider = type(trace_api.get_tracer_provider()).__name__ == "ProxyTracerProvider"
        self._tracer = init_tracing_from_env()

        base = Path(ctx.output_dir) / ctx.run_id

        if is_new_provider:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            from ...utils.otel import PerSessionFileExporter

            provider = trace_api.get_tracer_provider()
            if hasattr(provider, "add_span_processor"):
                provider.add_span_processor(SimpleSpanProcessor(PerSessionFileExporter(base)))

        session_root = base / "sessions" / ctx.session_id
        self._otel_logger = get_session_logger(
            session_root,
            f"{__name__} | pid={os.getpid()} tid={threading.get_native_id()}",
        )

    def _get_parent_context(self, kwargs) -> Any:
        """Reconstruct parent span context from environment variables or ContextVar.

        For proxy subprocess: reads from environment variables
        For direct callback: reads from Context ContextVar
        """
        from opentelemetry import context, trace
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        ctx = self.get_context(kwargs)

        trace_id_hex = ctx.otel_context.trace_id
        span_id_hex = ctx.otel_context.span_id

        if not trace_id_hex or not span_id_hex:
            return context.get_current()

        trace_id = int(trace_id_hex, 16)
        span_id = int(span_id_hex, 16)

        self._otel_logger.log_context_read(trace_id_hex, span_id_hex)

        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )

        parent_span = NonRecordingSpan(span_context)
        return trace.set_span_in_context(parent_span)

    def _create_llm_span(self, kwargs, name: str, start_time: Any | None = None) -> Any | None:
        """Create span for LLM call with CLIENT span kind.

        Args:
            kwargs: Keyword arguments containing context and other parameters
            name: Name of the span
            start_time: Optional datetime when the span started
        """
        from opentelemetry.trace import SpanKind

        parent_ctx = self._get_parent_context(kwargs)

        if start_time:
            # Convert datetime to nanoseconds since epoch for OTEL
            start_time_ns = int(start_time.timestamp() * 1_000_000_000)
            span = self._tracer.start_span(name, context=parent_ctx, start_time=start_time_ns, kind=SpanKind.CLIENT)
        else:
            span = self._tracer.start_span(name, context=parent_ctx, kind=SpanKind.CLIENT)

        span_ctx = span.get_span_context()

        # Extract parent span ID from context for logging
        ctx = self.get_context(kwargs)
        parent_span_id = ctx.otel_context.span_id if ctx and ctx.otel_context else None

        self._otel_logger.log_span_start(
            span_name=name,
            span_id=format(span_ctx.span_id, "016x"),
            trace_id=format(span_ctx.trace_id, "032x"),
            parent_span_id=parent_span_id,
            start_time=start_time,
        )

        return span

    def _set_attribute(self, span: Any, key: str, value: Any) -> None:
        span.set_attribute(key, value)
        span_ctx = span.get_span_context()
        self._otel_logger.log_attribute_set(key, value, format(span_ctx.span_id, "016x"))

    @staticmethod
    def _context_log_path(ctx) -> str:
        base = Path(ctx.output_dir) / ctx.run_id
        if ctx.session_id:
            return str(base / "sessions" / ctx.session_id / ctx.role.value / "litellm" / "trace.jsonl")
        return str(base / "run" / "litellm" / "trace.jsonl")

    def get_context(self, kwargs: dict[str, Any]):
        from ...core.context import Context, try_get_context

        self._ensure_context()
        context = kwargs.get("context")
        if isinstance(context, Context):
            return context
        return try_get_context()

    def _resolve_log_path(self, kwargs: dict[str, Any]) -> str:
        from ...core.context import Context

        if self._file_path:
            return self._file_path

        self._ensure_context()
        context = kwargs.get("context")
        if isinstance(context, Context):
            return self._context_log_path(context)

        file_path = os.environ.get(FILE_ENV)
        if file_path:
            return file_path

        ctx = self.get_context(kwargs)
        if ctx is not None:
            return self._context_log_path(ctx)

        return DEFAULT_FILE

    def _write_row(self, kwargs: dict[str, Any], response_obj: dict[str, Any], status: str) -> None:
        file_path = self._resolve_log_path(kwargs)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        usage = self._extract_usage(response_obj)
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": status,
            "model": kwargs.get("model"),
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

    def _write_otel(
        self,
        kwargs: dict[str, Any],
        response_obj: dict[str, Any],
        status: str,
        start_time=None,
        end_time=None,
    ) -> None:
        """Write OTEL span for LLM call following GenAI inference span conventions.

        Follows OpenTelemetry GenAI semantic conventions for inference spans:
        https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference

        Span name format: {gen_ai.operation.name} {gen_ai.request.model}
        Span kind: CLIENT
        """
        try:
            if not _otel_enabled():
                return

            # Lazy initialize OTEL if not already done
            if self._tracer is None:
                self._init_otel(kwargs)
            if self._tracer is None:
                return
            ctx = self.get_context(kwargs)

            # Lazy import GenAI semantic conventions
            from opentelemetry.trace import Status, StatusCode

            optional_params = kwargs.get("optional_params", {}) or {}
            litellm_params = kwargs.get("litellm_params", {}) or {}

            # ===== DETERMINE OPERATION TYPE =====
            # Required attribute: gen_ai.operation.name
            operation = "chat" if kwargs.get("messages") else "text_completion"

            # ===== CREATE SPAN WITH PROPER NAME =====
            # Span name format: {gen_ai.operation.name} {gen_ai.request.model}
            model = kwargs.get("model", "unknown")
            span_name = f"{operation} {model}"
            span = self._create_llm_span(kwargs, span_name, start_time=start_time)

            # ===== SET SPAN STATUS =====
            if status == "success":
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR))
                # Conditionally Required: error.type if operation ended in error
                error_info = kwargs.get("exception") or response_obj.get("error")
                if error_info:
                    if isinstance(error_info, dict):
                        error_type = error_info.get("type") or error_info.get("code") or "unknown_error"
                    elif isinstance(error_info, Exception):
                        error_type = type(error_info).__name__
                    else:
                        error_type = str(error_info)
                    self._set_attribute(span, "error.type", error_type)

            # ===== REQUIRED ATTRIBUTES =====
            # gen_ai.operation.name (Required)
            self._set_attribute(span, "gen_ai.operation.name", operation)

            # gen_ai.provider.name (Required) - maps LiteLLM provider to standard names
            provider = litellm_params.get("custom_llm_provider", "unknown")
            if provider is None:
                provider = "unknown"
            # Map common LiteLLM providers to standard GenAI provider names
            provider_mapping = {
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
            standard_provider = provider_mapping.get(provider.lower(), provider)
            self._set_attribute(span, "gen_ai.provider.name", standard_provider)

            # ===== CONDITIONALLY REQUIRED ATTRIBUTES =====
            # gen_ai.request.model (Conditionally Required if available)
            if model:
                self._set_attribute(span, "gen_ai.request.model", model)

            # gen_ai.request.choice.count (Conditionally Required if available and !=1)
            n = optional_params.get("n")
            if n is not None and n != 1:
                self._set_attribute(span, "gen_ai.request.choice.count", n)

            # gen_ai.request.seed (Conditionally Required if applicable and request includes seed)
            seed = optional_params.get("seed")
            if seed is not None:
                self._set_attribute(span, "gen_ai.request.seed", seed)

            # server.port (Conditionally Required if server.address is set)
            # Note: LiteLLM doesn't typically expose server details, skip for now

            # ===== RECOMMENDED REQUEST ATTRIBUTES =====
            self._set_attribute(span, "gen_ai.conversation.id", ctx.session_id)
            self._set_attribute(span, "exgentic.session.id", ctx.session_id)

            if optional_params.get("max_tokens") is not None:
                self._set_attribute(span, "gen_ai.request.max_tokens", optional_params["max_tokens"])

            if optional_params.get("temperature") is not None:
                self._set_attribute(span, "gen_ai.request.temperature", optional_params["temperature"])

            if optional_params.get("top_p") is not None:
                self._set_attribute(span, "gen_ai.request.top_p", optional_params["top_p"])

            if optional_params.get("top_k") is not None:
                self._set_attribute(span, "gen_ai.request.top_k", float(optional_params["top_k"]))

            if optional_params.get("frequency_penalty") is not None:
                self._set_attribute(
                    span,
                    "gen_ai.request.frequency_penalty",
                    optional_params["frequency_penalty"],
                )

            if optional_params.get("presence_penalty") is not None:
                self._set_attribute(
                    span,
                    "gen_ai.request.presence_penalty",
                    optional_params["presence_penalty"],
                )

            # gen_ai.request.stop_sequences (Recommended)
            stop = optional_params.get("stop")
            if stop is not None:
                if isinstance(stop, list):
                    self._set_attribute(span, "gen_ai.request.stop_sequences", stop)
                else:
                    self._set_attribute(span, "gen_ai.request.stop_sequences", [stop])

            # ===== RECOMMENDED RESPONSE ATTRIBUTES =====
            if response_obj:
                # Helper function to safely get attribute from dict or object
                def safe_get(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)

                # Get span context for logging
                span_ctx = span.get_span_context()
                span_id_hex = format(span_ctx.span_id, "016x")

                # gen_ai.response.id (Recommended)
                response_id = safe_get(response_obj, "id")
                if response_id:
                    self._set_attribute(span, "gen_ai.response.id", response_id)
                    if self._otel_logger:
                        self._otel_logger.log_attribute_set("gen_ai.response.id", response_id, span_id_hex)

                # gen_ai.response.model (Recommended)
                response_model = safe_get(response_obj, "model")
                if response_model:
                    self._set_attribute(span, "gen_ai.response.model", response_model)
                    if self._otel_logger:
                        self._otel_logger.log_attribute_set("gen_ai.response.model", response_model, span_id_hex)

                # gen_ai.usage.input_tokens and gen_ai.usage.output_tokens (Recommended)
                usage = safe_get(response_obj, "usage")
                if usage:
                    prompt_tokens = safe_get(usage, "prompt_tokens")
                    if prompt_tokens is not None:
                        self._set_attribute(span, "gen_ai.usage.input_tokens", prompt_tokens)
                        if self._otel_logger:
                            self._otel_logger.log_attribute_set("gen_ai.usage.input_tokens", prompt_tokens, span_id_hex)

                    completion_tokens = safe_get(usage, "completion_tokens")
                    if completion_tokens is not None:
                        self._set_attribute(span, "gen_ai.usage.output_tokens", completion_tokens)
                        if self._otel_logger:
                            self._otel_logger.log_attribute_set(
                                "gen_ai.usage.output_tokens",
                                completion_tokens,
                                span_id_hex,
                            )

                # gen_ai.response.finish_reasons (Recommended)
                choices = safe_get(response_obj, "choices", [])
                if choices:
                    finish_reasons = []
                    for choice in choices:
                        finish_reason = safe_get(choice, "finish_reason")
                        if finish_reason:
                            finish_reasons.append(finish_reason)
                    if finish_reasons:
                        self._set_attribute(span, "gen_ai.response.finish_reasons", finish_reasons)
                        if self._otel_logger:
                            self._otel_logger.log_attribute_set(
                                "gen_ai.response.finish_reasons",
                                finish_reasons,
                                span_id_hex,
                            )

            # ===== OPT-IN ATTRIBUTES (for content recording) =====
            # Note: These are opt-in and may contain sensitive data
            # Only include if explicitly enabled via settings
            if _otel_record_content():
                # gen_ai.tool.definitions (Opt-In)
                tools = kwargs.get("tools")
                if tools:
                    try:
                        self._set_attribute(
                            span,
                            "gen_ai.tool.definitions",
                            json.dumps(tools, default=str),
                        )
                    except Exception:
                        pass  # Skip if serialization fails

                # gen_ai.input.messages (Opt-In) - structured format
                messages = kwargs.get("messages")
                if messages:
                    # Convert to GenAI message format
                    try:
                        self._set_attribute(
                            span,
                            "gen_ai.input.messages",
                            json.dumps(messages, default=str),
                        )
                    except Exception:
                        pass  # Skip if serialization fails

                # gen_ai.output.messages (Opt-In) - structured format
                if response_obj:
                    # Helper function already defined above in the response attributes section
                    def safe_get(obj, key, default=None):
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return getattr(obj, key, default)

                    choices = safe_get(response_obj, "choices", [])
                    if choices:
                        output_messages = []
                        for choice in choices:
                            message = safe_get(choice, "message")
                            if message:
                                output_msg = {
                                    "role": safe_get(message, "role", "assistant"),
                                    "parts": [],
                                }
                                content = safe_get(message, "content")
                                if content:
                                    output_msg["parts"].append({"type": "text", "content": content})
                                # Include tool calls if present
                                tool_calls = safe_get(message, "tool_calls")
                                if tool_calls:
                                    for tc in tool_calls:
                                        func = safe_get(tc, "function", {})
                                        output_msg["parts"].append(
                                            {
                                                "type": "tool_call",
                                                "id": safe_get(tc, "id"),
                                                "name": safe_get(func, "name"),
                                                "arguments": safe_get(func, "arguments"),
                                            }
                                        )
                                finish_reason = safe_get(choice, "finish_reason")
                                if finish_reason:
                                    output_msg["finish_reason"] = finish_reason
                                output_messages.append(output_msg)

                        if output_messages:
                            try:
                                self._set_attribute(
                                    span,
                                    "gen_ai.output.messages",
                                    json.dumps(output_messages, default=str),
                                )
                            except Exception:
                                pass  # Skip if serialization fails

            # ===== END SPAN =====
            if end_time:
                end_time_ns = int(end_time.timestamp() * 1_000_000_000)
                span.end(end_time=end_time_ns)
            else:
                span.end()

            span_ctx = span.get_span_context()
            if self._otel_logger:
                self._otel_logger.log_span_end(
                    span_name=span_name,
                    span_id=format(span_ctx.span_id, "016x"),
                    status=status,
                    end_time=end_time,
                )
        except Exception as exc:
            warnings.warn(
                f"TraceLogger._write_otel failed: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )

    def log_success_event(self, kwargs: dict[str, Any], response_obj: dict[str, Any], start_time, end_time):
        self._write_row(kwargs, response_obj, status="success")
        self._write_otel(
            kwargs,
            response_obj,
            status="success",
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: dict[str, Any], start_time, end_time):
        self._write_row(kwargs, response_obj, status="success")
        self._write_otel(
            kwargs,
            response_obj,
            status="success",
            start_time=start_time,
            end_time=end_time,
        )

    def log_failure_event(self, kwargs: dict[str, Any], response_obj: dict[str, Any], start_time, end_time):
        self._write_row(kwargs, response_obj, status="failure")
        self._write_otel(
            kwargs,
            response_obj,
            status="failure",
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: dict[str, Any], start_time, end_time):
        self._write_row(kwargs, response_obj, status="failure")
        self._write_otel(
            kwargs,
            response_obj,
            status="failure",
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_stream_event(self, kwargs: dict[str, Any], response_obj: dict[str, Any], start_time, end_time):
        self._write_row(kwargs, response_obj, status="stream")
        self._write_otel(kwargs, response_obj, status="stream")

    # Helpers -----------------------------------------------------

    def _extract_usage(self, response_obj: Any) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        if isinstance(response_obj, dict):
            usage = response_obj.get("usage", {}) or {}
        else:
            usage_attr = getattr(response_obj, "usage", None)
            if isinstance(usage_attr, dict):
                usage = usage_attr
            elif hasattr(usage_attr, "__dict__"):
                usage = usage_attr.__dict__
        return usage

    def _capture_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key in (
            "messages",
            "prompt",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "temperature",
            "model",
        ):
            if key in kwargs:
                safe[key] = kwargs[key]
        return safe

    def _capture_response(self, response_obj: Any) -> dict[str, Any]:
        if isinstance(response_obj, dict):
            resp = dict(response_obj)
            resp.pop("usage", None)
            return resp
        out: dict[str, Any] = {}
        for attr in ("choices", "id", "object", "created", "model", "error"):
            if hasattr(response_obj, attr):
                out[attr] = getattr(response_obj, attr)
        if not out:
            out["repr"] = repr(response_obj)
        return out


# Sync logger for direct SDK calls (e.g. tool-calling agent).
class SyncTraceLogger(TraceLogger):
    def __call__(self, *args, **kwargs):
        return None


class AsyncTraceLogger(TraceLogger):
    async def __call__(self, *args, **kwargs):
        return None


# Expose module-level instances.
trace_logger = TraceLogger()
sync_trace_logger = SyncTraceLogger()
async_trace_logger = AsyncTraceLogger()
