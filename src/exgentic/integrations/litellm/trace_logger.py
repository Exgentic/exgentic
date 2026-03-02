# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""
Custom LiteLLM logger that writes token/cost usage and request/response trace to a JSONL file.

Environment variables:
    EXGENTIC_LLM_LOG_FILE: optional override of the output JSONL path (default: trace.jsonl in CWD).
    EXGENTIC_OTEL_ENABLED: enable OpenTelemetry span creation for LLM calls.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

from litellm.integrations.custom_logger import CustomLogger

from ...core.context import ENV_OTEL_TRACE_ID, ENV_OTEL_SPAN_ID
from ...utils.settings import get_settings


# Environment variable constants
FILE_ENV = "EXGENTIC_LLM_LOG_FILE"
DEFAULT_FILE = "trace.jsonl"


def _otel_enabled() -> bool:
    """Check if OTEL is enabled via settings."""
    return bool(get_settings().otel_enabled)


def _otel_record_content() -> bool:
    """Check if OTEL content recording is enabled via settings."""
    return bool(get_settings().otel_record_content)


class TraceLogger(CustomLogger):
    def __init__(self, file_path: str | None = None) -> None:
        super().__init__()
        self._file_path = file_path
        self._tracer = None
        self._otel_logger = None

    @staticmethod
    def _ensure_context() -> None:
        try:
            from ...core.context import init_context_from_env, try_get_context

            if try_get_context() is not None:
                return

            init_context_from_env()
        except RuntimeError:
            pass

    def _init_otel(self) -> None:
        """Initialize OTEL tracer and logger."""
        from ...utils.otel import init_tracing_from_env, get_session_logger

        # Initialize the TracerProvider (use simple processor for subprocess)
        self._tracer = init_tracing_from_env(
            service_name="exgentic-litellm", use_simple_processor=True
        )

        # Get session logger from ContextVar (initialize from env in subprocesses).
        try:
            self._ensure_context()
            from ...core.context import try_get_context

            ctx = try_get_context()
            if ctx is not None:
                base = Path(ctx.output_dir) / ctx.run_id
                if ctx.session_id:
                    session_root = base / "sessions" / ctx.session_id
                    self._otel_logger = get_session_logger(session_root, __name__)
        except Exception:
            pass

    # Removed async __call__ - it was preventing sync callbacks from being invoked!
    # LiteLLM only calls sync log_success_event for sync completion() calls

    def _get_parent_context(self) -> Any:
        """Reconstruct parent span context from environment variables or ContextVar.

        For proxy subprocess: reads from environment variables
        For direct callback: reads from Context ContextVar
        """
        from opentelemetry import trace, context
        from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan

        trace_id_hex = None
        span_id_hex = None

        # First, try to read from environment variables (proxy subprocess case)
        trace_id_hex = os.environ.get(ENV_OTEL_TRACE_ID)
        span_id_hex = os.environ.get(ENV_OTEL_SPAN_ID)

        # If not in env vars, try to read from Context ContextVar (same process case)
        if not trace_id_hex or not span_id_hex:
            try:
                self._ensure_context()
                from ...core.context import try_get_context

                ctx = try_get_context()
                if ctx and ctx.otel_context:
                    trace_id_hex = ctx.otel_context.trace_id
                    span_id_hex = ctx.otel_context.span_id
            except Exception:
                pass

        if not trace_id_hex or not span_id_hex:
            return context.get_current()

        trace_id = int(trace_id_hex, 16)
        span_id = int(span_id_hex, 16)

        if self._otel_logger:
            self._otel_logger.log_context_read(trace_id_hex, span_id_hex)

        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )

        parent_span = NonRecordingSpan(span_context)
        return trace.set_span_in_context(parent_span)

    def _create_llm_span(
        self, name: str, start_time: Optional[Any] = None
    ) -> Optional[Any]:
        """Create span for LLM call with CLIENT span kind.

        Args:
            name: Name of the span
            start_time: Optional datetime when the span started
        """
        from opentelemetry.trace import SpanKind

        parent_ctx = self._get_parent_context()

        if start_time:
            # Convert datetime to nanoseconds since epoch for OTEL
            start_time_ns = int(start_time.timestamp() * 1_000_000_000)
            span = self._tracer.start_span(
                name, context=parent_ctx, start_time=start_time_ns, kind=SpanKind.CLIENT
            )
        else:
            span = self._tracer.start_span(
                name, context=parent_ctx, kind=SpanKind.CLIENT
            )

        span_ctx = span.get_span_context()

        if self._otel_logger:
            self._otel_logger.log_span_start(
                span_name=name,
                span_id=format(span_ctx.span_id, "016x"),
                trace_id=format(span_ctx.trace_id, "032x"),
                start_time=start_time,
            )

        return span

    @staticmethod
    def _metadata_context(kwargs: Dict[str, Any]):
        from ...core.context import Context

        metadata = kwargs.get("litellm_metadata") or kwargs.get(
            "litellm_params", {}
        ).get("litellm_metadata", {})
        context = metadata.get("context") if isinstance(metadata, dict) else None
        return context if isinstance(context, Context) else None

    @staticmethod
    def _context_log_path(ctx) -> str:
        base = Path(ctx.output_dir) / ctx.run_id
        if ctx.session_id:
            return str(
                base
                / "sessions"
                / ctx.session_id
                / ctx.role.value
                / "litellm"
                / "trace.jsonl"
            )
        return str(base / "run" / "litellm" / "trace.jsonl")

    def get_context(self, kwargs: Dict[str, Any]):
        from ...core.context import Context, try_get_context

        self._ensure_context()
        metadata_context = self._metadata_context(kwargs)
        if isinstance(metadata_context, Context):
            return metadata_context
        context = kwargs.get("context")
        if isinstance(context, Context):
            return context
        return try_get_context()

    def _resolve_log_path(self, kwargs: Dict[str, Any]) -> str:
        from ...core.context import Context

        if self._file_path:
            return self._file_path

        self._ensure_context()
        metadata_context = self._metadata_context(kwargs)
        if metadata_context is not None:
            return self._context_log_path(metadata_context)

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

    def _write_row(
        self, kwargs: Dict[str, Any], response_obj: Dict[str, Any], status: str
    ) -> None:
        file_path = self._resolve_log_path(kwargs)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        usage = self._extract_usage(response_obj)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        kwargs: Dict[str, Any],
        response_obj: Dict[str, Any],
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
        if not _otel_enabled():
            return

        # Lazy initialize OTEL if not already done
        if self._tracer is None:
            self._init_otel()

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
        span = self._create_llm_span(span_name, start_time=start_time)

        # ===== SET SPAN STATUS =====
        if status == "success":
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR))
            # Conditionally Required: error.type if operation ended in error
            error_info = kwargs.get("exception") or response_obj.get("error")
            if error_info:
                if isinstance(error_info, dict):
                    error_type = (
                        error_info.get("type")
                        or error_info.get("code")
                        or "unknown_error"
                    )
                elif isinstance(error_info, Exception):
                    error_type = type(error_info).__name__
                else:
                    error_type = str(error_info)
                span.set_attribute("error.type", error_type)

        # ===== REQUIRED ATTRIBUTES =====
        # gen_ai.operation.name (Required)
        span.set_attribute("gen_ai.operation.name", operation)

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
        span.set_attribute("gen_ai.provider.name", standard_provider)

        # ===== CONDITIONALLY REQUIRED ATTRIBUTES =====
        # gen_ai.request.model (Conditionally Required if available)
        if model:
            span.set_attribute("gen_ai.request.model", model)

        # gen_ai.request.choice.count (Conditionally Required if available and !=1)
        n = optional_params.get("n")
        if n is not None and n != 1:
            span.set_attribute("gen_ai.request.choice.count", n)

        # gen_ai.request.seed (Conditionally Required if applicable and request includes seed)
        seed = optional_params.get("seed")
        if seed is not None:
            span.set_attribute("gen_ai.request.seed", seed)

        # server.port (Conditionally Required if server.address is set)
        # Note: LiteLLM doesn't typically expose server details, skip for now

        # ===== RECOMMENDED REQUEST ATTRIBUTES =====
        # ===== RECOMMENDED REQUEST ATTRIBUTES =====
        # gen_ai.conversation.id (Recommended) - get from context if available
        try:
            from ...core.context import try_get_context

            ctx = try_get_context()
            if ctx and ctx.session_id:
                span.set_attribute("gen_ai.conversation.id", ctx.session_id)
        except Exception:
            pass

        if optional_params.get("max_tokens") is not None:
            span.set_attribute(
                "gen_ai.request.max_tokens", optional_params["max_tokens"]
            )

        if optional_params.get("temperature") is not None:
            span.set_attribute(
                "gen_ai.request.temperature", optional_params["temperature"]
            )

        if optional_params.get("top_p") is not None:
            span.set_attribute("gen_ai.request.top_p", optional_params["top_p"])

        if optional_params.get("top_k") is not None:
            span.set_attribute("gen_ai.request.top_k", float(optional_params["top_k"]))

        if optional_params.get("frequency_penalty") is not None:
            span.set_attribute(
                "gen_ai.request.frequency_penalty", optional_params["frequency_penalty"]
            )

        if optional_params.get("presence_penalty") is not None:
            span.set_attribute(
                "gen_ai.request.presence_penalty", optional_params["presence_penalty"]
            )

        # gen_ai.request.stop_sequences (Recommended)
        stop = optional_params.get("stop")
        if stop is not None:
            if isinstance(stop, list):
                span.set_attribute("gen_ai.request.stop_sequences", stop)
            else:
                span.set_attribute("gen_ai.request.stop_sequences", [stop])

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
                span.set_attribute("gen_ai.response.id", response_id)
                if self._otel_logger:
                    self._otel_logger.log_attribute_set(
                        "gen_ai.response.id", response_id, span_id_hex
                    )

            # gen_ai.response.model (Recommended)
            response_model = safe_get(response_obj, "model")
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
                if self._otel_logger:
                    self._otel_logger.log_attribute_set(
                        "gen_ai.response.model", response_model, span_id_hex
                    )

            # gen_ai.usage.input_tokens and gen_ai.usage.output_tokens (Recommended)
            usage = safe_get(response_obj, "usage")
            if usage:
                prompt_tokens = safe_get(usage, "prompt_tokens")
                if prompt_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
                    if self._otel_logger:
                        self._otel_logger.log_attribute_set(
                            "gen_ai.usage.input_tokens", prompt_tokens, span_id_hex
                        )

                completion_tokens = safe_get(usage, "completion_tokens")
                if completion_tokens is not None:
                    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
                    if self._otel_logger:
                        self._otel_logger.log_attribute_set(
                            "gen_ai.usage.output_tokens", completion_tokens, span_id_hex
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
                    span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
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
            # gen_ai.input.messages (Opt-In) - structured format
            messages = kwargs.get("messages")
            if messages:
                # Convert to GenAI message format
                try:
                    span.set_attribute(
                        "gen_ai.input.messages", json.dumps(messages, default=str)
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
                                output_msg["parts"].append(
                                    {"type": "text", "content": content}
                                )
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
                            span.set_attribute(
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

    def log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Dict[str, Any], start_time, end_time
    ):
        self._write_row(kwargs, response_obj, status="success")
        self._write_otel(
            kwargs,
            response_obj,
            status="success",
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Dict[str, Any], start_time, end_time
    ):
        self._write_row(kwargs, response_obj, status="success")
        self._write_otel(
            kwargs,
            response_obj,
            status="success",
            start_time=start_time,
            end_time=end_time,
        )

    def log_failure_event(
        self, kwargs: Dict[str, Any], response_obj: Dict[str, Any], start_time, end_time
    ):
        self._write_row(kwargs, response_obj, status="failure")
        self._write_otel(
            kwargs,
            response_obj,
            status="failure",
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_failure_event(
        self, kwargs: Dict[str, Any], response_obj: Dict[str, Any], start_time, end_time
    ):
        self._write_row(kwargs, response_obj, status="failure")
        self._write_otel(
            kwargs,
            response_obj,
            status="failure",
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_stream_event(
        self, kwargs: Dict[str, Any], response_obj: Dict[str, Any], start_time, end_time
    ):
        self._write_row(kwargs, response_obj, status="stream")
        self._write_otel(kwargs, response_obj, status="stream")

    # Helpers -----------------------------------------------------

    def _extract_usage(self, response_obj: Any) -> Dict[str, Any]:
        usage: Dict[str, Any] = {}
        if isinstance(response_obj, dict):
            usage = response_obj.get("usage", {}) or {}
        else:
            usage_attr = getattr(response_obj, "usage", None)
            if isinstance(usage_attr, dict):
                usage = usage_attr
            elif hasattr(usage_attr, "__dict__"):
                usage = usage_attr.__dict__
        return usage

    def _capture_request(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
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

    def _capture_response(self, response_obj: Any) -> Dict[str, Any]:
        if isinstance(response_obj, dict):
            resp = dict(response_obj)
            resp.pop("usage", None)
            return resp
        out: Dict[str, Any] = {}
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
