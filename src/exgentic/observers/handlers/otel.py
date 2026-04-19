# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from opentelemetry import context, trace
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import AttributeValue

from ...core.context import OtelContext, get_context, set_context
from ...core.orchestrator.observer import Observer
from ...interfaces.registry import get_agent_entries, get_benchmark_entries
from ...utils.otel import (
    PerSessionFileExporter,
    flush_traces,
    get_session_logger,
    init_tracing_from_env,
    to_otel_attribute_value,
)
from ...utils.settings import get_settings

logger = logging.getLogger(__name__)

tracer = init_tracing_from_env()


# ---------------------------------------------------------------------------
# Session span manager
# ---------------------------------------------------------------------------


def _serialize_to_json(obj: Any) -> str:
    """Serialize to JSON, handling both Pydantic models and plain dicts."""
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json()
    return json.dumps(obj, default=str)


def _span_id_hex(span_ctx) -> str:
    return format(span_ctx.span_id, "016x")


def _trace_id_hex(span_ctx) -> str:
    return format(span_ctx.trace_id, "032x")


class SessionSpanManager:
    """Manages an isolated span hierarchy for a single session."""

    def __init__(self, session_id: str, session_root: Path, tracer: Tracer = tracer):
        self.session_id = session_id
        self._tracer = tracer
        self._span_stack: list[Span] = []
        self._heritable_attributes: dict[str, AttributeValue] = {}
        self._logger = get_session_logger(
            session_root,
            f"{__name__} | pid={os.getpid()} tid={threading.get_native_id()}",
        )
        self._logger.info(f"SessionSpanManager initialized for session {self.session_id}")

    @property
    def depth(self) -> int:
        return len(self._span_stack)

    @property
    def current_span(self) -> Span | None:
        return self._span_stack[-1] if self._span_stack else None

    # -- Span lifecycle ------------------------------------------------------

    def start_span(self, name: str, **kwargs) -> Span:
        parent = self._span_stack[-1] if self._span_stack else None
        ctx = trace.set_span_in_context(parent) if parent else context.get_current()

        span = cast(Span, self._tracer.start_span(name, context=ctx, **kwargs))
        self._span_stack.append(span)

        for k, v in self._heritable_attributes.items():
            span.set_attribute(k, v)

        span_ctx = span.get_span_context()
        self._logger.log_span_start(
            span_name=name,
            span_id=_span_id_hex(span_ctx),
            trace_id=_trace_id_hex(span_ctx),
            parent_span_id=_span_id_hex(parent.get_span_context()) if parent else None,
            is_root=(self.depth == 1),
            depth=self.depth,
            start_time=datetime.fromtimestamp(span.start_time / 1_000_000_000),
        )
        return span

    def end_current_span(self) -> None:
        if not self._span_stack:
            self._logger.warning("Attempted to end span with empty stack")
            return

        span = self._span_stack.pop()
        span_ctx = span.get_span_context()
        span_name = getattr(span, "_name", "unknown")
        span.end()

        self._logger.log_span_end(
            span_name=span_name,
            span_id=_span_id_hex(span_ctx),
            is_root=(self.depth == 0),
            depth=self.depth,
            end_time=datetime.fromtimestamp(span.end_time / 1_000_000_000),
        )

    def get_otel_context(self) -> OtelContext | None:
        if not self.current_span:
            return None
        span_ctx = self.current_span.get_span_context()
        return OtelContext(
            trace_id=_trace_id_hex(span_ctx),
            span_id=_span_id_hex(span_ctx),
        )

    def update_tracing_context(self) -> None:
        otel_context = self.get_otel_context()
        ctx = get_context()
        set_context(ctx.with_otel_context(otel_context))
        self._logger.log_context_update(
            otel_context.trace_id if otel_context else None,
            otel_context.span_id if otel_context else None,
            operation="write",
        )

    # -- Attributes ----------------------------------------------------------

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        if self.current_span is None:
            raise AttributeError("No current span")
        self.current_span.set_attribute(key, value)
        self._logger.log_attribute_set(key, value, _span_id_hex(self.current_span.get_span_context()))

    def set_attributes(self, attributes: dict[str, AttributeValue] | None = None, **kwargs) -> None:
        for k, v in {**(attributes or {}), **kwargs}.items():
            self.set_attribute(k, v)

    def set_heritable_attribute(self, key: str, value: AttributeValue) -> None:
        self._heritable_attributes[key] = value
        if self.current_span:
            self.set_attribute(key, value)

    def set_heritable_attributes(self, attributes: dict[str, AttributeValue] | None = None, **kwargs) -> None:
        merged = {**(attributes or {}), **kwargs}
        self._heritable_attributes.update(merged)
        if self.current_span:
            self.set_attributes(merged)

    def record_exception(self, exc: Exception, set_error_status: bool = True) -> None:
        if self.current_span:
            self.current_span.record_exception(exc)
            if set_error_status:
                self.current_span.set_status(Status(StatusCode.ERROR))
        self._logger.log_exception(exc)

    def update_current_span_name(self, new_name: str) -> None:
        if self.current_span:
            old_name = getattr(self.current_span, "_name", "unknown")
            self.current_span.update_name(new_name)
            self._logger.log_span_rename(old_name, new_name, _span_id_hex(self.current_span.get_span_context()))


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class OtelTracingObserver(Observer):
    """OpenTelemetry tracing observer that creates a trace per session."""

    def __init__(self):
        super().__init__()
        self._run_attributes: dict[str, AttributeValue] = {}
        self._span_managers: dict[str, SessionSpanManager] = {}
        self._session_step_counters: dict[str, int] = {}
        self._session_agents: dict[str, Any] = {}

    def _get_span_manager(self, session_id: str) -> SessionSpanManager:
        return self._span_managers[session_id]

    # -- Run lifecycle -------------------------------------------------------

    def on_run_start(self, run_config) -> None:
        bench_entry = get_benchmark_entries().get(run_config.benchmark)
        agent_entry = get_agent_entries().get(run_config.agent)
        model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")

        from ...utils.paths import get_run_paths

        run_paths = get_run_paths()
        provider = trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider.add_span_processor(SimpleSpanProcessor(PerSessionFileExporter(run_paths.root, run_paths.run_id)))

        self._run_attributes = {
            "exgentic.benchmark.slug_name": bench_entry.slug_name if bench_entry else run_config.benchmark,
            "exgentic.benchmark.subset": run_config.subset or "",
            "exgentic.benchmark.agent.name": agent_entry.slug_name if agent_entry else run_config.agent,
            "exgentic.agent.slug": run_config.agent,
            "exgentic.run.id": get_run_paths().run_id,
        }
        if model_name:
            self._run_attributes["gen_ai.request.model"] = model_name

    # -- Session lifecycle ---------------------------------------------------

    def on_session_enter(self, session_id: str, task_id: str | None) -> None:
        span_manager = SessionSpanManager(session_id, self.paths.session(session_id).root)
        self._span_managers[session_id] = span_manager
        self._session_step_counters[session_id] = 0

        bench_name = self._run_attributes.get("exgentic.benchmark.slug_name", "unknown_benchmark")
        subset = self._run_attributes.get("exgentic.benchmark.subset", "subset")
        span_manager.start_span(f"{bench_name} {subset} session")
        span_manager.update_tracing_context()

        span_manager.set_heritable_attributes(self._run_attributes)
        span_manager.set_heritable_attribute("gen_ai.conversation.id", session_id)
        span_manager.set_heritable_attribute("exgentic.session.id", session_id)
        if task_id is not None:
            span_manager.set_attribute("exgentic.session.task_id", task_id)

    def on_session_creation(self, session) -> None:
        span_manager = self._span_managers.get(session.session_id)
        if span_manager is None:
            import warnings

            warnings.warn(
                f"on_session_enter was not called before on_session_creation "
                f"for session {session.session_id}. OTEL trace propagation "
                f"to child services may be incomplete.",
                stacklevel=2,
            )
            self.on_session_enter(session.session_id, session.task_id)
            span_manager = self._span_managers[session.session_id]

        if get_settings().otel_record_content:
            task = getattr(session, "task", None)
            if task is not None:
                span_manager.set_attribute("exgentic.session.task", task)

        for action in session.actions:
            prefix = f"exgentic.session.action.{action.name}"
            span_manager.set_attributes(
                {
                    f"{prefix}.name": action.name,
                    f"{prefix}.description": action.description,
                    f"{prefix}.is_message": action.is_message,
                    f"{prefix}.is_finish": action.is_finish,
                }
            )
        tools_list = [
            {
                "name": action.name,
                "description": action.description,
                "is_message": action.is_message,
                "is_finish": action.is_finish,
            }
            for action in session.actions
        ]
        span_manager.set_attribute("exgentic.session.tools", _serialize_to_json(tools_list))

        for k, v in session.context.items():
            otel_value = to_otel_attribute_value(v)
            if otel_value is not None:
                span_manager.set_attribute(f"exgentic.context.{k}", otel_value)

    def on_session_start(self, session, agent, observation) -> None:
        self._session_agents[session.session_id] = agent
        span_manager = self._span_managers[session.session_id]

        span_manager.set_attribute("exgentic.session.agent.id", agent.agent_id)
        agent_path = to_otel_attribute_value(agent.paths.agent_dir)
        if agent_path is not None:
            span_manager.set_attribute("exgentic.session.agent.path", agent_path)

        # Record initial observation as an execute_tool span
        span_manager.start_span("execute_tool initial_observation", kind=SpanKind.CLIENT)
        span_manager.current_span.set_attribute("gen_ai.operation.name", "execute_tool")
        span_manager.current_span.set_attribute("gen_ai.tool.name", "initial_observation")
        span_manager.current_span.set_attribute("gen_ai.tool.description", "Initial observation from benchmark")
        self._record_observation(session.session_id, observation)
        span_manager.end_current_span()

        self._session_step_counters[session.session_id] += 1

    def on_react_success(self, session, action) -> None:
        span_manager = self._get_span_manager(session.session_id)
        action_list = action.to_action_list() if action else []
        tool_name = action_list[0].name if action_list and action_list[0] else "unknown"

        span_manager.start_span(f"execute_tool {tool_name}", kind=SpanKind.CLIENT)
        span_manager.current_span.set_attribute("gen_ai.operation.name", "execute_tool")
        span_manager.current_span.set_attribute("gen_ai.tool.name", tool_name)

        if action_list:
            first = action_list[0]
            span_manager.current_span.set_attribute("gen_ai.tool.id", first.id)

            desc = self._get_action_description(session, tool_name)
            if desc:
                span_manager.current_span.set_attribute("gen_ai.tool.description", desc)

            if get_settings().otel_record_content:
                try:
                    params_json = _serialize_to_json(first.arguments)
                    span_manager.current_span.set_attribute("gen_ai.tool.parameters", params_json)
                except Exception:
                    logger.debug("Failed to serialize tool parameters for %s", tool_name, exc_info=True)

    def on_react_error(self, session, error) -> None:
        return None

    def on_step_success(self, session, observation) -> None:
        span_manager = self._get_span_manager(session.session_id)
        self._record_observation(session.session_id, observation)
        span_manager.end_current_span()
        self._session_step_counters[session.session_id] += 1

    def on_step_error(self, session, error) -> None:
        self._get_span_manager(session.session_id).record_exception(error)

    def on_session_success(self, session, score, agent) -> None:
        span_manager = self._get_span_manager(session.session_id)
        self._close_trailing_tool_span(span_manager)

        span_manager.set_attributes(
            {
                "exgentic.score.success": score.success,
                "exgentic.score": score.score,
                "exgentic.score.is_finished": score.is_finished,
                "exgentic.session.steps": self._session_step_counters[session.session_id],
                "exgentic.session.task_id": session.task_id,
            }
        )
        for prefix, data in (
            ("exgentic.score.metrics", score.session_metrics),
            ("exgentic.score.metadata", score.session_metadata),
        ):
            for key, value in data.items():
                otel_value = to_otel_attribute_value(value)
                if otel_value is not None:
                    span_manager.set_attribute(f"{prefix}.{key}", otel_value)

        self._set_cost_attr(span_manager, "exgentic.agent.agent_cost", lambda: agent.get_cost())
        self._set_cost_attr(span_manager, "exgentic.session.cost", lambda: session.get_cost())

        self._finalize_session(session.session_id, span_manager)

    def on_session_error(self, session, error) -> None:
        span_manager = self._get_span_manager(session.session_id)
        self._close_trailing_tool_span(span_manager)
        span_manager.record_exception(error)

        self._set_cost_attr(span_manager, "exgentic.session.cost", lambda: session.get_cost())
        span_manager.set_attribute("exgentic.session.task_id", session.task_id)

        self._finalize_session(session.session_id, span_manager)

    # -- Helpers -------------------------------------------------------------

    def _get_action_description(self, session, action_name: str) -> str | None:
        for action_type in session.actions:
            if action_type.name == action_name:
                return action_type.description
        return None

    def _record_observation(self, session_id: str, observation) -> None:
        if not get_settings().otel_record_content:
            return
        span_manager = self._get_span_manager(session_id)
        observation_list = observation.to_observation_list() if observation is not None else []
        otel_value = to_otel_attribute_value(observation_list)
        if otel_value is not None:
            span_manager.current_span.set_attribute("gen_ai.tool.result", otel_value)

    @staticmethod
    def _close_trailing_tool_span(span_manager: SessionSpanManager) -> None:
        if span_manager.depth == 2:
            span_manager.end_current_span()

    @staticmethod
    def _set_cost_attr(span_manager: SessionSpanManager, key: str, get_cost) -> None:
        try:
            span_manager.set_attribute(key, json.dumps(get_cost(), default=str))
        except Exception:
            logger.debug("Failed to serialize cost for %s", key, exc_info=True)

    def _finalize_session(self, session_id: str, span_manager: SessionSpanManager) -> None:
        span_manager.end_current_span()
        flush_traces()
        self._sort_session_spans(session_id)

        del self._span_managers[session_id]
        del self._session_step_counters[session_id]
        self._session_agents.pop(session_id, None)

    def _sort_session_spans(self, session_id: str) -> None:
        """Sort otel_spans.jsonl by start_time for consistent ordering."""
        path = self.paths.session(session_id).root / "otel_spans.jsonl"
        if not path.exists():
            return
        try:
            with open(path) as f:
                spans = [json.loads(line) for line in f if line.strip()]
            spans.sort(key=lambda s: s.get("start_time", ""))
            with open(path, "w") as f:
                for span in spans:
                    f.write(json.dumps(span, separators=(",", ":")) + "\n")
        except Exception:
            logger.warning("Failed to sort session spans for %s", session_id, exc_info=True)
