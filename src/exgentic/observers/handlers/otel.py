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


class SessionSpanManager:
    """Manages isolated span hierarchy for a single session."""

    def __init__(self, session_id: str, session_root: Path, tracer: Tracer = tracer):
        self.session_id = session_id
        self._tracer = tracer
        self._span_stack: list[Span] = []
        self._heritable_attributes: dict[str, AttributeValue] = {}

        # Initialize session logger
        self._logger = get_session_logger(
            session_root,
            f"{__name__} | pid={os.getpid()} tid={threading.get_native_id()}",
        )
        self._logger.info(f"SessionSpanManager initialized for session {self.session_id}")

    def start_span(self, name: str, **kwargs) -> Span:
        """Start a new span as a child of the current span.

        Args:
            name: Name of the span
            **kwargs: Additional arguments passed to tracer.start_span
        """
        parent_span = self._span_stack[-1] if self._span_stack else None
        ctx = trace.set_span_in_context(parent_span) if parent_span else context.get_current()

        span = cast(Span, self._tracer.start_span(name, context=ctx, **kwargs))
        self._span_stack.append(span)

        # Apply heritable attributes to new span
        for k, v in self._heritable_attributes.items():
            span.set_attribute(k, v)

        # Log span start
        span_ctx = span.get_span_context()
        parent_span_id = format(parent_span.get_span_context().span_id, "016x") if parent_span else None
        start_time = datetime.fromtimestamp(span.start_time / 1_000_000_000)

        self._logger.log_span_start(
            span_name=name,
            span_id=format(span_ctx.span_id, "016x"),
            trace_id=format(span_ctx.trace_id, "032x"),
            parent_span_id=parent_span_id,
            is_root=(len(self._span_stack) == 1),
            depth=len(self._span_stack),
            start_time=start_time,
        )
        return span

    def end_current_span(self) -> None:
        """End the current span and set parent as current."""
        if not self._span_stack:
            self._logger.warning("Attempted to end span with empty stack")
            return

        span = self._span_stack.pop()
        span_ctx = span.get_span_context()
        span_name = getattr(span, "_name", "unknown")  # Try to get span name

        span.end()
        end_time = datetime.fromtimestamp(span.end_time / 1_000_000_000)

        self._logger.log_span_end(
            span_name=span_name,
            span_id=format(span_ctx.span_id, "016x"),
            is_root=(len(self._span_stack) == 0),
            depth=len(self._span_stack),
            end_time=end_time,
        )

    @property
    def current_span(self) -> Span | None:
        return self._span_stack[-1] if self._span_stack else None

    def get_otel_context(self) -> OtelContext | None:
        """Export current span context for subprocess transport.

        Returns:
            OtelContext with trace_id and span_id as hex strings, or None if no current span.
        """
        if not self.current_span:
            return None

        span_ctx = self.current_span.get_span_context()
        return OtelContext(
            trace_id=format(span_ctx.trace_id, "032x"),
            span_id=format(span_ctx.span_id, "016x"),
        )

    def update_tracing_context(self) -> None:
        """Update the global Context with current OTEL span context."""
        otel_context = self.get_otel_context()
        ctx = get_context()
        new_ctx = ctx.with_otel_context(otel_context)
        set_context(new_ctx)
        self._logger.log_context_update(
            otel_context.trace_id if otel_context is not None else None,
            otel_context.span_id if otel_context is not None else None,
            operation="write",
        )

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        if self.current_span is None:
            raise (AttributeError("No current span"))
        self.current_span.set_attribute(key, value)
        span_ctx = self.current_span.get_span_context()
        self._logger.log_attribute_set(key, value, format(span_ctx.span_id, "016x"))

    def set_attributes(self, attributes: dict[str, AttributeValue] | None = None, **kwargs) -> None:
        if attributes is None:
            attributes = {}
        attributes.update(kwargs)
        for k, v in attributes.items():
            self.set_attribute(k, v)

    def set_heritable_attribute(self, key: str, value: AttributeValue) -> None:
        self._heritable_attributes[key] = value
        if self.current_span:
            self.set_attribute(key, value)

    def set_heritable_attributes(self, attributes: dict[str, AttributeValue] | None = None, **kwargs) -> None:
        if attributes:
            self._heritable_attributes.update(attributes)
        self._heritable_attributes.update(kwargs)
        if self.current_span:
            self.set_attributes(attributes, **kwargs)

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
            span_ctx = self.current_span.get_span_context()
            self._logger.log_span_rename(old_name, new_name, format(span_ctx.span_id, "016x"))


# OBSERVER
class OtelTracingObserver(Observer):
    """OpenTelemetry tracing observer for sessions.

    This observer creates a complete trace for each session, with spans for:
    - Session (root span)
    - Steps (child spans)
    - Actions and observations (nested child spans)

    Each session gets its own SessionSpanContext instance, ensuring complete
    isolation between concurrent sessions
    """

    def __init__(self):
        super().__init__()
        self._run_attributes: dict[str, AttributeValue] = {}
        self._span_managers: dict[str, SessionSpanManager] = {}
        self._session_step_counters: dict[str, int] = {}
        self._session_agents: dict[str, Any] = {}  # Store agent instances by session_id
        self._session_actions: dict[str, list] = {}  # Store session actions for tool definitions

    def _get_span_manager(self, session_id: str) -> SessionSpanManager:
        return self._span_managers[session_id]

    def _get_action_description(self, session, action_name: str) -> str | None:
        """Look up action description from session.actions by name."""
        for action_type in session.actions:
            if action_type.name == action_name:
                return action_type.description
        return None

    @staticmethod
    def _serialize_to_json(obj: Any) -> str:
        """Serialize an object to JSON, handling both Pydantic models and plain dicts."""
        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json()
        return json.dumps(obj, default=str)

    def _get_tool_definitions(self, session_id: str) -> str:
        """Generate gen_ai.tool.definitions JSON from session actions."""
        actions = self._session_actions.get(session_id, [])
        tool_definitions = []

        for action_type in actions:
            tool_def = {
                "type": "function",
                "function": {
                    "name": action_type.name,
                    "description": action_type.description,
                },
            }

            # Add parameters schema if available
            try:
                schema = action_type.arguments.model_json_schema()
                # Convert to OpenAI function calling format
                tool_def["function"]["parameters"] = {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                }
            except Exception:
                pass

            tool_definitions.append(tool_def)

        try:
            return json.dumps(tool_definitions)
        except Exception:
            return "[]"

    def on_run_start(self, run_config) -> None:
        bench_entry = get_benchmark_entries().get(run_config.benchmark)
        agent_entry = get_agent_entries().get(run_config.agent)

        # Extract model name from run_config
        model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")

        from ...utils.paths import get_run_paths

        # Write per-session otel_spans.jsonl files automatically.
        run_root = get_run_paths().root
        provider = trace.get_tracer_provider()
        if hasattr(provider, "add_span_processor"):
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider.add_span_processor(SimpleSpanProcessor(PerSessionFileExporter(run_root)))

        self._run_attributes = {
            "exgentic.benchmark.slug_name": bench_entry.slug_name if bench_entry is not None else run_config.benchmark,
            "exgentic.benchmark.subset": run_config.subset or "",
            "exgentic.benchmark.agent.name": agent_entry.slug_name if agent_entry is not None else run_config.agent,
            "exgentic.agent.slug": run_config.agent,
            "exgentic.run.id": get_run_paths().run_id,
        }

        # Store model name as heritable attribute
        if model_name:
            self._run_attributes["gen_ai.request.model"] = model_name

    def on_session_enter(self, session_id: str, task_id: str | None) -> None:
        """Start the root session span + publish OTEL context.

        This fires before benchmark/agent services are spawned, so the
        trace_id and span_id land in their per-service runtime.json
        files and their LLM calls can attach to the session trace.
        """
        span_manager = SessionSpanManager(session_id, self.paths.session(session_id).root)
        self._span_managers[session_id] = span_manager
        self._session_step_counters[session_id] = 0

        # Start root session span
        bench_name = self._run_attributes.get("exgentic.benchmark.slug_name", "unknown_benchmark")
        subset = self._run_attributes.get("exgentic.benchmark.subset", "subset")
        span_manager.start_span(f"{bench_name} {subset} session")
        span_manager.update_tracing_context()  # pass otel context to trace_logger

        span_manager.set_heritable_attributes(self._run_attributes)

        # gen_ai.conversation.id is the primary correlation attribute (heritable)
        span_manager.set_heritable_attribute("gen_ai.conversation.id", session_id)
        # Also keep exgentic.session.id for backwards compatibility
        span_manager.set_heritable_attribute("exgentic.session.id", session_id)
        if task_id is not None:
            span_manager.set_attribute("exgentic.session.task_id", task_id)

    def on_session_creation(self, session) -> None:
        """Add session-dependent attributes to the already-started span."""
        span_manager = self._span_managers.get(session.session_id)
        if span_manager is None:
            # on_session_enter wasn't called first — start the span now
            # for backward compatibility (tests and observers that only
            # hook on_session_creation).  Note: child services spawned
            # before this point will NOT have the OTEL trace_id in their
            # runtime.json since the span didn't exist yet.
            import warnings

            warnings.warn(
                f"on_session_enter was not called before on_session_creation "
                f"for session {session.session_id}. OTEL trace propagation "
                f"to child services may be incomplete.",
                stacklevel=2,
            )
            self.on_session_enter(session.session_id, session.task_id)
            span_manager = self._span_managers[session.session_id]
        self._session_actions[session.session_id] = session.actions  # Store actions for tool definitions

        # Only record task content if otel_record_content is enabled
        if get_settings().otel_record_content:
            task = getattr(session, "task", None)
            if task is not None:
                span_manager.set_attribute(
                    "exgentic.session.task",
                    task,
                )

        for action in session.actions:
            span_manager.set_attribute(f"exgentic.session.action.{action.name}.name", action.name)
            span_manager.set_attribute(f"exgentic.session.action.{action.name}.description", action.description)
            span_manager.set_attribute(f"exgentic.session.action.{action.name}.is_message", action.is_message)
            span_manager.set_attribute(f"exgentic.session.action.{action.name}.is_finish", action.is_finish)
        for k, v in session.context.items():
            otel_value = to_otel_attribute_value(v)
            if otel_value is not None:
                span_manager.set_attribute(f"exgentic.context.{k}", otel_value)

    def on_session_start(self, session, agent, observation) -> None:
        self._session_agents[session.session_id] = agent  # Store agent instance
        span_manager = self._span_managers[session.session_id]

        span_manager.set_attribute("exgentic.session.agent.id", agent.agent_id)
        agent_path_otel = to_otel_attribute_value(agent.paths.agent_dir)
        if agent_path_otel is not None:
            span_manager.set_attribute("exgentic.session.agent.path", agent_path_otel)

        # Record initial observation as execute_tool span
        span_manager.start_span("execute_tool initial_observation", kind=SpanKind.CLIENT)
        span_manager.current_span.set_attribute("gen_ai.operation.name", "execute_tool")
        span_manager.current_span.set_attribute("gen_ai.tool.name", "initial_observation")
        span_manager.current_span.set_attribute("gen_ai.tool.description", "Initial observation from benchmark")

        self._record_observation(session.session_id, observation)
        span_manager.end_current_span()  # end initial observation span

        # Increment step counter (no invoke_agent span created)
        self._session_step_counters[session.session_id] += 1

    def _record_observation(self, session_id: str, observation) -> None:
        """Record observation details on the current span."""
        span_manager = self._get_span_manager(session_id)
        observation_list = observation.to_observation_list() if observation is not None else []

        # Only record observation content if otel_record_content is enabled
        if get_settings().otel_record_content:
            observation_otel = to_otel_attribute_value(observation_list)
            if observation_otel is not None:
                span_manager.current_span.set_attribute("gen_ai.tool.result", observation_otel)

    def on_react_success(self, session, action) -> None:
        span_manager = self._get_span_manager(session.session_id)

        # Create execute_tool span with semantic conventions
        action_list = action.to_action_list() if action else []
        tool_name = action_list[0].name if action_list and action_list[0] else "unknown"
        span_manager.start_span(f"execute_tool {tool_name}", kind=SpanKind.CLIENT)

        # Set required semantic convention attributes
        span_manager.current_span.set_attribute("gen_ai.operation.name", "execute_tool")
        span_manager.current_span.set_attribute("gen_ai.tool.name", tool_name)

        # Set recommended attributes
        if action_list:
            first_action = action_list[0]
            span_manager.current_span.set_attribute("gen_ai.tool.id", first_action.id)

            # Get tool description from session.actions
            tool_desc = self._get_action_description(session, tool_name)
            if tool_desc:
                span_manager.current_span.set_attribute("gen_ai.tool.description", tool_desc)

            # Set tool parameters as JSON
            if get_settings().otel_record_content:
                try:
                    params_json = self._serialize_to_json(first_action.arguments)
                    span_manager.current_span.set_attribute("gen_ai.tool.parameters", params_json)
                except Exception:
                    logger.warning(
                        "Failed to serialize tool parameters for %s",
                        tool_name,
                        exc_info=True,
                    )

    def on_react_error(self, session, error) -> None:
        return None

    def on_step_success(self, session, observation) -> None:
        span_manager = self._get_span_manager(session.session_id)
        self._record_observation(session.session_id, observation)
        span_manager.end_current_span()  # end execute_tool span

        self._session_step_counters[session.session_id] += 1

    def on_step_error(self, session, error) -> None:
        span_manager = self._get_span_manager(session.session_id)
        span_manager.record_exception(error)

    def on_session_success(self, session, score, agent) -> None:
        span_manager = self._get_span_manager(session.session_id)

        # Certain session conditions may lead to a trailing execute_tool span
        if len(span_manager._span_stack) == 2:
            span_manager.end_current_span()  # end execute_tool span

        # Add final session attributes (with exgentic. prefix)
        span_manager.set_attribute("exgentic.score.success", score.success)
        span_manager.set_attribute("exgentic.score", score.score)
        span_manager.set_attribute("exgentic.score.is_finished", score.is_finished)
        span_manager.set_attribute("exgentic.session.steps", self._session_step_counters[session.session_id])

        # Convert cost objects to JSON strings for OTEL compatibility
        try:
            agent_cost = agent.get_cost()
            span_manager.set_attribute("exgentic.agent.agent_cost", json.dumps(agent_cost, default=str))
        except Exception:
            pass

        try:
            session_cost = session.get_cost()
            span_manager.set_attribute("exgentic.session.cost", json.dumps(session_cost, default=str))
        except Exception:
            pass

        span_manager.set_attribute("exgentic.session.task_id", session.task_id)

        # Close session span
        span_manager.end_current_span()

        # Flush traces to ensure they are exported
        flush_traces()
        self._sort_session_spans(session.session_id)

        # Clean up
        del self._span_managers[session.session_id]
        del self._session_step_counters[session.session_id]
        del self._session_agents[session.session_id]
        del self._session_actions[session.session_id]

    def on_session_error(self, session, error) -> None:
        span_manager = self._get_span_manager(session.session_id)

        # Certain session conditions may lead to a trailing execute_tool span
        if len(span_manager._span_stack) == 2:
            span_manager.end_current_span()  # end execute_tool span

        # Record error on session span
        span_manager.record_exception(error)

        # Convert cost object to JSON string for OTEL compatibility
        try:
            session_cost = session.get_cost()
            span_manager.set_attribute("exgentic.session.cost", json.dumps(session_cost, default=str))
        except Exception:
            pass

        span_manager.set_attribute("exgentic.session.task_id", session.task_id)

        span_manager.end_current_span()  # Close session span

        # Flush traces to ensure they are exported
        flush_traces()
        self._sort_session_spans(session.session_id)

        # Clean up
        del self._span_managers[session.session_id]
        del self._session_step_counters[session.session_id]
        if session.session_id in self._session_agents:
            del self._session_agents[session.session_id]
        if session.session_id in self._session_actions:
            del self._session_actions[session.session_id]

    def _sort_session_spans(self, session_id: str) -> None:
        """Sort otel_spans.jsonl by start_time so output matches Jaeger ordering."""
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
            pass  # Non-critical — unsorted file is still valid


# Made with Bob
