# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A Agent Executor for Exgentic agents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)


class A2AEventEmitter:
    """Handle events for A2A Agent."""

    def __init__(self, task_updater: Any):
        self.task_updater = task_updater

    async def emit_event(self, message: str, final: bool = False, failed: bool = False) -> None:
        """Emit an event to update task status."""
        logger.info("Emitting event: %s", message)

        if final or failed:
            parts = [TextPart(text=message)]
            await self.task_updater.add_artifact(parts)
            if final:
                await self.task_updater.complete()
            if failed:
                await self.task_updater.failed()
        else:
            try:
                await self.task_updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        message,
                        self.task_updater.context_id,
                        self.task_updater.task_id,
                    ),
                )
            except RuntimeError as e:
                # Task may already be in terminal state if fire-and-forget
                # status updates race with completion. This is expected.
                error_msg = str(e).lower()
                if "terminal state" in error_msg or "already" in error_msg:
                    logging.debug(
                        "Task update skipped - task already in terminal state (expected race condition): %s",
                        e,
                    )
                else:
                    logging.warning(
                        "Unexpected RuntimeError during task status update: %s",
                        e,
                        exc_info=True,
                    )


class ExgenticAgentExecutor:
    """Execute tasks using Exgentic agents with MCP tools."""

    def __init__(
        self,
        agent_cls: type,
        agent_kwargs: dict[str, Any],
        mcp_address: str,
        agent_display_name: str,
        tool_metadata: list[dict[str, Any]],
    ):
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.mcp_address = mcp_address
        self.agent_display_name = agent_display_name
        self.tool_metadata = tool_metadata

        # Convert MCP tools to OpenAI format, then to ActionTypes using existing functions
        from ...adapters.schemas.openai import mcp_to_openai_tool, openai_tools_to_action_types

        openai_tools = []
        for tool_meta in tool_metadata:

            class MCPTool:
                def __init__(self, meta):
                    self.name = meta["name"]
                    self.description = meta.get("description", "")
                    self.inputSchema = meta.get("inputSchema", {})

            mcp_tool = MCPTool(tool_meta)
            openai_tool = mcp_to_openai_tool(mcp_tool)
            openai_tools.append(openai_tool)

        self.action_types = openai_tools_to_action_types(openai_tools)

        for action_type in self.action_types:
            if action_type.name == "message":
                action_type.is_message = True
            if action_type.name == "finish":
                action_type.is_finish = True
            if action_type.name == "submit":
                action_type.is_finish = True

        self._background_tasks: set[asyncio.Task] = set()

    def _fire_and_forget(self, coro) -> None:
        """Schedule a coroutine without blocking; prevent GC from cancelling it."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _get_span_manager(self, tracker, session_id):
        """Get the SessionSpanManager from the OtelTracingObserver, or None."""
        from ...observers.handlers.otel import OtelTracingObserver

        for obs in tracker._observers:
            if isinstance(obs, OtelTracingObserver):
                return obs._get_span_manager(session_id)
        return None

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute a task using the exgentic agent."""
        from datetime import datetime
        from pathlib import Path

        from opentelemetry import trace as otel_trace

        from ...core.context import Context, OtelContext, get_context, set_context, set_context_fallback
        from ...core.orchestrator.tracker import Tracker
        from ...utils.settings import get_settings

        settings = get_settings()
        run_id = f"a2a_{datetime.now().isoformat().replace(':', '--')}"
        output_dir_path = Path(settings.output_dir).resolve()

        # Extract trace context from OpenTelemetry context (propagated from HTTP headers)
        parent_otel_context = None
        try:
            current_span = otel_trace.get_current_span()
            if current_span and current_span.get_span_context().is_valid:
                span_ctx = current_span.get_span_context()
                trace_id = format(span_ctx.trace_id, "032x")
                span_id = format(span_ctx.span_id, "016x")
                parent_otel_context = OtelContext(trace_id=trace_id, span_id=span_id)
                logger.info(f"Extracted trace context from OTEL: trace_id={trace_id}, span_id={span_id}")
        except Exception as e:
            logger.debug(f"Could not extract OTEL trace context: {e}")

        ctx = Context(
            run_id=run_id,
            output_dir=str(output_dir_path),
            cache_dir=str(Path(settings.cache_dir).resolve()),
            otel_context=parent_otel_context,
        )
        set_context(ctx)
        set_context_fallback(ctx)

        # Default observers include OtelTracingObserver when otel_enabled
        tracker = Tracker()

        # Setup Event Emitter
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        event_emitter = A2AEventEmitter(task_updater)

        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            await event_emitter.emit_event("Error: Empty input provided", failed=True)
            return

        logger.info(f"Processing task: {user_input}")
        logger.info(f"Task execution logs: {output_dir_path / run_id}")
        print(f"\n📁 New request - Logs: {output_dir_path / run_id}")

        self._fire_and_forget(event_emitter.emit_event(f"🚀 Starting task execution with {self.agent_display_name}..."))

        # Extract session_id early (pure string parsing, no I/O)
        session_id_match = re.search(r'session[_ ]id["\s:]+([a-f0-9-]{36})', user_input, re.IGNORECASE)
        if not session_id_match:
            error_msg = "No session_id found in task context. Task must include session_id."
            await event_emitter.emit_event(f"❌ {error_msg}", failed=True)
            return
        session_id = session_id_match.group(1)

        # Create mock session for tracker lifecycle
        from ...core.session import Session as SessionBase

        class A2ASession(SessionBase):
            def __init__(self, task_str, actions, session_id_val, task_id_val):
                self._session_id = session_id_val
                self._task = task_str
                self._actions = actions
                self._task_id = task_id_val
                super().__init__()

            @property
            def task(self) -> str:
                return self._task

            @property
            def context(self) -> dict:
                return {}

            @property
            def actions(self):
                return self._actions

            @property
            def task_id(self) -> str:
                return self._task_id

            def start(self):
                return None

            def step(self, action):
                return None

            def done(self) -> bool:
                return False

            def score(self) -> dict:
                return {"score": 1.0, "success": True}

            def close(self) -> None:
                pass

        mock_session = A2ASession(user_input, self.action_types, session_id, f"a2a_{session_id}")

        # Create the root invoke_agent span BEFORE setup so it captures full duration
        tracker.on_session_enter(session_id, f"a2a_{session_id}")
        tracker.on_session_creation(mock_session)

        # Connect to MCP server for tool execution
        mcp_session = None
        http_context = None
        span_manager = None

        try:
            # Connect to MCP server (captured as child span of invoke_agent)
            if settings.otel_enabled:
                span_manager = self._get_span_manager(tracker, session_id)
                if span_manager:
                    from opentelemetry.trace import SpanKind

                    span_manager.start_span("connect_mcp", kind=SpanKind.INTERNAL)

            self._fire_and_forget(event_emitter.emit_event(f"🔗 Connecting to MCP server at {self.mcp_address}..."))

            http_context = streamable_http_client(self.mcp_address)
            read_stream, write_stream, _ = await http_context.__aenter__()

            mcp_session = ClientSession(read_stream, write_stream)
            await mcp_session.__aenter__()
            await mcp_session.initialize()

            self._fire_and_forget(event_emitter.emit_event("✓ Connected to MCP server"))
            self._fire_and_forget(event_emitter.emit_event(f"✓ Using session_id from task: {session_id}"))

            if span_manager:
                span_manager.end_current_span()

            # Rename the root span and set GenAI/MLflow/Phoenix attributes
            if settings.otel_enabled:
                span_manager = self._get_span_manager(tracker, session_id)
                if span_manager:
                    # Rename from "unknown_benchmark subset session" to proper agent name
                    span_manager.update_current_span_name(f"invoke_agent {self.agent_display_name}")

                    # GenAI semantic conventions (Required)
                    span_manager.set_attribute("gen_ai.operation.name", "invoke_agent")
                    span_manager.set_attribute("gen_ai.provider.name", "exgentic")
                    span_manager.set_attribute("gen_ai.agent.name", self.agent_display_name)

                    # GenAI (Conditionally Required)
                    span_manager.set_attribute("gen_ai.conversation.id", session_id)
                    model_name = self.agent_kwargs.get("model")
                    if model_name:
                        span_manager.set_attribute("gen_ai.request.model", model_name)

                    # MLflow attributes
                    truncated_input = user_input[:1000]
                    span_manager.set_attribute("mlflow.spanInputs", truncated_input)
                    span_manager.set_attribute("mlflow.spanType", "AGENT")
                    span_manager.set_attribute("mlflow.traceName", self.agent_display_name)
                    span_manager.set_attribute("mlflow.trace.session", session_id)

                    # OpenInference (Phoenix) attributes
                    span_manager.set_attribute("openinference.span.kind", "AGENT")
                    span_manager.set_attribute("input.value", truncated_input)

                    # Propagate OTEL context so the venv runner subprocess inherits it
                    span_manager.update_tracing_context()
                    otel_ctx = span_manager.get_otel_context()
                    if otel_ctx:
                        current_ctx = get_context()
                        updated_ctx = current_ctx.with_session(session_id).with_otel_context(otel_ctx)
                        set_context(updated_ctx)
                        set_context_fallback(updated_ctx)
                        logger.info(
                            f"Updated Context: session_id={session_id}, trace_id={otel_ctx.trace_id}, "
                            f"span_id={otel_ctx.span_id}"
                        )

            # Create agent instance — inherits OTEL context via RuntimeConfig
            if span_manager:
                from opentelemetry.trace import SpanKind

                span_manager.start_span("create_agent", kind=SpanKind.INTERNAL)
            loop = asyncio.get_event_loop()
            agent_instance = await loop.run_in_executor(
                None,
                lambda: self.agent_cls(**self.agent_kwargs).get_instance(session_id=session_id),
            )

            await loop.run_in_executor(
                None,
                lambda: agent_instance.start(
                    task=user_input,
                    context={},
                    actions=self.action_types,
                ),
            )
            if span_manager:
                span_manager.end_current_span()

            # on_session_start records initial_observation — must be after create_agent span closes
            tracker.on_session_start(mock_session, agent_instance, None)

            self._fire_and_forget(event_emitter.emit_event(f"✓ Agent initialized with {len(self.action_types)} tools"))

            # Agent loop
            max_iterations = 50
            current_observation = None
            final_result = None
            executor = ThreadPoolExecutor(max_workers=1)

            try:
                for _ in range(max_iterations):
                    loop = asyncio.get_event_loop()
                    action = await loop.run_in_executor(executor, agent_instance.react, current_observation)

                    if action is None:
                        self._fire_and_forget(event_emitter.emit_event("✓ Agent completed execution"))
                        break

                    tracker.on_react_success(mock_session, action)
                    actions_to_execute = action.to_action_list()

                    if len(actions_to_execute) > 1:
                        self._fire_and_forget(
                            event_emitter.emit_event(f"🔧 Parallel Action: {len(actions_to_execute)} actions")
                        )
                    else:
                        self._fire_and_forget(event_emitter.emit_event(f"🔧 Action: {actions_to_execute[0].name}"))

                    # Check if any action is a finish action by matching action names to action types
                    has_finish_action = False
                    for single_action in actions_to_execute:
                        # Find the corresponding ActionType by name
                        action_type = next((at for at in self.action_types if at.name == single_action.name), None)
                        if action_type and action_type.is_finish:
                            has_finish_action = True
                            break

                    results = []
                    for single_action in actions_to_execute:
                        tool_name = single_action.name
                        args_dict = single_action.arguments.model_dump()

                        if tool_name == "message" and "session_id" not in args_dict:
                            args_dict["session_id"] = session_id

                        try:
                            result = await mcp_session.call_tool(tool_name, args_dict)

                            result_text = ""
                            if hasattr(result, "content") and result.content and len(result.content) > 0:
                                first_content = result.content[0]
                                result_text = (
                                    first_content.text if hasattr(first_content, "text") else str(first_content)
                                )
                            else:
                                result_text = str(result)

                            results.append(result_text)
                            self._fire_and_forget(
                                event_emitter.emit_event(f"📊 Result ({tool_name}): {str(result_text)[:200]}")
                            )

                            try:
                                result_json = json.loads(result_text)
                                if result_json.get("status") == "completed":
                                    self._fire_and_forget(event_emitter.emit_event("✓ Session completed successfully"))
                                    final_result = "Session completed"
                                    break
                            except (json.JSONDecodeError, AttributeError):
                                pass

                        except Exception as e:
                            error_msg = f"Error executing {tool_name}: {e}"
                            results.append(error_msg)
                            self._fire_and_forget(event_emitter.emit_event(f"❌ {error_msg}"))

                            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                                self._fire_and_forget(
                                    event_emitter.emit_event("⚠️  Timeout detected, assuming session completed")
                                )
                                final_result = "Session completed (timeout)"
                                break

                    if final_result is not None:
                        break

                    # Build observation for next react() call
                    if len(actions_to_execute) == 1:
                        from ...core.types.observation import SingleObservation

                        current_observation = SingleObservation(
                            result=results[0] if results else "", invoking_actions=actions_to_execute
                        )
                    else:
                        from ...core.types.observation import MultiObservation, SingleObservation

                        observations = []
                        for idx, single_action in enumerate(actions_to_execute):
                            obs = SingleObservation(
                                result=results[idx] if idx < len(results) else "", invoking_actions=[single_action]
                            )
                            observations.append(obs)
                        current_observation = MultiObservation(observations=observations)

                    tracker.on_step_success(mock_session, current_observation)

                    # Terminate after observation if any action was a finish action
                    if has_finish_action:
                        self._fire_and_forget(event_emitter.emit_event("✓ Finish action executed, terminating"))
                        break
            finally:
                executor.shutdown(wait=False)

            # Determine final result
            if final_result is None:
                if current_observation and hasattr(current_observation, "result"):
                    final_result = str(current_observation.result)
                else:
                    final_result = "Task completed"

            # Set output attributes on root span
            if span_manager:
                truncated_output = final_result[:1000]
                span_manager.set_attribute("gen_ai.completion", truncated_output)
                span_manager.set_attribute("mlflow.spanOutputs", truncated_output)
                span_manager.set_attribute("output.value", truncated_output)

            # Notify tracker of session success
            from ...core.types import SessionScore

            score = SessionScore(success=False, score=-1.0, is_finished=True)
            tracker.on_session_success(mock_session, score, agent_instance)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, agent_instance.close)
            await event_emitter.emit_event(final_result, final=True)

            # Flush traces to ensure they're exported
            from ...utils.otel import flush_traces

            await loop.run_in_executor(None, flush_traces)

        except Exception as e:
            logger.exception(f"Error executing task: {e}")

            # Record error on root span
            if span_manager:
                span_manager.record_exception(e)

            if mock_session:
                tracker.on_session_error(mock_session, e)

            await event_emitter.emit_event(f"Error: {e!s}", failed=True)
        finally:
            if mcp_session:
                try:
                    await mcp_session.__aexit__(None, None, None)
                except Exception:
                    pass
            if http_context:
                try:
                    await http_context.__aexit__(None, None, None)
                except Exception:
                    pass

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel task execution (not implemented)."""
        raise Exception("cancel not supported")


__all__ = ["A2AEventEmitter", "ExgenticAgentExecutor"]

# Made with Bob
