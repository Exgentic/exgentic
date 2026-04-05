# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A Agent Executor for Exgentic agents."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task

logger = logging.getLogger(__name__)


class A2AEventEmitter:
    """Handle events for A2A Agent."""

    def __init__(self, task_updater: TaskUpdater):
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
            await self.task_updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    message,
                    self.task_updater.context_id,
                    self.task_updater.task_id,
                ),
            )


class ExgenticAgentExecutor(AgentExecutor):
    """Execute tasks using Exgentic agents with MCP tools."""

    def __init__(
        self,
        agent_cls: type,
        agent_kwargs: dict[str, Any],
        mcp_tools: list[Callable],
        agent_display_name: str,
    ):
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs
        self.mcp_tools = mcp_tools
        self.agent_display_name = agent_display_name

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute a task using the exgentic agent."""
        from ...core.context import Context, set_context, set_context_fallback
        from ...core.types import ActionType, SingleAction
        from pydantic import BaseModel, create_model
        from ...utils.settings import get_settings
        import inspect
        from datetime import datetime
        from pathlib import Path
        
        # Create a context for this execution
        settings = get_settings()
        run_id = f"a2a_{datetime.now().isoformat().replace(':', '--')}"
        ctx = Context(
            run_id=run_id,
            output_dir=str(Path(settings.output_dir).resolve()),
            cache_dir=str(Path(settings.cache_dir).resolve()),
        )
        set_context(ctx)
        set_context_fallback(ctx)

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
        await event_emitter.emit_event(f"🚀 Starting task execution with {self.agent_display_name}...")

        try:
            # Generate a unique session ID for this task execution
            session_id = str(uuid.uuid4())
            
            # Create agent instance
            agent_instance = self.agent_cls(**self.agent_kwargs).get_instance(session_id=session_id)
            
            # Convert MCP tools to ActionTypes
            action_types = []
            for tool_func in self.mcp_tools:
                # Get function signature
                sig = inspect.signature(tool_func)
                
                # Create pydantic model for arguments
                field_definitions = {}
                for param_name, param in sig.parameters.items():
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                    if param.default == inspect.Parameter.empty:
                        field_definitions[param_name] = (param_type, ...)
                    else:
                        field_definitions[param_name] = (param_type, param.default)
                
                if field_definitions:
                    ArgsModel = create_model(
                        f"{tool_func.__name__}_args",
                        **field_definitions,
                    )
                else:
                    class ArgsModel(BaseModel):
                        pass
                
                # Create action class
                class ToolAction(SingleAction):
                    name: str = tool_func.__name__
                    arguments: ArgsModel  # type: ignore
                
                # Create ActionType
                action_type = ActionType(
                    name=tool_func.__name__,
                    description=tool_func.__doc__ or f"Execute {tool_func.__name__}",
                    cls=ToolAction,
                )
                action_types.append(action_type)
            
            # Start the agent with task and actions
            agent_instance.start(
                task=user_input,
                context={},
                actions=action_types,
            )
            
            await event_emitter.emit_event(f"✓ Agent initialized with {len(action_types)} tools")
            
            # Create a simple adapter to execute actions
            class SimpleAdapter:
                def __init__(self, tools):
                    self.tools = {t.__name__: t for t in tools}
                    self.observation = None
                
                def get_observation(self):
                    return self.observation
                
                def execute_action(self, action):
                    tool_name = action.name
                    if tool_name in self.tools:
                        tool_func = self.tools[tool_name]
                        args_dict = action.arguments.model_dump()
                        result = tool_func(**args_dict)
                        self.observation = result
                        return result
                    else:
                        self.observation = f"Unknown tool: {tool_name}"
                        return self.observation
            
            adapter = SimpleAdapter(self.mcp_tools)
            
            # Run the agent loop
            max_iterations = 50
            for i in range(max_iterations):
                observation = adapter.get_observation()
                action = agent_instance.react(observation)
                
                if action is None:
                    # Agent is done
                    await event_emitter.emit_event("✓ Agent completed execution")
                    break
                
                # Log action
                await event_emitter.emit_event(f"🔧 Action: {action.name}")
                
                # Execute the action
                result = adapter.execute_action(action)
                
                # Log result
                result_str = str(result)[:200]
                await event_emitter.emit_event(f"📊 Result: {result_str}")
                
                # Check if this is a finish action
                if hasattr(action, 'name') and 'finish' in action.name.lower():
                    break
            
            # Get final result
            final_result = adapter.observation or "Task completed"
            
            # Cleanup
            agent_instance.close()
            
            await event_emitter.emit_event(str(final_result), final=True)
            
        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            await event_emitter.emit_event(f"Error: {str(e)}", failed=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel task execution (not implemented)."""
        raise Exception("cancel not supported")


__all__ = ["ExgenticAgentExecutor", "A2AEventEmitter"]

# Made with Bob
