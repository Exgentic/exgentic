# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A Agent Executor for Exgentic agents."""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
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
            await self.task_updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    message,
                    self.task_updater.context_id,
                    self.task_updater.task_id,
                ),
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
            # Create a simple object with the required attributes
            class MCPTool:
                def __init__(self, meta):
                    self.name = meta["name"]
                    self.description = meta.get("description", "")
                    self.inputSchema = meta.get("inputSchema", {})
            
            mcp_tool = MCPTool(tool_meta)
            openai_tool = mcp_to_openai_tool(mcp_tool)
            openai_tools.append(openai_tool)
        
        # Use the existing function that creates picklable ActionTypes
        self.action_types = openai_tools_to_action_types(openai_tools)
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute a task using the exgentic agent."""
        from ...core.context import Context, set_context, set_context_fallback
        from ...utils.settings import get_settings
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

        # Connect to MCP server for tool execution
        mcp_session = None
        http_context = None
        try:
            await event_emitter.emit_event(f"🔗 Connecting to MCP server at {self.mcp_address}...")
            
            # Connect to MCP server
            http_context = streamable_http_client(self.mcp_address)
            read_stream, write_stream, _ = await http_context.__aenter__()
            
            mcp_session = ClientSession(read_stream, write_stream)
            await mcp_session.__aenter__()
            await mcp_session.initialize()
            
            await event_emitter.emit_event(f"✓ Connected to MCP server")
            
            # Generate a unique session ID for this task execution
            session_id = str(uuid.uuid4())
            
            # Create agent instance
            agent_instance = self.agent_cls(**self.agent_kwargs).get_instance(session_id=session_id)
            
            # Start the agent with task and actions
            agent_instance.start(
                task=user_input,
                context={},
                actions=self.action_types,
            )
            
            await event_emitter.emit_event(f"✓ Agent initialized with {len(self.action_types)} tools")
            
            # Run the agent loop - call MCP directly without storing session in objects
            max_iterations = 50
            current_observation = None
            final_result = None
            
            # Create a thread pool executor for running synchronous agent.react() calls
            executor = ThreadPoolExecutor(max_workers=1)
            
            try:
                for i in range(max_iterations):
                    # Run agent.react() in a thread pool to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    action = await loop.run_in_executor(executor, agent_instance.react, current_observation)
                    
                    if action is None:
                        # Agent is done
                        await event_emitter.emit_event("✓ Agent completed execution")
                        break
                    
                    # Log action
                    await event_emitter.emit_event(f"🔧 Action: {action.name}")
                    
                    # Check if this is a message action (agent's final response)
                    if action.name == "message":
                        # Extract the message content
                        from ...core.types.action import MessageAction
                        if isinstance(action, MessageAction):
                            message_content = action.arguments.content
                            await event_emitter.emit_event(f"💬 Agent response: {message_content}")
                            # This is the final answer, break the loop
                            final_result = message_content
                            break
                        else:
                            # Fallback: treat as string
                            final_result = str(action.arguments)
                            break
                    
                    # Execute the action by calling MCP directly
                    tool_name = action.name
                    args_dict = action.arguments.model_dump()
                    
                    try:
                        # Call MCP server directly without storing in adapter
                        result = await mcp_session.call_tool(tool_name, args_dict)
                        
                        # Extract the result content
                        result_text = ""
                        if hasattr(result, "content") and result.content:
                            if len(result.content) > 0:
                                first_content = result.content[0]
                                if hasattr(first_content, "text"):
                                    result_text = first_content.text
                                else:
                                    result_text = str(first_content)
                            else:
                                result_text = str(result)
                        else:
                            result_text = str(result)
                        
                        # Create proper Observation object
                        from ...core.types.observation import SingleObservation
                        current_observation = SingleObservation(
                            result=result_text,
                            invoking_actions=[action]
                        )
                        
                        # Log result
                        result_str = str(result_text)[:200]
                        await event_emitter.emit_event(f"📊 Result: {result_str}")
                        
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {e}"
                        from ...core.types.observation import SingleObservation
                        current_observation = SingleObservation(
                            result=error_msg,
                            invoking_actions=[action]
                        )
                        await event_emitter.emit_event(f"❌ {error_msg}")
                    
                    # Check if this is a finish action
                    if hasattr(action, 'name') and 'finish' in action.name.lower():
                        break
            finally:
                executor.shutdown(wait=False)
            
            # Get final result
            if final_result is None:
                if current_observation and hasattr(current_observation, 'result'):
                    final_result = str(current_observation.result)
                else:
                    final_result = "Task completed"
            
            # Cleanup
            agent_instance.close()
            
            await event_emitter.emit_event(final_result, final=True)
            
        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            await event_emitter.emit_event(f"Error: {str(e)}", failed=True)
        finally:
            # Cleanup MCP session and HTTP context
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


__all__ = ["ExgenticAgentExecutor", "A2AEventEmitter"]

# Made with Bob
