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
        
        # Mark the message action as is_message=True so agents handle it correctly
        for action_type in self.action_types:
            if action_type.name == "message":
                action_type.is_message = True
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute a task using the exgentic agent."""
        from ...core.context import Context, set_context, set_context_fallback
        from ...utils.settings import get_settings
        from datetime import datetime
        from pathlib import Path
        
        # Create a context for this execution
        settings = get_settings()
        run_id = f"a2a_{datetime.now().isoformat().replace(':', '--')}"
        output_dir_path = Path(settings.output_dir).resolve()
        ctx = Context(
            run_id=run_id,
            output_dir=str(output_dir_path),
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

        # Print log location for this request
        logger.info(f"Processing task: {user_input}")
        logger.info(f"📁 Task execution logs will be written to: {output_dir_path / run_id}")
        print(f"\n📁 New request - Logs: {output_dir_path / run_id}")
        
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
            
            # Extract session_id from user_input (must be present in task context)
            import re
            session_id_match = re.search(r'session[_ ]id["\s:]+([a-f0-9-]{36})', user_input, re.IGNORECASE)
            if not session_id_match:
                error_msg = "No session_id found in task context. Task must include session_id."
                await event_emitter.emit_event(f"❌ {error_msg}", failed=True)
                raise ValueError(error_msg)
            
            session_id = session_id_match.group(1)
            await event_emitter.emit_event(f"✓ Using session_id from task: {session_id}")
            
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
                    
                    # Convert action to list of SingleActions (works for all action types)
                    actions_to_execute = action.to_action_list()
                    
                    # Log action(s)
                    if len(actions_to_execute) > 1:
                        await event_emitter.emit_event(f"🔧 Parallel Action: {len(actions_to_execute)} actions")
                    else:
                        await event_emitter.emit_event(f"🔧 Action: {actions_to_execute[0].name}")
                    
                    # Execute all actions
                    results = []
                    for single_action in actions_to_execute:
                        # Check if this is a message action (agent's final response)
#                        if single_action.name == "message":
#                            # Extract the message content
#                            from ...core.types.action import MessageAction
#                            if isinstance(single_action, MessageAction):
#                                message_content = single_action.arguments.content
#                                await event_emitter.emit_event(f"💬 Agent response: {message_content}")
#                                # This is the final answer, break the loop
#                                final_result = message_content
#                                break
#                            else:
#                                # Fallback: treat as string
#                                final_result = str(single_action.arguments)
#                                break
                        
                        # Execute the action by calling MCP directly
                        tool_name = single_action.name
                        args_dict = single_action.arguments.model_dump()
                        
                        # If this is the message tool, inject the session_id
                        if tool_name == "message" and "session_id" not in args_dict:
                            args_dict["session_id"] = session_id
                        
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
                            
                            results.append(result_text)
                            
                            # Log result
                            result_str = str(result_text)[:200]
                            await event_emitter.emit_event(f"📊 Result ({tool_name}): {result_str}")
                            
                            # Check if session is completed
                            try:
                                import json
                                result_json = json.loads(result_text)
                                if result_json.get("status") == "completed":
                                    await event_emitter.emit_event("✓ Session completed successfully")
                                    final_result = "Session completed"
                                    break
                            except (json.JSONDecodeError, AttributeError):
                                pass
                            
                        except Exception as e:
                            error_msg = f"Error executing {tool_name}: {e}"
                            results.append(error_msg)
                            await event_emitter.emit_event(f"❌ {error_msg}")
                            
                            # Check if this is a timeout error indicating completion
                            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                                await event_emitter.emit_event("⚠️  Timeout detected, assuming session completed")
                                final_result = "Session completed (timeout)"
                                break
                        
                        # Check if this is a finish action
                        if hasattr(single_action, 'name') and 'finish' in single_action.name.lower():
                            break
                    
                    # If we got a final result from a message action, break outer loop
                    if final_result is not None:
                        break
                    
                    # Create proper Observation object
                    if len(actions_to_execute) == 1:
                        # Single action - use SingleObservation
                        from ...core.types.observation import SingleObservation
                        current_observation = SingleObservation(
                            result=results[0] if results else "",
                            invoking_actions=actions_to_execute
                        )
                    else:
                        # Multiple actions - use MultiObservation with one SingleObservation per action
                        from ...core.types.observation import SingleObservation, MultiObservation
                        observations = []
                        for idx, single_action in enumerate(actions_to_execute):
                            obs = SingleObservation(
                                result=results[idx] if idx < len(results) else "",
                                invoking_actions=[single_action]
                            )
                            observations.append(obs)
                        current_observation = MultiObservation(observations=observations)
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
