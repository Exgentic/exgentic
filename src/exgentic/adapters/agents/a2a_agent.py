# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A agent adapter — consumes an external A2A-speaking agent as an Exgentic
benchmark participant.

Uses the official A2A Python SDK (``a2a-sdk``) which supports JSON-RPC, gRPC,
and REST transports via a generic protobuf data binding — demonstrating that A2A
is **not** bound to a single transport.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Optional

from a2a.client import ClientConfig, create_client
from a2a.client.client import Client
from a2a.types import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)

from ...core.agent_instance import AgentInstance
from ...core.types import (
    Action,
    ActionType,
    Message as ExgMessage,
    MessageAction,
    Observation,
    SingleAction,
    SingleObservation,
)


def _action_types_to_schema(actions: list[ActionType]) -> list[dict[str, Any]]:
    """Serialize Exgentic ActionTypes to JSON-serializable dicts for the prompt."""
    schemas: list[dict[str, Any]] = []
    for at in actions:
        entry: dict[str, Any] = {
            "name": at.name,
            "description": at.description,
        }
        arg_model = at.arguments
        if arg_model is not None and hasattr(arg_model, "model_json_schema"):
            entry["parameters"] = arg_model.model_json_schema()
        schemas.append(entry)
    return schemas


def _build_system_prompt(
    task: str,
    context: dict[str, Any],
    action_schemas: list[dict[str, Any]],
) -> str:
    """Build the initial prompt sent to the A2A agent."""
    parts: list[str] = [task]
    if context:
        for key, value in context.items():
            parts.append(f"\n<{key}>\n{value}\n</{key}>")
    parts.append(
        "\n\nYou MUST respond with a JSON object selecting one of the following actions:\n"
    )
    parts.append(json.dumps(action_schemas, indent=2))
    parts.append(
        '\n\nRespond with: {"action": "<action_name>", "arguments": {<action_arguments>}}'
    )
    return "".join(parts)


def _parse_agent_response(
    text: str,
    actions: list[ActionType],
) -> Optional[Action]:
    """Parse the A2A agent's text response into an Exgentic Action.

    Tries structured JSON first, then falls back to a ``message`` action
    (same strategy as the LiteLLM tool-calling agent).
    """
    text = text.strip()
    if not text:
        return None

    # Try to extract JSON from the response
    json_text = text
    # Handle markdown code blocks
    if "```" in json_text:
        # Extract content between code fences
        lines = json_text.split("\n")
        in_block = False
        block_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                block_lines.append(line)
        if block_lines:
            json_text = "\n".join(block_lines)

    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        # Fall back to message action
        return MessageAction(arguments=ExgMessage(content=text))

    if not isinstance(data, dict):
        return MessageAction(arguments=ExgMessage(content=text))

    action_name = data.get("action") or data.get("name")
    arguments = data.get("arguments") or data.get("params") or {}

    if not action_name:
        return MessageAction(arguments=ExgMessage(content=text))

    # Find the matching ActionType
    for at in actions:
        if at.name == action_name:
            return at.build_action(arguments)

    # Unknown action name — wrap as message
    return MessageAction(arguments=ExgMessage(content=text))


def _extract_text_from_message(message: Message) -> str:
    """Extract concatenated text from an A2A Message's parts."""
    texts: list[str] = []
    for part in message.parts:
        if part.text:
            texts.append(part.text)
        elif part.HasField("data"):
            from google.protobuf.json_format import MessageToDict

            texts.append(json.dumps(MessageToDict(part.data)))
    return "\n".join(texts)


def _run_async(coro):
    """Run an async coroutine from sync code, handling event-loop presence."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop — use a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class A2AAgentInstance(AgentInstance):
    """AgentInstance that delegates to an external A2A agent.

    Communicates using the A2A protocol via the official SDK, supporting
    any transport the remote agent advertises (JSON-RPC, gRPC, REST).
    """

    max_steps: int | None = 150

    def __init__(
        self,
        session_id: str,
        agent_url: str,
        max_steps: int = 150,
    ) -> None:
        super().__init__(session_id)
        self._agent_url = agent_url
        self.max_steps = max_steps
        self._client: Optional[Client] = None
        self._task_id: Optional[str] = None
        self._context_id: str = str(uuid.uuid4())
        self._step_count = 0
        self._actions: list[ActionType] = []

    # -- lifecycle --------------------------------------------------------

    def start(self, task: str, context: dict[str, Any], actions: list[ActionType]):
        """Send initial prompt to the A2A agent with task + context + action schemas."""
        super().start(task, context, actions)
        self._actions = list(actions)

        # Build the system prompt that tells the agent what actions are available
        action_schemas = _action_types_to_schema(self._actions)
        prompt = _build_system_prompt(task, context, action_schemas)

        self.logger.info("Connecting to A2A agent at %s", self._agent_url)

        # Create client and send initial message
        response_text = _run_async(self._async_send(prompt))
        self.logger.info("A2A agent initial response: %s", response_text[:500])

    async def _async_init_client(self) -> Client:
        """Lazily initialize the A2A client."""
        if self._client is None:
            config = ClientConfig(streaming=False)
            self._client = await create_client(
                self._agent_url,
                client_config=config,
            )
        return self._client

    async def _async_send(self, text: str) -> str:
        """Send a message to the A2A agent and return the response text."""
        client = await self._async_init_client()

        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
        )

        if self._task_id:
            message.task_id = self._task_id
        message.context_id = self._context_id

        request = SendMessageRequest(message=message)

        response_text = ""
        async for stream_response in client.send_message(request):
            # Extract final task state and response text
            if stream_response.HasField("task"):
                task = stream_response.task
                self._task_id = task.id

                if task.status and task.status.message:
                    response_text = _extract_text_from_message(
                        task.status.message
                    )

                # Also check artifacts for response content
                for artifact in task.artifacts:
                    for part in artifact.parts:
                        if part.text:
                            if response_text:
                                response_text += "\n"
                            response_text += part.text

            elif stream_response.HasField("message"):
                msg_text = _extract_text_from_message(stream_response.message)
                if msg_text:
                    response_text = msg_text

            elif stream_response.HasField("status_update"):
                update = stream_response.status_update
                if update.status and update.status.message:
                    response_text = _extract_text_from_message(
                        update.status.message
                    )

        return response_text

    # -- AgentInstance interface -------------------------------------------

    def react(self, observation: Optional[Observation]) -> Optional[Action]:
        """Send the observation to the A2A agent, parse response into an Action."""
        self._step_count += 1
        if self.max_steps is not None and self._step_count > self.max_steps:
            self.logger.warning("Finished: max steps reached (%d)", self.max_steps)
            return None

        # Build observation text to send
        obs_text = self._observation_to_text(observation)
        if obs_text is None:
            return None

        # Send to A2A agent
        self.logger.info("Sending observation to A2A agent (step %d)", self._step_count)
        response_text = _run_async(self._async_send(obs_text))
        self.logger.info("A2A agent response: %s", response_text[:500])

        if not response_text:
            return None

        # Parse the response into an Action
        action = _parse_agent_response(response_text, self._actions)
        if action is not None:
            self.logger.info("Parsed action: %s", action)
        return action

    def _observation_to_text(self, observation: Optional[Observation]) -> Optional[str]:
        """Convert an Observation to text for the A2A agent."""
        if observation is None:
            return "No observation. Please select your next action."

        if observation.is_empty():
            return "The previous action returned no output. Please select your next action."

        observations = observation.to_observation_list()
        parts: list[str] = []
        for obs in observations:
            if isinstance(obs, SingleObservation):
                if obs.invoking_actions:
                    invoking = obs.invoking_actions[0]
                    parts.append(
                        f"Result of action '{invoking.name}':\n{obs.result}"
                    )
                else:
                    parts.append(str(obs.result))

        if not parts:
            return "No observation content. Please select your next action."

        result = "\n\n".join(parts)
        result += '\n\nPlease respond with your next action as JSON: {"action": "<name>", "arguments": {<args>}}'
        return result

    def close(self) -> None:
        """Close the A2A client connection."""
        if self._client is not None:
            try:
                _run_async(self._client.close())
            except Exception:
                pass
            self._client = None
