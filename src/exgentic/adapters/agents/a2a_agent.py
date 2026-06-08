# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A agent adapter — consumes an external A2A-speaking agent as an Exgentic
benchmark participant.

Uses the official A2A Python SDK (``a2a-sdk``) which provides a high-level
:class:`Client` abstraction over JSON-RPC / REST / gRPC transports.

Protocol flow
~~~~~~~~~~~~~
1.  :meth:`start` sends the initial prompt (task + context + action schemas)
    via ``Client.send_message``.
2.  Each :meth:`react` call serializes the latest
    :class:`~exgentic.core.types.Observation` to text, sends it as the next
    user turn, and parses the agent response into an
    :class:`~exgentic.core.types.Action`.
3.  Multi-turn session continuity is maintained via ``context_id``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Optional

from a2a.client import ClientConfig, ClientFactory
from a2a.client.client import Client
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

from ...core.agent_instance import AgentInstance
from ...core.types import (
    Action,
    ActionType,
    Message as ExgMessage,
    MessageAction,
    Observation,
    SingleObservation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers – action schema serialization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers – prompt construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers – response parsing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers – A2A message text extraction
# ---------------------------------------------------------------------------


def _extract_text_from_message(message: Message) -> str:
    """Extract concatenated text from an A2A Message's parts."""
    texts: list[str] = []
    if not message.parts:
        return ""
    for part in message.parts:
        # Part is a RootModel; the discriminated union is in part.root
        inner = part.root
        if isinstance(inner, TextPart):
            texts.append(inner.text)
        elif hasattr(inner, "data") and inner.data is not None:
            # DataPart — serialize its structured data
            try:
                texts.append(json.dumps(inner.data))
            except (TypeError, ValueError):
                texts.append(str(inner.data))
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# Helpers – async ↔ sync bridge
# ---------------------------------------------------------------------------


class _AsyncBridge:
    """Manages a persistent event loop in a background thread.

    This is necessary because the A2A SDK's :class:`Client` holds an
    ``httpx.AsyncClient`` whose connections are bound to the event loop
    that created them.  Using ``asyncio.run()`` per call would destroy the
    loop (and connections) between ``start()`` and ``react()`` calls.
    """

    def __init__(self) -> None:
        import threading

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

    def run(self, coro, timeout: float | None = None):
        """Submit a coroutine to the persistent loop and block for result.

        Args:
            coro: The coroutine to run.
            timeout: Maximum seconds to wait.  ``None`` means wait forever.
                     Raises :class:`TimeoutError` if exceeded.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        """Stop the background loop and join the thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


# ---------------------------------------------------------------------------
# A2AAgentInstance
# ---------------------------------------------------------------------------


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
        timeout: float = 300.0,
    ) -> None:
        super().__init__(session_id)
        self._agent_url = agent_url
        self.max_steps = max_steps
        self._timeout = timeout
        self._client: Optional[Client] = None
        self._task_id: Optional[str] = None
        self._context_id: str = str(uuid.uuid4())
        self._step_count = 0
        self._actions: list[ActionType] = []
        self._bridge = _AsyncBridge()

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
        response_text = self._bridge.run(
            self._async_send(prompt), timeout=self._timeout
        )
        self.logger.info(
            "A2A agent initial response: %s",
            response_text[:500] if response_text else "(empty)",
        )

    async def _async_init_client(self) -> Client:
        """Lazily initialize the A2A client via ClientFactory."""
        if self._client is None:
            config = ClientConfig(streaming=True)
            self._client = await ClientFactory.connect(
                self._agent_url,
                client_config=config,
            )
        return self._client

    async def _async_send(self, text: str) -> str:
        """Send a message to the A2A agent and return the response text.

        Uses ``Client.send_message`` which takes a :class:`Message` directly
        and returns an async iterator yielding either:
        - ``tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None]``
        - ``Message``
        """
        client = await self._async_init_client()

        # Build the A2A Message using the SDK helper pattern
        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid.uuid4()),
            context_id=self._context_id,
            task_id=self._task_id,
        )

        response_text = ""
        try:
            async for event in client.send_message(message):
                # Client.send_message yields either:
                #   - tuple[Task, update_event | None]  (task lifecycle)
                #   - Message                           (message-only)
                if isinstance(event, Message):
                    msg_text = _extract_text_from_message(event)
                    if msg_text:
                        response_text = msg_text
                elif isinstance(event, tuple):
                    task_obj, update_event = event
                    if isinstance(task_obj, Task):
                        self._task_id = task_obj.id

                        # Check task status message
                        if task_obj.status and task_obj.status.message:
                            status_text = _extract_text_from_message(
                                task_obj.status.message
                            )
                            if status_text:
                                response_text = status_text

                        # Check artifacts for response content
                        if task_obj.artifacts:
                            for artifact in task_obj.artifacts:
                                for part in artifact.parts:
                                    inner = part.root
                                    if isinstance(inner, TextPart) and inner.text:
                                        if response_text:
                                            response_text += "\n"
                                        response_text += inner.text

                        # Handle task state transitions
                        if task_obj.status:
                            state = task_obj.status.state
                            if state in (
                                TaskState.completed,
                                TaskState.failed,
                                TaskState.canceled,
                                TaskState.rejected,
                            ):
                                # Terminal state: clear task_id so the next
                                # send creates a new task within the same
                                # context (same context_id).  This is
                                # how A2A multi-turn works.
                                self._task_id = None

                            if state == TaskState.failed:
                                if not response_text:
                                    response_text = "Agent task failed"
                                logger.warning(
                                    "A2A task %s failed: %s",
                                    task_obj.id,
                                    response_text[:200],
                                )
                            elif state == TaskState.rejected:
                                if not response_text:
                                    response_text = "Agent rejected the task"
                                logger.warning(
                                    "A2A task %s rejected", task_obj.id
                                )
                            elif state == TaskState.input_required:
                                # The agent needs more input — keep the
                                # task_id so we continue the same task
                                self._task_id = task_obj.id
                                logger.info(
                                    "A2A task %s requires additional input",
                                    task_obj.id,
                                )

                    # Process status update events for additional text
                    if isinstance(update_event, TaskStatusUpdateEvent):
                        if update_event.status and update_event.status.message:
                            update_text = _extract_text_from_message(
                                update_event.status.message
                            )
                            if update_text:
                                response_text = update_text
                    elif isinstance(update_event, TaskArtifactUpdateEvent):
                        if update_event.artifact:
                            for part in update_event.artifact.parts:
                                inner = part.root
                                if isinstance(inner, TextPart) and inner.text:
                                    if response_text:
                                        response_text += "\n"
                                    response_text += inner.text

        except Exception as exc:
            logger.error("A2A send_message failed: %s", exc, exc_info=True)
            raise

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
        try:
            response_text = self._bridge.run(
                self._async_send(obs_text), timeout=self._timeout
            )
        except Exception as exc:
            self.logger.error("A2A communication error: %s", exc)
            return None

        self.logger.info(
            "A2A agent response: %s",
            response_text[:500] if response_text else "(empty)",
        )

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
        """Close the A2A client connection and the async bridge."""
        self._client = None
        try:
            self._bridge.shutdown()
        except Exception:
            pass
