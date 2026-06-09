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
from a2a.types.a2a_pb2 import (
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
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
        # In a2a-sdk v1.x, Part is a protobuf message with a ``text`` oneof field.
        if part.HasField("text"):
            texts.append(part.text)
        elif part.HasField("data"):
            # DataPart — serialize its structured data
            from google.protobuf.json_format import MessageToDict  # noqa: PLC0415

            try:
                texts.append(json.dumps(MessageToDict(part.data)))
            except (TypeError, ValueError):
                texts.append(str(part.data))
    return "\n".join(texts)


def _extract_text_from_task(task: Task) -> str:
    """Extract text from a Task's status message and artifacts."""
    parts: list[str] = []

    # Check task status message
    if task.status and task.status.HasField("message"):
        status_text = _extract_text_from_message(task.status.message)
        if status_text:
            parts.append(status_text)

    # Check artifacts
    for artifact in task.artifacts:
        for part in artifact.parts:
            if part.HasField("text") and part.text:
                parts.append(part.text)

    return "\n".join(parts)


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
            import httpx

            httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout))
            config = ClientConfig(streaming=False, httpx_client=httpx_client)
            factory = ClientFactory(config)
            try:
                self._client = await factory.create_from_url(self._agent_url)
            except ValueError:
                # Agent card may not include supportedInterfaces — fall back
                # to constructing a card with the known URL and JSON-RPC binding.
                from a2a.types import AgentCard, AgentInterface, AgentCapabilities

                jsonrpc_url = self._agent_url.rstrip("/") + "/jsonrpc"
                card = AgentCard(
                    name="a2a-agent",
                    version="1.0.0",
                    capabilities=AgentCapabilities(),
                    supported_interfaces=[
                        AgentInterface(
                            url=jsonrpc_url,
                            protocol_binding="JSONRPC",
                        ),
                    ],
                )
                self._client = factory.create(card)
        return self._client

    async def _async_send(self, text: str) -> str:
        """Send a message to the A2A agent and return the response text.

        Uses ``Client.send_message`` which yields ``StreamResponse`` protobuf
        messages with a ``payload`` oneof of ``task``, ``message``,
        ``status_update``, or ``artifact_update``.
        """
        client = await self._async_init_client()

        # Build the A2A SendMessageRequest using protobuf types
        message = Message(
            role=Role.ROLE_USER,
            parts=[Part(text=text)],
            message_id=str(uuid.uuid4()),
            context_id=self._context_id,
        )
        # Attach task_id if we have one from a previous exchange
        if self._task_id:
            message.task_id = self._task_id

        request = SendMessageRequest(message=message)

        response_text = ""
        try:
            async for stream_resp in client.send_message(request):
                # stream_resp is a StreamResponse protobuf with a payload oneof
                if not isinstance(stream_resp, StreamResponse):
                    continue

                if stream_resp.HasField("message"):
                    msg = stream_resp.message
                    msg_text = _extract_text_from_message(msg)
                    if msg_text:
                        response_text = msg_text

                elif stream_resp.HasField("task"):
                    task_obj = stream_resp.task
                    self._task_id = task_obj.id

                    # Extract text from task
                    task_text = _extract_text_from_task(task_obj)
                    if task_text:
                        response_text = task_text

                    # Handle task state transitions
                    if task_obj.status:
                        state = task_obj.status.state
                        if state in (
                            TaskState.TASK_STATE_COMPLETED,
                            TaskState.TASK_STATE_FAILED,
                            TaskState.TASK_STATE_CANCELED,
                            TaskState.TASK_STATE_REJECTED,
                        ):
                            # Terminal state: clear task_id so the next
                            # send creates a new task within the same
                            # context (same context_id).
                            self._task_id = None

                        if state == TaskState.TASK_STATE_FAILED:
                            if not response_text:
                                response_text = "Agent task failed"
                            logger.warning(
                                "A2A task %s failed: %s",
                                task_obj.id,
                                response_text[:200],
                            )
                        elif state == TaskState.TASK_STATE_REJECTED:
                            if not response_text:
                                response_text = "Agent rejected the task"
                            logger.warning(
                                "A2A task %s rejected", task_obj.id
                            )
                        elif state == TaskState.TASK_STATE_INPUT_REQUIRED:
                            # The agent needs more input — keep the
                            # task_id so we continue the same task
                            self._task_id = task_obj.id
                            logger.info(
                                "A2A task %s requires additional input",
                                task_obj.id,
                            )

                elif stream_resp.HasField("status_update"):
                    update = stream_resp.status_update
                    if update.status and update.status.HasField("message"):
                        update_text = _extract_text_from_message(
                            update.status.message
                        )
                        if update_text:
                            response_text = update_text

                elif stream_resp.HasField("artifact_update"):
                    artifact = stream_resp.artifact_update
                    if artifact.HasField("artifact"):
                        for part in artifact.artifact.parts:
                            if part.HasField("text") and part.text:
                                if response_text:
                                    response_text += "\n"
                                response_text += part.text

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
