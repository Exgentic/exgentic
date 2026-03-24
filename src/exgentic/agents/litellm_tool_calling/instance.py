# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import time
from typing import Any, Union

import litellm
from litellm import (
    ChatCompletionAssistantMessage,
    ChatCompletionDeveloperMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
)

from ...core.agent_instance import AgentInstance
from ...core.context import get_context
from ...core.types import (
    Action,
    ActionType,
    Message,
    MessageAction,
    MessageObservation,
    MessagePayload,
    ModelSettings,
    Observation,
    RetryStrategy,
)
from ...integrations.litellm.health import check_model_accessible_sync
from ...utils.cost import LiteLLMCostReport
from ...utils.settings import get_settings
from .utils import ToolCall, ToolsActionsRegistry

settings = get_settings()


class NonRetryableCompletionError(ValueError):
    """Raised when a completion response should not be retried."""


class LiteLLMToolCallingAgentInstance(AgentInstance):
    """Ultra-simple tool-calling agent.

    - If the model produces tool_calls, convert them directly to Actions without schema verification.
    - If the model produces a plain assistant message, interpret it as a `message` action
      (even if that tool is not advertised) and emit a corresponding Action.
    """

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o-mini",
        max_steps: int = 150,
        enable_tool_shortlisting: bool = True,
        max_selected_tools: int = 30,
        model_settings: ModelSettings | None = None,
        allow_truncated_messages: bool = False,
    ):
        super().__init__(session_id)
        self.model = model
        self.max_steps = max_steps
        self.enable_tool_shortlisting = enable_tool_shortlisting
        self.max_selected_tools = max_selected_tools
        if model_settings is None:
            self.model_settings = ModelSettings()
        elif isinstance(model_settings, ModelSettings):
            self.model_settings = model_settings
        else:
            raise ValueError("model_settings must be a ModelSettings instance.")
        self._allow_truncated_messages = allow_truncated_messages
        self._use_cache = settings.litellm_caching
        self.logger.debug(
            "LiteLLM cache %s (dir=%s)",
            "enabled" if self._use_cache else "disabled",
            settings.resolved_litellm_cache_dir(),
        )

        self.messages: list[
            Union[
                ChatCompletionAssistantMessage,
                ChatCompletionToolMessage,
                ChatCompletionUserMessage,
            ]
        ] = []
        self._step_count = 0
        self._cost_data = LiteLLMCostReport.initialize_empty(model_name=self.model)

        # Check model accessibility
        check_model_accessible_sync(self.model, logger=self.logger)

    def start(self, task, context, actions):
        """Receive work payload, build tool registry, and seed conversation."""
        super().start(task, context, actions)

        for a in self.actions:
            if not isinstance(a, ActionType):
                raise ValueError("Invalid action type provided to agent")

        self._all_actions: list[ActionType] = list(self.actions)
        self._registry = ToolsActionsRegistry(self._all_actions)

        # Seed conversation with task + context
        ctx = ""
        if self.context:
            ctx = "".join(f"\n<{k}>\n{v}\n</{k}>" for k, v in self.context.items())
        self._add_message(ChatCompletionUserMessage(role="user", content=f"{self.task}\n{ctx}"))

    def _register_cost(self, usage: litellm.Usage):
        self._cost_data.update_cost_from_tokens(usage.prompt_tokens, usage.completion_tokens)

    def _add_message(self, message):
        self.logger.info(f"Adding message to chat history: {message}")
        self.messages.append(message)

    def _observe(self, observation: Observation | None):
        if observation is None:
            self.logger.info("Skipping observation: None")
            return

        observations = observation.to_observation_list()
        if observation.is_empty():
            # Preserve tool results even when the result payload is empty.
            if not any(obs.invoking_actions for obs in observations):
                self.logger.info("Skipping observation: empty with no invoking_actions")
                return

        for obs in observations:
            # Structured user messages: add and move on
            if isinstance(obs, MessageObservation) and isinstance(obs.result, MessagePayload):
                self._add_message(ChatCompletionUserMessage(role="user", content=obs.result.message))
                continue

            if len(obs.invoking_actions) > 0:
                invoking = obs.invoking_actions[0]
                if invoking.name == "message":
                    # Fallback: treat as user-visible content
                    self._add_message(ChatCompletionUserMessage(role="user", content=str(obs)))
                    continue
                action_id = invoking.id
                tool_call_id = invoking.id
                if not (isinstance(tool_call_id, str) and tool_call_id.startswith("call_")):
                    tool_call_id = self._registry.action_id_to_tool_call_id.get(action_id)
                if tool_call_id is None:
                    raise RuntimeError(f"Unable to map tool call id for action {action_id}")
                value = obs.result
                try:
                    content = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                except TypeError:
                    content = str(value)
                self._add_message(ChatCompletionToolMessage(role="tool", tool_call_id=tool_call_id, content=content))
            else:
                self._add_message(ChatCompletionUserMessage(role="user", content=str(obs)))

    def _assistant_tools(self) -> list[dict[str, Any]]:
        """Returns list of available tools in openai format.

        If the number of available tools is less the max_selected_tools parameter,
        returns all available tools.

        Otherwise, calls an LLM to shortlist the tools, and find the most relevant one for
        the current stage in the chat,

        """
        tools = self._registry.openai_tools()

        if not self.enable_tool_shortlisting:
            self.logger.info(
                "Tool shortlisting disabled: returning all %d tools",
                len(tools),
            )
            return tools

        if len(tools) <= self.max_selected_tools:
            self.logger.info(
                "Tool shortlist bypassed: %d tools <= max_selected_tools",
                len(tools),
            )
            return tools
        self.logger.info(
            "Selecting tools: %d available -> top %d",
            len(tools),
            self.max_selected_tools,
        )

        names = [tool["function"]["name"] for tool in tools]

        names_str = ""
        for tool in tools:
            names_str += f"\n- {tool['function']['name']}: {tool['function']['description']}"

        history_text = self._render_history_for_shortlist()
        self.logger.info("Tool shortlist history chars: %d", len(history_text))

        dev = ChatCompletionDeveloperMessage(
            role="developer",
            content=(
                f"Please before providing your next move list the names of the top "
                f"{self.max_selected_tools} tools that are somewhat relevant for the next step, "
                "ordered by relevancy (most to least). Return ONLY a JSON object with this shape: "
                '{\n  "tools": ["tool_name_1", "tool_name_2", ...]\n}.\n'
                f"Choose from these tools only: {names_str}.\n"
                f"Do not call any of those tools just return the list of the top "
                f"{self.max_selected_tools} relevant tools names in the required format."
            ),
        )
        history = ChatCompletionUserMessage(
            role="user",
            content=f"Conversation so far (plain text):\n{history_text}",
        )

        response = self._completion(
            model=self.model,
            messages=[dev, history],
            caching=self._use_cache,
        )

        self._register_cost(response.usage)

        text = response.choices[0].message["content"]

        if text is None:
            text = str(response.choices[0].message)

        self.logger.info("Tool shortlist model response: %s", text)

        positions = []
        for name in names:
            idx = text.find(name)
            if idx != -1:
                positions.append((idx, name))

        if len(positions) == 0:
            selected_tools = tools[: self.max_selected_tools]
            self.logger.info(
                "Tool shortlist fallback: %d -> %d (no matches in model response)",
                len(tools),
                len(selected_tools),
            )
            if len(selected_tools) == 0:
                self.logger.warning("Tool shortlist reduced to 0 tools")
            return selected_tools

        # Sort tools by the order they appear in the model response
        positions.sort(key=lambda x: x[0])

        ordered_tools = [name for _, name in positions]

        selected_names = ordered_tools[: self.max_selected_tools]
        name_to_tool = {tool["function"]["name"]: tool for tool in tools}
        selected_tools = [name_to_tool[name] for name in selected_names]
        self.logger.info(
            "Tool shortlist from model: %d -> %d",
            len(tools),
            len(selected_tools),
        )
        if len(selected_tools) == 0:
            self.logger.warning("Tool shortlist reduced to 0 tools")
        return selected_tools

    def _render_history_for_shortlist(self) -> str:
        parts = []
        for message in self.messages:
            msg = self._message_to_dict(message)
            role = msg.get("role") or "unknown"
            if role == "tool":
                content = msg.get("content", "")
                parts.append(f"tool: {content}")
                continue
            content = msg.get("content")
            if content:
                parts.append(f"{role}: {content}")
            tool_calls = msg.get("tool_calls") or []
            for tool_call in tool_calls:
                function = tool_call.get("function") or {}
                name = function.get("name") or tool_call.get("name")
                arguments = function.get("arguments")
                parts.append(f"{role} tool_call: {name}({arguments})")
        return "\n".join(parts)

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            return message.model_dump()
        if hasattr(message, "dict"):
            return message.dict()
        raise TypeError(f"Unsupported message type: {type(message).__name__}")

    def _extract_tool_calls(self, message: litellm.Message) -> list[ToolCall]:
        """Extract tool calls from the message object returned from the litellm call."""
        if not message.tool_calls:
            return []

        tool_calls: list[ToolCall] = []

        for tool_call in message.tool_calls:
            tool_calls.append(
                {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                    "id": tool_call.id,
                }
            )

        return tool_calls

    def react(self, observation: Observation | None) -> Action | None:
        self._step_count += 1
        if self._step_count > self.max_steps:
            self.logger.warning("Finished: max steps reached (%d)", self.max_steps)
            return None

        self._observe(observation)

        response = self._completion(
            model=self.model,
            messages=self.messages,
            tools=self._assistant_tools(),
            caching=self._use_cache,
        )

        self._register_cost(response.usage)

        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")

        if finish_reason == "tool_calls":
            tool_calls = self._extract_tool_calls(message)
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                            },
                        }
                        for tool_call in tool_calls
                    ],
                )
            )
            actions = self._registry.tool_calls_to_action(tool_calls)
        else:
            actions = MessageAction(arguments=Message(content=message.content))
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    content=message.content,
                )
            )

        self.logger.info(f"Invoking action: {actions}")
        return actions

    def _completion(self, **kwargs):
        call_kwargs = self.model_settings.model_dump(
            exclude_none=True,
            exclude={"num_retries", "retry_after", "retry_strategy"},
        )
        call_kwargs.update(kwargs)
        # Use 'metadata' parameter instead of 'litellm_metadata' - LiteLLM passes this to callbacks
        call_kwargs.setdefault("metadata", {})["context"] = get_context()
        return self._completion_with_retries(call_kwargs)

    def _completion_with_retries(self, call_kwargs: dict[str, Any]):
        num_retries = self.model_settings.num_retries or 0
        max_attempts = max(1, num_retries + 1)
        for attempt in range(max_attempts):
            try:
                response = litellm.completion(max_retries=0, **call_kwargs)
                self._raise_if_invalid_completion(response)
                return response
            except NonRetryableCompletionError:
                raise
            except Exception as exc:
                if attempt >= num_retries:
                    raise
                delay = self.model_settings.retry_after
                retry_strategy = self.model_settings.retry_strategy.value
                if retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF.value:
                    delay *= 2**attempt
                self.logger.warning(
                    "LiteLLM completion failed (attempt %d/%d): %s",
                    attempt + 1,
                    num_retries + 1,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)
        return None

    def _raise_if_invalid_completion(self, response: Any) -> None:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
        except Exception:
            return

        if finish_reason == "length" and not self._allow_truncated_messages:
            self.logger.error(
                "LiteLLM completion truncated (finish_reason=length). Raw response: %s",
                response,
            )
            raise NonRetryableCompletionError(
                "LiteLLM completion truncated (finish_reason=length). "
                "To allow truncated responses, configure the agent with "
                "allow_truncated_messages=True, or increase max_tokens."
            )

        if finish_reason != "tool_calls":
            if message is None or message.content is None:
                self.logger.error(
                    "LiteLLM completion missing assistant content " "(finish_reason=%s). Raw response: %s",
                    finish_reason,
                    response,
                )
                raise ValueError("LiteLLM completion missing assistant content.")

    def close(self) -> None:
        pass

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost_data
