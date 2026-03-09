# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Chat/tool-call helpers for translating between Exgentic actions and chat payloads."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from ...core.actions import extract_argument
from ...core.types import (
    MessageObservation,
    MessagePayload,
    MultiObservation,
    SingleAction,
    SingleObservation,
)


class ChatActionContext:
    """Helper to map actions to chat content/tool calls and back to observations."""

    def __init__(self) -> None:
        self.message_actions: dict[str, SingleAction] = {}
        self.tool_actions: dict[str, SingleAction] = {}

    @staticmethod
    def action_to_tool_call_payload(action: SingleAction) -> dict[str, Any]:
        arguments: Any = action.arguments
        if isinstance(arguments, str):
            arguments = {"error_parsing": arguments}
        elif isinstance(arguments, BaseModel):
            arguments = arguments.model_dump()
        return {"name": action.name, "arguments": arguments, "id": action.id}

    def actions_to_chat_components(self, actions: list[SingleAction]) -> tuple[Optional[str], list[dict[str, Any]]]:
        content: Optional[str] = None
        tool_calls: list[dict[str, Any]] = []
        self.message_actions = {}
        self.tool_actions = {}

        for act in actions:
            if act.name == "message":
                self.message_actions[act.id] = act
                msg_val = extract_argument(act.arguments, "content", None)
                if msg_val is None:
                    try:
                        msg_val = str(act.arguments)
                    except Exception:
                        msg_val = None
                if msg_val is not None:
                    if content is None:
                        content = ""
                    content += str(msg_val)
                continue

            self.tool_actions[act.id] = act
            tool_calls.append(self.action_to_tool_call_payload(act))

        return content, tool_calls

    def actions_to_assistant_message(self, actions: list[SingleAction]) -> dict[str, Any]:
        """Convert actions into an assistant message dict with content and tool_calls."""
        content, tool_calls = self.actions_to_chat_components(actions)
        message: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def message_to_observation(self, message: Any) -> SingleObservation | MultiObservation:
        # Support a list of messages (e.g., multiple tool responses)
        if isinstance(message, list):
            items = [self.message_to_observation(m) for m in message]
            flat: list[SingleObservation] = []
            for obs in items:
                if isinstance(obs, MultiObservation):
                    flat.extend(obs.observations)
                else:
                    flat.append(obs)
            return MultiObservation(observations=flat)

        if isinstance(message, dict):
            role = message.get("role")
            if role == "user":
                acts = list(self.message_actions.values())
                content = message.get("content") or ""
                payload = MessagePayload(sender="user", message=content)
                return MessageObservation(invoking_actions=acts, result=payload)
            if role == "tool":
                act = self.tool_actions.get(str(message.get("tool_call_id")))
                return SingleObservation(
                    invoking_actions=([act] if act else []),
                    result=message.get("content"),
                )
            if "id" in message and "content" in message:
                act = self.tool_actions.get(str(message.get("id")))
                return SingleObservation(
                    invoking_actions=([act] if act else []),
                    result=message.get("content"),
                )

        return SingleObservation(invoking_actions=[], result=str(message))
