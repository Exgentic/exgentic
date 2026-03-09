# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from typing import Any, Optional

from typing_extensions import TypedDict

from ...core.actions import build_action, build_unknown_action
from ...core.types import (
    Action,
    ActionType,
    ParallelAction,
    SingleAction,
    SingleObservation,
)


class PartialAction(SingleAction):
    arguments: dict


class ToolCall(TypedDict):
    name: str
    arguments: str
    id: str


def extract_arguments(action_type: ActionType):
    return action_type.arguments


class ToolsActionsRegistry:
    _MAX_SAFE_SCHEMA_INT = 2_147_483_647

    @classmethod
    def _clamp_schema_ints(cls, obj):
        if isinstance(obj, dict):
            return {k: cls._clamp_schema_ints(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._clamp_schema_ints(v) for v in obj]
        if isinstance(obj, int) and not isinstance(obj, bool):
            if obj > cls._MAX_SAFE_SCHEMA_INT:
                return cls._MAX_SAFE_SCHEMA_INT
        if isinstance(obj, float):
            if obj > cls._MAX_SAFE_SCHEMA_INT:
                return float(cls._MAX_SAFE_SCHEMA_INT)
        return obj

    @staticmethod
    def format_observation(observation: SingleObservation) -> str:
        """Serialize observation.result to JSON if possible to satisfy tool message requirements."""
        value = observation.result
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(value)

    def __init__(self, actions: list[ActionType]):
        self.action_types: list[ActionType] = []
        self.name_to_action: dict[str, ActionType] = {}
        self.action_id_to_tool_call_id: dict[str, str] = {}
        for action in actions:
            self.add_action(action)

    def add_action(self, action: ActionType):
        if not isinstance(action, ActionType):
            raise ValueError("bad action")
        self.action_types.append(action)
        self.name_to_action[action.name] = action

    def openai_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for action in self.action_types:
            # Skip non-environment messaging actions; agents handle messaging flow
            if action.is_message:
                continue
            arguments_type = extract_arguments(action)
            schema = arguments_type.model_json_schema()  # type: ignore[attr-defined]
            # Bedrock rejects oversized integer values in tool schemas.
            schema = self._clamp_schema_ints(schema)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "description": action.description,
                        "parameters": schema,
                    },
                }
            )
        tools.sort(
            key=lambda tool: (
                tool.get("type", ""),
                tool.get("function", {}).get("name", ""),
            )
        )
        return tools

    def _tool_call_to_single_action(self, tool_call: ToolCall) -> SingleAction:
        name = tool_call["name"]
        action_type = self.name_to_action.get(name)

        action_id = tool_call.get("id")
        if action_type:
            action = build_action(action_type, tool_call["arguments"], action_id=action_id)
        else:
            action = build_unknown_action(name, tool_call.get("arguments", {}), action_id=action_id)

        if "id" not in tool_call:
            tool_call["id"] = action.id

        self.action_id_to_tool_call_id[action.id] = tool_call["id"]

        return action

    def tool_calls_to_action(self, tool_calls: list[ToolCall]) -> Optional[Action]:
        actions: list[SingleAction] = []
        for tool_call in tool_calls:
            actions.append(self._tool_call_to_single_action(tool_call))
        if len(actions) == 0:
            return None
        if len(actions) == 1:
            return actions[0]
        return ParallelAction(actions=actions)


def tool_call_to_dict(tool_call):
    return {
        "function": vars(tool_call.function),
        "id": tool_call.id,
        "type": tool_call.type,
    }
