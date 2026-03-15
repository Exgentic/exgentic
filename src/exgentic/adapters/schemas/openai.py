# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, Literal

from pydantic import create_model

from ...core.types import ActionType, SingleAction
from .json_schema import make_args_model_from_json_schema


def openai_tools_to_action_types(tools: list[dict[str, Any]]) -> list[ActionType]:
    """Translate OpenAI-style tools into ActionType definitions.

    Builds concrete argument models from each tool's parameter schema so no
    information is lost when emitting tools back to the LLM via ActionType.
    """
    actions: list[ActionType] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        desc = fn.get("description") or ""
        params = fn.get("parameters") or {}

        args_model = make_args_model_from_json_schema(name, params)

        action_model = create_model(
            f"{name}_Action",
            __base__=SingleAction,
            name=(Literal[name], name),
            arguments=(args_model, ...),
        )
        actions.append(ActionType(name=name, description=str(desc), cls=action_model))

    if not actions:
        raise ValueError("No OpenAI function tools provided to translate into ActionTypes")
    return actions


def mcp_to_openai_tool(mcp_tool: Any) -> dict[str, Any]:
    """Converts a tool definition from a 'mcp' format  into  OpenAI tool schema."""
    function_name = mcp_tool.name
    description = mcp_tool.description
    parameters_schema = mcp_tool.inputSchema

    if not all([function_name, description, parameters_schema]):
        # Raise an informative error if the expected keys are missing
        raise ValueError("MCP tool definition is missing required keys: 'function_name', 'summary', or 'input_schema'.")

    tool_schema = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": description,
            "parameters": parameters_schema,
        },
    }
    return tool_schema


def mcp_tools_to_openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [mcp_to_openai_tool(mcp_tool) for mcp_tool in mcp_tools]
