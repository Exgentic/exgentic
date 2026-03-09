# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from exgentic.agents.litellm_tool_calling.utils import ToolsActionsRegistry
from exgentic.core.actions import ActionsHandler
from exgentic.core.types import SingleAction
from pydantic import BaseModel


class EmptyArgs(BaseModel):
    pass


class DummyAction(SingleAction):
    arguments: EmptyArgs


def test_unknown_tool_call_name_yields_unknown_action_observation():
    registry = ToolsActionsRegistry(actions=[])
    tool_calls = [{"name": "not_a_tool", "arguments": "{}", "id": "call-1"}]

    action = registry.tool_calls_to_action(tool_calls)

    assert action is not None
    assert action.validation.name_valid is False
    assert action.validation.error == "Unknown action"

    handler = ActionsHandler()
    observation = handler.execute(action)

    assert "Unknown action" in str(observation.result)
