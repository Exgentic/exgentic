# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from typing import ClassVar

from .base_agent import SmolagentBaseAgent


class SmolagentToolCallingAgent(SmolagentBaseAgent):
    display_name: ClassVar[str] = "SmolAgents Tool Calling"
    slug_name: ClassVar[str] = "smolagents_tool"

    @classmethod
    def _get_instance_class(cls):
        from .tool_calling_instance import SmolagentToolCallingAgentInstance

        return SmolagentToolCallingAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.smolagents.tool_calling_instance:SmolagentToolCallingAgentInstance"
