# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from typing import ClassVar

from .base_agent import SmolagentBaseAgent


class SmolagentCodeAgent(SmolagentBaseAgent):
    display_name: ClassVar[str] = "SmolAgents Code"
    slug_name: ClassVar[str] = "smolagents_code"

    @classmethod
    def _get_instance_class(cls):
        from .code_instance import SmolagentCodeAgentInstance

        return SmolagentCodeAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "framework.agents.smolagents.code_instance:SmolagentCodeAgentInstance"
