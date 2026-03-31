# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from ...core.agent import Agent
from ...core.types import ModelSettings


class LiteLLMToolCallingAgent(Agent):
    """Agent factory that always assigns the message-to-user mapping variant."""

    display_name: ClassVar[str] = "LiteLLM Tool Calling"
    slug_name: ClassVar[str] = "tool_calling"

    model: str = "watsonx/openai/gpt-oss-120b"
    max_steps: int = 150
    enable_tool_shortlisting: bool = False
    max_selected_tools: int = 30
    model_settings: ModelSettings | None = None
    allow_truncated_messages: bool = False

    @classmethod
    def _get_instance_class(cls):
        from .instance import LiteLLMToolCallingAgentInstance

        return LiteLLMToolCallingAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.litellm_tool_calling.instance:LiteLLMToolCallingAgentInstance"

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model)]

    def _get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model": self.model,
            "enable_tool_shortlisting": self.enable_tool_shortlisting,
            "max_selected_tools": self.max_selected_tools,
            "max_steps": self.max_steps,
            "model_settings": self.model_settings,
            "allow_truncated_messages": self.allow_truncated_messages,
        }
