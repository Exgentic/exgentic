# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from ...core.agent import Agent


class A2AAgent(Agent):
    """Agent that delegates tasks to a remote A2A-compliant agent.

    The ``url`` should be the base URL of the A2A agent (e.g.
    ``https://host.example.com``).  The adapter discovers the agent
    card at ``{url}/.well-known/agent-card.json`` and sends messages
    via ``POST {url}/message:send`` using JSON-RPC 2.0.
    """

    display_name: ClassVar[str] = "A2A Agent"
    slug_name: ClassVar[str] = "a2a"

    url: str
    """Base URL of the A2A-compliant agent."""

    timeout: float = 120.0
    """HTTP request timeout in seconds."""

    @classmethod
    def _get_instance_class(cls):
        from .a2a_instance import A2AAgentInstance

        return A2AAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.a2a.a2a_instance:A2AAgentInstance"

    def _get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "url": self.url,
            "timeout": self.timeout,
        }

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return "a2a"

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return ["a2a"]
