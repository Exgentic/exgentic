# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A Agent — configuration for consuming an external A2A-speaking agent
as an Exgentic benchmark participant.

This is the *complement* of the a2a_executor (PR #187) which *exposes*
Exgentic agents as A2A endpoints.  Here we go the other direction:
an external A2A agent is wrapped so that Exgentic can drive it through
any benchmark.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...core.agent import Agent


class A2AAgent(Agent):
    """Agent factory that wraps an external A2A-speaking agent.

    All A2A transports (JSON-RPC, gRPC, REST) are supported — the SDK
    auto-negotiates based on the remote agent's advertised interfaces.
    """

    display_name: ClassVar[str] = "A2A Agent"
    slug_name: ClassVar[str] = "a2a"

    agent_url: str
    """URL of the A2A agent endpoint (e.g. ``http://localhost:8080``)."""

    max_steps: int = 150

    @classmethod
    def _get_instance_class(cls):
        from ...adapters.agents.a2a_agent import A2AAgentInstance

        return A2AAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.adapters.agents.a2a_agent:A2AAgentInstance"

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return "a2a-external"

    def _get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "agent_url": self.agent_url,
            "max_steps": self.max_steps,
        }
