# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from typing import Any, Dict, List, Optional

from exgentic.core.agent import Agent
from exgentic.core.agent_instance import AgentInstance
from exgentic.core.types import ModelSettings


class SimpleTestAgentInstance(AgentInstance):
    """Simple test agent that responds with basic actions."""

    def __init__(self, session_id: str, task: str, context: Dict[str, Any], actions: List[str]):
        super().__init__(session_id)
        self.task = task
        self.context = context or {}
        self.actions = actions
        self.step_count = 0

    def react(self, observation: Optional[str]) -> Optional[str]:
        """React to observation with simple response."""
        self.step_count += 1

        if observation is None:
            # First step
            return f"Starting task: {self.task}"

        # Simple logic: respond a few times then finish
        if self.step_count <= 2:
            return f"Responding to: {observation}"
        # Signal completion
        return None

    def close(self):
        pass


class SimpleTestAgent(Agent):
    """Agent factory that creates simple test agents."""

    display_name: str = "Simple Test Agent"
    slug_name: str = "simple_test"

    def __init__(self, model_settings: ModelSettings | None = None) -> None:
        if model_settings is not None and not isinstance(model_settings, ModelSettings):
            raise ValueError("model_settings must be a ModelSettings instance.")
        self.model_settings = model_settings

    def assign(self, task: str, context: Dict[str, Any], actions: List[str], session_id: str) -> AgentInstance:
        return SimpleTestAgentInstance(session_id, task, context, actions)

    def get_models_names(self) -> List[str]:  # type: ignore[override]
        return []
