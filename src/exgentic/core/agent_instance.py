# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..observers.logging import get_logger
from ..utils.cost import CostReport
from ..utils.paths import SessionPaths
from .types import Action, ActionType, Observation


class AgentInstance(ABC):
    """Agent instance - handles decision making for one task execution."""

    max_steps: int | None = None

    def __init__(self, session_id: str) -> None:
        """Create a new agent bound to a specific session.

        The session id is the single source of truth for scoping all agent-side
        logs and artifacts under `outputs/<run_id>/sessions/<session_id>/agent/`.
        """
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        """Generates a unique id for the agent."""
        if not hasattr(self, "_agent_id"):
            self._agent_id = str(uuid.uuid4()).replace("-", "_")
        return self._agent_id

    @property
    def paths(self) -> SessionPaths:
        """All filesystem paths for this session."""
        if not hasattr(self, "_paths"):
            from .context import try_get_context

            ctx = try_get_context()
            if ctx is not None:
                self._paths = SessionPaths(
                    session_id=self.session_id,
                    run_id=ctx.run_id,
                    output_dir=ctx.output_dir,
                )
            else:
                self._paths = SessionPaths(session_id=self.session_id, run_id="default", output_dir="outputs")
        return self._paths

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = get_logger(f"Agent_{self.agent_id}", str(self.paths.agent_log))
        return self._logger

    def get_cost(self) -> CostReport:
        """Estimated monetary cost; default 0.0."""
        return CostReport.initialize_empty()

    @abstractmethod
    def react(self, observation: Optional[Observation]) -> Optional[Action]:
        """React to observation - agent controls decision making, None = done."""
        pass

    def start(self, task: str, context: dict[str, Any], actions: list[ActionType]):
        """Receive the work payload and start the agent.

        Called via HTTP transport after the instance is constructed, so
        large payloads (e.g. dozens of ActionTypes) are never serialized
        as CLI arguments.
        """
        self.task = task
        self.context = context or {}
        self.actions = actions

    @abstractmethod
    def close(self) -> None:
        """Cleanup agent resources - agent manages its own state."""
        pass
