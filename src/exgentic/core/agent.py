# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

from pydantic import BaseModel, ConfigDict

from .agent_instance import AgentInstance
from .runner_mixin import RunnerMixin
from .types import ActionType
from .types.model_settings import ModelSettings
from ..utils.settings import RunnerName


class Agent(BaseModel, RunnerMixin, ABC):
    """Agent factory - creates AgentInstance objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    display_name: ClassVar[str]
    slug_name: ClassVar[str]
    model_settings: ModelSettings | None = None
    runner: RunnerName | None = None
    docker_socket: bool = False

    @abstractmethod
    def assign(
        self,
        task: str,
        context: Dict[str, Any],
        actions: List[ActionType],
        session_id: str,
    ) -> AgentInstance:
        """Create agent for specific task - agent factory controls instance creation."""
        pass

    @classmethod
    def setup(cls) -> None:
        """Override to perform non-pip setup (e.g. Docker build, npm install).

        Called by ``exgentic setup --agent <slug>`` after deps are installed.
        """

    # Optional metadata property for dashboard/leaderboards
    @property
    def model_name(self) -> str:
        return "unknown"

    def get_models_names(self) -> List[str]:
        name = self.model_name
        if not name or name == "unknown":
            return []
        return [name]
