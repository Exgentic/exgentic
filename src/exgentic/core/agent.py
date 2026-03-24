# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..utils.settings import RunnerName
from .runner_mixin import RunnerMixin
from .types.model_settings import ModelSettings

if TYPE_CHECKING:
    from .agent_instance import AgentInstance


class Agent(BaseModel, RunnerMixin, ABC):
    """Agent configuration — lightweight config that lives on the host.

    Provides ``get_instance_class()`` to lazily resolve the execution class,
    which can be wrapped with ``with_runner()`` for container/venv isolation,
    mirroring how ``Benchmark`` provides ``get_session_class()`` and
    ``get_evaluator_class()``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    display_name: ClassVar[str]
    slug_name: ClassVar[str]
    model_settings: ModelSettings | None = None
    runner: RunnerName | None = None
    docker_socket: bool = False

    @classmethod
    @abstractmethod
    def get_instance_class(cls) -> type[AgentInstance]:
        """Return the AgentInstance subclass for this agent.

        Subclasses implement this with a lazy import so heavy deps
        (litellm, smolagents, …) are only loaded inside the runner.
        """
        ...

    @classmethod
    def get_instance_class_ref(cls) -> str:
        """Return a ``"module:qualname"`` string for the instance class.

        By default calls ``get_instance_class()`` and converts to string.
        Override in subclasses whose instance module has heavy third-party
        imports to return the string directly without triggering the import.
        """
        klass = cls.get_instance_class()
        return f"{klass.__module__}:{klass.__qualname__}"

    @abstractmethod
    def get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Return kwargs for creating the instance class.

        Task, context, and actions are passed separately via
        ``AgentInstance.start()`` (through HTTP transport) to avoid
        OS argument-list size limits.
        """
        ...

    @classmethod
    def setup(cls) -> None:
        """Override to perform non-pip setup (e.g. Docker build, npm install).

        Called by ``exgentic setup --agent <slug>`` after deps are installed.
        """

    # Optional metadata property for dashboard/leaderboards
    @property
    def model_name(self) -> str:
        return "unknown"

    def get_models_names(self) -> list[str]:
        name = self.model_name
        if not name or name == "unknown":
            return []
        return [name]
