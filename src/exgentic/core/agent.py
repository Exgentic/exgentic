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

    Callers use ``get_instance(session_id)`` to obtain a running
    ``AgentInstance`` wrapped in the configured runner, mirroring
    ``Benchmark.get_evaluator()`` and ``Benchmark.get_session()``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    display_name: ClassVar[str]
    slug_name: ClassVar[str]
    model_settings: ModelSettings | None = None
    runner: RunnerName | None = None
    docker_socket: bool = False

    @classmethod
    @abstractmethod
    def _get_instance_class(cls) -> type[AgentInstance]:
        """Return the AgentInstance subclass for this agent.

        Subclasses implement this with a lazy import so heavy deps
        (litellm, smolagents, …) are only loaded inside the runner.
        """
        ...

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        """Return a ``"module:qualname"`` string for the instance class.

        By default calls ``_get_instance_class()`` and converts to string.
        Override in subclasses whose instance module has heavy third-party
        imports to return the string directly without triggering the import.
        """
        klass = cls._get_instance_class()
        return f"{klass.__module__}:{klass.__qualname__}"

    @abstractmethod
    def _get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Return kwargs for creating the instance class.

        Task, context, and actions are passed separately via
        ``AgentInstance.start()`` (through HTTP transport) to avoid
        OS argument-list size limits.
        """
        ...

    def get_instance(self, session_id: str) -> AgentInstance:
        """Create an ``AgentInstance`` wrapped in the configured runner."""
        from ..adapters.runners import with_runner

        return with_runner(
            self._get_instance_class_ref(),
            runner=self.resolve_runner(),
            **self._get_instance_kwargs(session_id=session_id),
            **self.runner_kwargs(),
        )

    # Optional metadata property for dashboard/leaderboards
    @property
    def model_name(self) -> str:
        return "unknown"

    def get_models_names(self) -> list[str]:
        name = self.model_name
        if not name or name == "unknown":
            return []
        return [name]
