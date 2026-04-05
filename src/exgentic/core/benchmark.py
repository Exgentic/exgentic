# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ..utils.settings import RunnerName
from .runner_mixin import RunnerMixin

if TYPE_CHECKING:
    from .evaluator import Evaluator
    from .session import Session


class Benchmark(BaseModel, RunnerMixin, ABC):
    """Benchmark configuration — lightweight config that lives on the host.

    Callers use ``get_evaluator()`` and ``get_session()`` to obtain
    instances wrapped in the configured runner for container isolation.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    subset: str | None = None
    seed: int = 300
    runner: RunnerName | None = None
    use_cache: bool = True
    max_interactions: int | None = 150
    docker_socket: bool = False

    @property
    def subset_name(self) -> str:
        """Stable subset identifier for this benchmark run."""
        return str(self.subset) if self.subset else "unknown"

    def list_subsets(self) -> list[str]:
        """Return available subset identifiers for this benchmark."""
        subset = self.subset_name
        return [subset] if subset and subset != "unknown" else []

    @classmethod
    def _get_evaluator_class(cls) -> type[Evaluator]:
        """Return the Evaluator subclass for this benchmark.

        Subclasses implement this with a lazy import so heavy deps
        are only loaded inside the runner.
        """
        raise NotImplementedError

    @classmethod
    def _get_session_class(cls) -> type[Session]:
        """Return the Session subclass for this benchmark.

        Subclasses implement this with a lazy import so heavy deps
        are only loaded inside the runner.
        """
        raise NotImplementedError

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        """Return kwargs for constructing the Evaluator.

        Subclasses override this to pass benchmark-specific config.
        """
        return {}

    def get_evaluator(self) -> Evaluator:
        """Create an ``Evaluator`` wrapped in the configured runner."""
        from ..adapters.runners import with_runner

        return with_runner(
            self._get_evaluator_class(),
            runner=self.resolve_runner(),
            **self._get_evaluator_kwargs(),
            **self.runner_kwargs(),
        )

    def get_session(self, **session_kwargs: Any) -> Session:
        """Create a ``Session`` wrapped in the configured runner."""
        from ..adapters.runners import with_runner

        return with_runner(
            self._get_session_class(),
            runner=self.resolve_runner(),
            **session_kwargs,
            **self.runner_kwargs(),
        )
