# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from abc import ABC
from typing import Any, ClassVar, Dict, List, Type

from pydantic import BaseModel, ConfigDict

from ..utils.settings import RunnerName
from .evaluator import Evaluator
from .runner_mixin import RunnerMixin
from .session import Session


class Benchmark(BaseModel, RunnerMixin, ABC):
    """Benchmark configuration — lightweight config that lives on the host.

    Points to an ``evaluator_class`` (task discovery & aggregation) and
    a ``session_class`` (task execution). Both can be wrapped with
    ``with_runner()`` for container isolation.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    # Subclasses must set these class variables.
    evaluator_class: ClassVar[Type[Evaluator]]
    session_class: ClassVar[Type[Session]]

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

    def list_subsets(self) -> List[str]:
        """Return available subset identifiers for this benchmark."""
        subset = self.subset_name
        return [subset] if subset and subset != "unknown" else []

    @classmethod
    def setup(cls) -> None:
        """Override to download data or perform non-pip setup.

        Called by ``exgentic setup --benchmark <slug>`` after deps are installed.
        Use ``settings.resolve_cache_path() / "<slug>"`` for data storage.
        """

    def get_evaluator_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for constructing the Evaluator.

        Subclasses override this to pass benchmark-specific config.
        """
        return {}
