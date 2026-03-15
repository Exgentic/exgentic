# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import inspect
from abc import ABC
from pathlib import Path
from typing import ClassVar, Dict, Any, List, Type

from pydantic import BaseModel, ConfigDict

from ..utils.paths import SessionPaths, get_run_paths
from ..utils.settings import ExecuterName, RunnerName, get_settings

from .session import Session
from .evaluator import Evaluator
from .types import SessionIndex


class Benchmark(BaseModel, ABC):
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
    executer: ExecuterName | None = None
    runner: RunnerName | None = None
    use_cache: bool = True
    max_interactions: int | None = 150

    @property
    def subset_name(self) -> str:
        """Stable subset identifier for this benchmark run."""
        return str(self.subset) if self.subset else "unknown"

    def list_subsets(self) -> List[str]:
        """Return available subset identifiers for this benchmark."""
        subset = self.subset_name
        return [subset] if subset and subset != "unknown" else []

    def get_evaluator_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for constructing the Evaluator.

        Subclasses override this to pass benchmark-specific config.
        """
        return {}

    _EXECUTER_TO_RUNNER: dict[str, RunnerName] = {
        "inprocess": "direct",
        "remote_process": "process",
    }

    def resolve_runner(self) -> RunnerName:
        """Resolve the runner name from ``runner``, ``executer``, or settings."""
        if self.runner is not None:
            return self.runner
        if self.executer is not None:
            return self._EXECUTER_TO_RUNNER.get(self.executer, "process")  # type: ignore[arg-type]
        return get_settings().default_runner

    @property
    def setup_script(self) -> str | None:
        """Auto-discover ``setup.sh`` next to the benchmark module.

        Subclasses can override this to point to a different script.
        """
        module_dir = Path(inspect.getfile(type(self))).parent
        script = module_dir / "setup.sh"
        return str(script) if script.exists() else None

    # Override in subclasses that need Docker socket access (e.g. SWE-bench).
    docker_socket: bool = False

    def runner_kwargs(self) -> Dict[str, Any]:
        """Return extra kwargs for ``with_runner()`` based on runner type.

        When the runner is ``"docker"``, this includes the setup script,
        docker socket, and output volume mount. For other runners these
        are ignored.
        """
        runner = self.resolve_runner()
        if runner != "docker":
            return {}
        kw: Dict[str, Any] = {}
        if self.setup_script:
            kw["setup_script"] = self.setup_script
        if self.docker_socket:
            kw["docker_socket"] = True
        # Mount the output directory so results are visible on the host.
        output_dir = str(Path(get_settings().output_dir).resolve())
        kw["volumes"] = {output_dir: output_dir}
        return kw

    def close(self) -> None:
        """Optional cleanup hook."""
        return None
