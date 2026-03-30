# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from pathlib import Path
from typing import Any

from ..utils.settings import RunnerName, get_settings
from .context import try_get_context


class RunnerMixin:
    """Shared runner/Docker logic for Agent and Benchmark."""

    runner: RunnerName | None
    docker_socket: bool
    slug_name: str

    def resolve_runner(self) -> RunnerName:
        """Resolve the runner name from ``runner`` field or settings default."""
        if self.runner is not None:
            return self.runner
        return get_settings().default_runner

    def runner_kwargs(self) -> dict[str, Any]:
        """Return extra kwargs for ``with_runner()`` when runner is docker or venv."""
        runner = self.resolve_runner()

        kind = "agents" if self._is_agent() else "benchmarks"

        if runner == "venv":
            return {
                "env_name": f"{kind}/{self.slug_name}",
                "module_path": type(self).__module__,
            }

        if runner != "docker":
            return {}

        kw: dict[str, Any] = {
            "env_name": f"{kind}/{self.slug_name}",
            "module_path": type(self).__module__,
        }
        if self.docker_socket:
            kw["docker_socket"] = True
        ctx = try_get_context()
        output_dir = ctx.output_dir if ctx is not None else get_settings().output_dir
        output_dir = str(Path(output_dir).resolve())
        kw["volumes"] = {output_dir: output_dir}
        return kw

    def _is_agent(self) -> bool:
        """Return True if this instance is an Agent (not a Benchmark)."""
        from .agent import Agent

        return isinstance(self, Agent)

    def close(self) -> None:
        """Optional cleanup hook."""
        return
