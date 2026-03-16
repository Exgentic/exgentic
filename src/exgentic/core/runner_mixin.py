# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import inspect
from pathlib import Path
from typing import Any, Dict

from ..utils.settings import RunnerName, get_settings


class RunnerMixin:
    """Shared runner/Docker logic for Agent and Benchmark."""

    runner: RunnerName | None
    docker_socket: bool

    def resolve_runner(self) -> RunnerName:
        """Resolve the runner name from ``runner`` field or settings default."""
        if self.runner is not None:
            return self.runner
        return get_settings().default_runner

    @property
    def setup_script(self) -> str | None:
        """Auto-discover ``setup.sh`` next to the module."""
        module_dir = Path(inspect.getfile(type(self))).parent
        script = module_dir / "setup.sh"
        return str(script) if script.exists() else None

    @property
    def requirements_txt(self) -> str | None:
        """Auto-discover ``requirements.txt`` next to the module."""
        module_dir = Path(inspect.getfile(type(self))).parent
        # Walk up to find requirements.txt (handles nested dirs like agents/cli/claude/)
        while module_dir.name != "exgentic":
            req = module_dir / "requirements.txt"
            if req.exists():
                return str(req)
            module_dir = module_dir.parent
        return None

    def runner_kwargs(self) -> Dict[str, Any]:
        """Return extra kwargs for ``with_runner()`` when runner is docker."""
        runner = self.resolve_runner()
        if runner != "docker":
            return {}
        kw: Dict[str, Any] = {}
        if self.setup_script:
            kw["setup_script"] = self.setup_script
        if self.docker_socket:
            kw["docker_socket"] = True
        if self.requirements_txt:
            kw["requirements_txt"] = self.requirements_txt
        output_dir = str(Path(get_settings().output_dir).resolve())
        kw["volumes"] = {output_dir: output_dir}
        return kw

    def close(self) -> None:
        """Optional cleanup hook."""
        return
