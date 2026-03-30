# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Local environment: installs dependencies into the current Python."""

from __future__ import annotations

import sys
from pathlib import Path

from .helpers import (
    build_subprocess_env,
    install_packages,
    install_project,
    install_requirements,
    require_uv,
    run_setup_sh,
    validate_system_deps,
)


class LocalBackend:
    """Backend that installs dependencies into the current Python."""

    def install(
        self,
        env_dir: Path,
        *,
        module_path: str | None = None,
        **kwargs: object,
    ) -> dict:
        """Install dependencies into the current Python environment.

        Args:
            env_dir: Root directory for data/markers.
            module_path: Dotted module path for locating package resources.
            **kwargs: Accepts ``project_root`` (Path) and ``packages`` (list).

        Returns:
            Extra marker data (``python`` path).
        """
        project_root: Path | None = kwargs.get("project_root")  # type: ignore[assignment]
        packages: list[str] | None = kwargs.get("packages")  # type: ignore[assignment]

        if project_root is not None or packages or module_path is not None:
            uv = require_uv()
            env = build_subprocess_env()
            # Ensure VIRTUAL_ENV is set so uv installs into the correct
            # environment when the current Python lives inside a venv
            # (e.g. when exgentic is installed as a uv tool).
            if sys.prefix != sys.base_prefix:
                env["VIRTUAL_ENV"] = sys.prefix

            if project_root is not None:
                install_project(uv, sys.executable, project_root, env)

            if packages:
                install_packages(uv, sys.executable, packages, env)

            if module_path is not None:
                install_requirements(uv, sys.executable, module_path, env)
                validate_system_deps(module_path)
                run_setup_sh(module_path, env_dir)

        return {"python": sys.executable}

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """No-op. Cannot remove deps from the current Python environment."""
