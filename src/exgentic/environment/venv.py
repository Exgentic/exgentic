# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Venv environment: creates an isolated Python venv with dependencies."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .helpers import (
    build_subprocess_env,
    install_packages,
    install_requirements,
    require_uv,
    run_setup_sh,
    validate_system_deps,
)


class VenvBackend:
    """Backend that creates an isolated Python venv with dependencies."""

    def install(
        self,
        env_dir: Path,
        *,
        module_path: str | None = None,
        **kwargs: object,
    ) -> dict:
        """Create a venv-based environment.

        Args:
            env_dir: Root directory for this environment.
            module_path: Dotted module path for locating package resources.
            **kwargs: Accepts ``venv_packages`` (list of extra packages).

        Returns:
            Empty dict (no extra marker data).
        """
        venv_packages: list[str] | None = kwargs.get("venv_packages")  # type: ignore[assignment]

        venv_dir = env_dir / "venv"
        if venv_dir.exists():
            shutil.rmtree(venv_dir)

        try:
            uv = require_uv()

            subprocess.run(
                [uv, "venv", str(venv_dir), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
                check=True,
                capture_output=True,
                text=True,
            )

            venv_py = str(venv_dir / "bin" / "python")
            env = build_subprocess_env()

            if venv_packages:
                install_packages(uv, venv_py, venv_packages, env)

            if module_path is not None:
                install_requirements(uv, venv_py, module_path, env)
                validate_system_deps(module_path)
                run_setup_sh(module_path, env_dir, venv_dir=venv_dir)
        except BaseException:
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
            raise

        return {}

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """Remove the venv directory."""
        venv_dir = env_dir / "venv"
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
