# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Local environment: installs dependencies into the current Python."""

from __future__ import annotations

import sys
from pathlib import Path

from ._helpers import (
    build_subprocess_env,
    install_requirements,
    require_uv,
    run_setup_sh,
    validate_system_deps,
)


def install(
    env_dir: Path,
    *,
    module_path: str | None = None,
) -> dict:
    """Install dependencies into the current Python environment.

    Args:
        env_dir: Root directory for data/markers.
        module_path: Dotted module path for locating package resources.

    Returns:
        Extra marker data (``python`` path).
    """
    if module_path is not None:
        uv = require_uv()
        env = build_subprocess_env()
        install_requirements(uv, sys.executable, module_path, env)
        validate_system_deps(module_path)
        run_setup_sh(module_path, env_dir)

    return {"python": sys.executable}


def uninstall() -> None:
    """No-op. Cannot remove deps from the current Python environment."""
