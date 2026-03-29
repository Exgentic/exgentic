# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Common protocol for environment backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class EnvironmentBackend(Protocol):
    """Uniform interface every environment backend must satisfy."""

    def install(self, env_dir: Path, *, module_path: str | None = None, **kwargs: object) -> dict:
        """Create / set up the environment.

        Args:
            env_dir: Root directory for this environment.
            module_path: Dotted module path for locating package resources.
            **kwargs: Backend-specific options (e.g. ``venv_packages``).

        Returns:
            Extra marker data to persist (may be empty).
        """
        ...

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """Tear down the environment.

        Args:
            env_dir: Root directory for this environment.
            marker_data: Data previously stored in the marker file.
        """
        ...
