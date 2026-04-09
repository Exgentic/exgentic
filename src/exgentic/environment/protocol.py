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
            **kwargs: Common options forwarded by the manager:
                ``project_root`` (Path | None) - root of a Python project to install.
                ``packages`` (list[str] | None) - extra pip packages to install.
                Backends may also receive backend-specific options (e.g. ``name``,
                ``force`` for Docker).

        Returns:
            Extra marker data to persist (may be empty).
        """
        ...

    def exists(self, env_dir: Path, marker_data: dict) -> bool:
        """Check whether the environment artifacts actually exist on disk.

        Args:
            env_dir: Root directory for this environment.
            marker_data: Data previously stored in the marker file.

        Returns:
            True if the environment is usable, False if it needs reinstalling.
        """
        ...

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """Tear down the environment.

        Args:
            env_dir: Root directory for this environment.
            marker_data: Data previously stored in the marker file.
        """
        ...
