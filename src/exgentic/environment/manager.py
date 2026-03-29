# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Environment manager: orchestrates install, uninstall, and queries."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from .docker import DockerBackend
from .local import LocalBackend
from .protocol import EnvironmentBackend
from .venv import VenvBackend


class EnvType(StrEnum):
    """Supported environment types."""

    VENV = "venv"
    LOCAL = "local"
    DOCKER = "docker"


class EnvironmentManager:
    """Manages isolated environments identified by name.

    Each environment lives at ``{base_dir}/{name}/`` and can have
    multiple environment types (venv, local, docker) installed
    simultaneously.  The ``.installed`` marker file tracks which
    types are present as a JSON dict keyed by :class:`EnvType`.
    """

    MARKER_FILE = ".installed"

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.home() / ".exgentic"
        self._backends: dict[EnvType, EnvironmentBackend] = {
            EnvType.VENV: VenvBackend(),
            EnvType.LOCAL: LocalBackend(),
            EnvType.DOCKER: DockerBackend(),
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def install(
        self,
        name: str,
        *,
        env_type: EnvType = EnvType.VENV,
        force: bool = False,
        module_path: str | None = None,
        venv_packages: list[str] | None = None,
    ) -> Path:
        """Install an environment.

        Args:
            name: Environment name (e.g. ``"benchmarks/tau2"``).
            env_type: Type of environment to create.
            force: Re-create even if already installed.
            module_path: Dotted module path for locating package resources.
            venv_packages: Extra packages to install (venv only).

        Returns:
            The environment directory path.
        """
        env_type = EnvType(env_type)
        if not force and self.is_installed(name, env_type=env_type):
            return self.env_path(name)

        env_dir = self.env_path(name)
        env_dir.mkdir(parents=True, exist_ok=True)
        self._remove_marker_entry(name, env_type)

        backend = self._backends[env_type]

        # Build kwargs for the backend.
        kwargs: dict[str, object] = {}
        if venv_packages is not None:
            kwargs["venv_packages"] = venv_packages
        if env_type is EnvType.DOCKER:
            kwargs["name"] = name
            kwargs["force"] = force

        extra = backend.install(env_dir, module_path=module_path, **kwargs)
        self._add_marker_entry(name, env_type, {"installed_at": _now_iso(), **extra})

        return env_dir

    def uninstall(self, name: str, *, env_type: EnvType | None = None) -> None:
        """Remove an installed environment.

        Args:
            name: Environment name.
            env_type: Specific type to remove, or *None* to remove all.
        """
        if env_type is not None:
            env_type = EnvType(env_type)

        env_dir = self.env_path(name)
        if not env_dir.exists():
            return

        marker = self._read_marker(name)

        if env_type is None:
            # Remove all env types, then the whole directory.
            for et in EnvType:
                if et in marker:
                    self._backends[et].uninstall(env_dir, marker.get(et, {}))
            shutil.rmtree(env_dir)
            return

        # Remove a single env type.
        self._backends[env_type].uninstall(env_dir, marker.get(env_type, {}))
        self._remove_marker_entry(name, env_type)

        # Clean up directory if no env types remain.
        if not self._read_marker(name) and env_dir.exists():
            shutil.rmtree(env_dir)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_installed(self, name: str, *, env_type: EnvType | None = None) -> bool:
        """Check if an environment is installed.

        Args:
            name: Environment name.
            env_type: Check a specific type, or *None* for any.
        """
        if env_type is not None:
            env_type = EnvType(env_type)
        marker = self._read_marker(name)
        if env_type is None:
            return bool(marker)
        return env_type in marker

    def get_info(self, name: str) -> dict | None:
        """Return installation info or *None* if not installed."""
        marker = self._read_marker(name)
        if not marker:
            return None
        return {"name": name, "environments": marker}

    def list_installed(self) -> list[dict]:
        """List all installed environments with details."""
        result: list[dict] = []
        if not self.base_dir.is_dir():
            return result
        for child in sorted(self.base_dir.rglob(self.MARKER_FILE)):
            try:
                marker = json.loads(child.read_text())
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(marker, dict) and marker:
                name = str(child.parent.relative_to(self.base_dir))
                result.append({"name": name, "environments": marker})
        return result

    # ------------------------------------------------------------------
    # Paths & accessors
    # ------------------------------------------------------------------

    def env_path(self, name: str) -> Path:
        """Return the environment directory path."""
        return self.base_dir / name

    def venv_python(self, name: str) -> str:
        """Return the path to the venv Python binary."""
        return str(self.env_path(name) / "venv" / "bin" / "python")

    def docker_image(self, name: str) -> str | None:
        """Return the Docker image tag, or *None* if not installed."""
        return self._read_marker(name).get(EnvType.DOCKER, {}).get("image")

    def local_python(self, name: str) -> str | None:
        """Return the Python path used for local install, or *None*."""
        return self._read_marker(name).get(EnvType.LOCAL, {}).get("python")

    # ------------------------------------------------------------------
    # Marker management
    # ------------------------------------------------------------------

    def _read_marker(self, name: str) -> dict:
        marker = self.env_path(name) / self.MARKER_FILE
        if not marker.is_file():
            return {}
        try:
            data = json.loads(marker.read_text())
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def _write_marker(self, name: str, data: dict) -> None:
        env_dir = self.env_path(name)
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / self.MARKER_FILE).write_text(json.dumps(data, indent=2))

    def _add_marker_entry(self, name: str, env_type: EnvType, info: dict) -> None:
        data = self._read_marker(name)
        data[env_type] = info
        self._write_marker(name, data)

    def _remove_marker_entry(self, name: str, env_type: EnvType) -> None:
        data = self._read_marker(name)
        data.pop(env_type, None)
        if data:
            self._write_marker(name, data)
        else:
            marker = self.env_path(name) / self.MARKER_FILE
            if marker.exists():
                marker.unlink()


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
