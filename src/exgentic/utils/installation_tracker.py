# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Track installed agents and benchmarks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal


def get_installations_dir() -> Path:
    """Get the directory where installation records are stored."""
    # Store in project directory under .exgentic_installations
    # Find the project root by looking for pyproject.toml
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            installations_dir = current / ".exgentic_installations"
            installations_dir.mkdir(parents=True, exist_ok=True)
            return installations_dir
        current = current.parent

    # Fallback for uvx / non-project environments
    installations_dir = Path.home() / ".cache" / "exgentic" / ".installations"
    installations_dir.mkdir(parents=True, exist_ok=True)
    return installations_dir


def _unified_install_file(name: str, install_type: Literal["agent", "benchmark"]) -> Path:
    """Return the unified installation marker path under the cache dir."""
    from .cache import benchmark_cache_dir

    cache = benchmark_cache_dir(name)
    cache.mkdir(parents=True, exist_ok=True)
    return cache / "installations.json"


def record_installation(
    name: str,
    install_type: Literal["agent", "benchmark"],
) -> None:
    """Record that an agent or benchmark was installed.

    Writes to the unified location (``{cache_dir}/{name}/installations.json``)
    and to the legacy location for backwards compatibility.

    Args:
        name: The slug name of the agent or benchmark
        install_type: Either "agent" or "benchmark"
    """
    install_data = {
        "name": name,
        "type": install_type,
        "installed_at": datetime.utcnow().isoformat() + "Z",
    }
    payload = json.dumps(install_data, indent=2)

    # Write to unified location.
    _unified_install_file(name, install_type).write_text(payload, encoding="utf-8")

    # Write to legacy location for backwards compatibility.
    installations_dir = get_installations_dir()
    type_dir = installations_dir / install_type
    type_dir.mkdir(exist_ok=True)
    (type_dir / f"{name}.json").write_text(payload, encoding="utf-8")


def get_installation_info(
    name: str,
    install_type: Literal["agent", "benchmark"],
) -> dict[str, str] | None:
    """Get installation information for an agent or benchmark.

    Checks the unified location first, then the legacy location.

    Args:
        name: The slug name of the agent or benchmark
        install_type: Either "agent" or "benchmark"

    Returns:
        Dictionary with installation info or None if not installed
    """
    # Check unified location first.
    unified = _unified_install_file(name, install_type)
    if unified.exists():
        try:
            return json.loads(unified.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to legacy location.
    installations_dir = get_installations_dir()
    install_file = installations_dir / install_type / f"{name}.json"

    if not install_file.exists():
        return None

    try:
        return json.loads(install_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_installed(
    name: str,
    install_type: Literal["agent", "benchmark"],
) -> bool:
    """Check if an agent or benchmark is installed.

    Args:
        name: The slug name of the agent or benchmark
        install_type: Either "agent" or "benchmark"

    Returns:
        True if installed, False otherwise
    """
    return get_installation_info(name, install_type) is not None


def get_all_installations(
    install_type: Literal["agent", "benchmark"],
) -> dict[str, dict[str, str]]:
    """Get all installation records for a given type.

    Scans both the unified cache directory and the legacy location.

    Args:
        install_type: Either "agent" or "benchmark"

    Returns:
        Dictionary mapping names to installation info
    """
    installations: dict[str, dict[str, str]] = {}

    # Scan legacy location.
    installations_dir = get_installations_dir()
    type_dir = installations_dir / install_type
    if type_dir.exists():
        for install_file in type_dir.glob("*.json"):
            try:
                data = json.loads(install_file.read_text(encoding="utf-8"))
                name = data.get("name")
                if name:
                    installations[name] = data
            except (json.JSONDecodeError, OSError):
                continue

    # Scan unified location (overrides legacy if both exist).
    from .settings import get_settings

    cache_root = Path(get_settings().cache_dir).expanduser()
    if cache_root.exists():
        for slug_dir in cache_root.iterdir():
            unified = slug_dir / "installations.json"
            if unified.exists():
                try:
                    data = json.loads(unified.read_text(encoding="utf-8"))
                    name = data.get("name")
                    if name and data.get("type") == install_type:
                        installations[name] = data
                except (json.JSONDecodeError, OSError):
                    continue

    return installations


__all__ = [
    "get_all_installations",
    "get_installation_info",
    "get_installations_dir",
    "is_installed",
    "record_installation",
]

# Made with Bob
