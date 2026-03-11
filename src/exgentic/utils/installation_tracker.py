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

    # Fallback to current directory if pyproject.toml not found
    installations_dir = Path.cwd() / ".exgentic_installations"
    installations_dir.mkdir(parents=True, exist_ok=True)
    return installations_dir


def record_installation(
    name: str,
    install_type: Literal["agent", "benchmark"],
) -> None:
    """Record that an agent or benchmark was installed.

    Args:
        name: The slug name of the agent or benchmark
        install_type: Either "agent" or "benchmark"
    """
    installations_dir = get_installations_dir()
    type_dir = installations_dir / install_type
    type_dir.mkdir(exist_ok=True)

    install_file = type_dir / f"{name}.json"
    install_data = {
        "name": name,
        "type": install_type,
        "installed_at": datetime.utcnow().isoformat() + "Z",
    }

    install_file.write_text(json.dumps(install_data, indent=2), encoding="utf-8")


def get_installation_info(
    name: str,
    install_type: Literal["agent", "benchmark"],
) -> dict[str, str] | None:
    """Get installation information for an agent or benchmark.

    Args:
        name: The slug name of the agent or benchmark
        install_type: Either "agent" or "benchmark"

    Returns:
        Dictionary with installation info or None if not installed
    """
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

    Args:
        install_type: Either "agent" or "benchmark"

    Returns:
        Dictionary mapping names to installation info
    """
    installations_dir = get_installations_dir()
    type_dir = installations_dir / install_type

    if not type_dir.exists():
        return {}

    installations = {}
    for install_file in type_dir.glob("*.json"):
        try:
            data = json.loads(install_file.read_text(encoding="utf-8"))
            name = data.get("name")
            if name:
                installations[name] = data
        except (json.JSONDecodeError, OSError):
            continue

    return installations


__all__ = [
    "record_installation",
    "get_installation_info",
    "is_installed",
    "get_all_installations",
    "get_installations_dir",
]

# Made with Bob
