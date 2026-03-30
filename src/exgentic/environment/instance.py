# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared EnvironmentManager instance."""

from __future__ import annotations

from pathlib import Path

from .manager import EnvironmentManager


def get_manager() -> EnvironmentManager:
    """Return the shared EnvironmentManager instance.

    Environments live at ``~/.exgentic/`` — a fixed absolute path,
    independent of the working directory or cache settings.
    """
    return EnvironmentManager(base_dir=Path.home() / ".exgentic")
