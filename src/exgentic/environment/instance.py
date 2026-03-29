# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared EnvironmentManager instance."""

from __future__ import annotations

from pathlib import Path

from .manager import EnvironmentManager


def get_manager() -> EnvironmentManager:
    """Return the shared EnvironmentManager instance.

    Uses the exgentic settings cache_dir as the base directory.
    """
    from ..utils.settings import get_settings

    return EnvironmentManager(base_dir=Path(get_settings().cache_dir).expanduser())
