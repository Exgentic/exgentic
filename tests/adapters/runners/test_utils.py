# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for :mod:`exgentic.adapters.runners._utils`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from exgentic.adapters.runners._utils import find_project_root


def test_find_project_root_returns_repo_root():
    """When a pyproject.toml exists in a parent, return that directory."""
    root = find_project_root()
    assert (root / "pyproject.toml").exists()


def test_find_project_root_falls_back_to_dot_exgentic(tmp_path: Path):
    """When no pyproject.toml is found, fall back to ~/.exgentic/."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    # Create a fake __file__ path with no pyproject.toml in any parent
    fake_file = tmp_path / "lib" / "pkg" / "mod.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()

    with (
        patch(
            "exgentic.adapters.runners._utils.Path.__file__",
            create=True,
        ),
        patch(
            "exgentic.adapters.runners._utils.Path.home",
            return_value=fake_home,
        ),
    ):
        # Patch __file__ at the module level so Path(__file__) resolves
        # to a location without pyproject.toml in any ancestor.
        import exgentic.adapters.runners._utils as mod

        original_file = mod.__file__
        try:
            mod.__file__ = str(fake_file)
            result = find_project_root()
        finally:
            mod.__file__ = original_file

    expected = fake_home / ".exgentic"
    assert result == expected
    assert expected.is_dir()


def test_find_project_root_fallback_is_idempotent(tmp_path: Path):
    """Calling find_project_root twice with fallback doesn't error."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    fake_file = tmp_path / "lib" / "mod.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()

    with patch(
        "exgentic.adapters.runners._utils.Path.home",
        return_value=fake_home,
    ):
        import exgentic.adapters.runners._utils as mod

        original_file = mod.__file__
        try:
            mod.__file__ = str(fake_file)
            result1 = find_project_root()
            result2 = find_project_root()
        finally:
            mod.__file__ = original_file

    assert result1 == result2
    assert result1 == fake_home / ".exgentic"
