# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for :mod:`exgentic.adapters.runners._utils`."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from exgentic.adapters.runners._utils import find_project_root, prepare_subprocess_env


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


class TestPrepareSubprocessEnv:
    """Tests for prepare_subprocess_env env-var forwarding."""

    def test_forwards_pythonpath(self):
        """PYTHONPATH must be forwarded for CE-Manager bootstrap."""
        with patch.dict(
            os.environ,
            {"PYTHONPATH": "/some/path:/another/path"},
            clear=False,
        ):
            env = prepare_subprocess_env()
            assert "PYTHONPATH" in env
            assert env["PYTHONPATH"] == "/some/path:/another/path"

    def test_forwards_ce_manager_vars(self):
        """CE_MANAGER_* prefix vars must be forwarded."""
        with patch.dict(
            os.environ,
            {
                "CE_MANAGER_BOOTSTRAP_ENABLED": "1",
                "CE_MANAGER_BOOTSTRAP_CONFIG_PATH": "/path/to/config.json",
            },
            clear=False,
        ):
            env = prepare_subprocess_env()
            assert env.get("CE_MANAGER_BOOTSTRAP_ENABLED") == "1"
            assert env.get("CE_MANAGER_BOOTSTRAP_CONFIG_PATH") == "/path/to/config.json"

    def test_forwards_provider_api_keys(self):
        """Provider credential vars matching suffix patterns are forwarded."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "SOME_PROVIDER_API_BASE": "http://x"},
            clear=False,
        ):
            env = prepare_subprocess_env()
            assert env.get("OPENAI_API_KEY") == "sk-test"
            assert env.get("SOME_PROVIDER_API_BASE") == "http://x"

    def test_does_not_forward_unrelated_vars(self):
        """Unrelated env vars must not leak into the subprocess."""
        with patch.dict(
            os.environ,
            {"EDITOR": "vim", "TERM": "xterm-256color", "SHELL": "/bin/zsh"},
            clear=False,
        ):
            env = prepare_subprocess_env()
            assert "EDITOR" not in env
            assert "TERM" not in env
            assert "SHELL" not in env

    def test_forwards_home(self):
        """HOME is needed for package resolution and tool discovery."""
        with patch.dict(os.environ, {"HOME": "/Users/testuser"}, clear=False):
            env = prepare_subprocess_env()
            assert env.get("HOME") == "/Users/testuser"
