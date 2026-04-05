# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for Claude Code CLI configuration writing.

Verifies that the Claude CLI wrapper pre-creates directories that
the Claude Code CLI expects to write into, preventing EACCES errors
when running inside containers with --user flags.
"""

from __future__ import annotations

from exgentic.agents.cli.claude.cli import ClaudeCodeCLI, ExecutionBackend


def test_settings_config_creates_required_subdirs(tmp_path):
    """_write_settings_config must pre-create subdirectories for the container."""
    cli = ClaudeCodeCLI(runner=ExecutionBackend.PROCESS)
    cli._write_settings_config(tmp_path)

    claude_dir = tmp_path / ".claude"
    assert claude_dir.is_dir()
    assert (claude_dir / "settings.json").exists()

    # These directories must be pre-created so the container
    # process doesn't need mkdir permissions on .claude/
    for subdir in ("debug", "conversations", "projects", "todos"):
        assert (claude_dir / subdir).is_dir(), (
            f".claude/{subdir} must be pre-created to avoid " f"EACCES errors in container environments"
        )
