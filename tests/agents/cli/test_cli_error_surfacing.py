# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for CLI error surfacing.

Verifies that when a CLI agent fails, the stderr/stdout from the process
are included in the error message so they propagate through the
coordinator and session orchestrator to the user.
"""

from __future__ import annotations

from exgentic.agents.cli.command_runner import CLIExecutionError


def test_cli_execution_error_includes_stderr():
    """CLIExecutionError.__str__() must include stderr output."""
    err = CLIExecutionError(
        "CLI exited non-zero (1): some-cmd",
        code=1,
        stdout="",
        stderr="Error: permission denied, mkdir '/work/.claude/debug'",
        cmd=["some-cmd"],
    )
    msg = str(err)
    assert "CLI exited non-zero (1)" in msg
    assert "STDERR:" in msg
    assert "permission denied" in msg


def test_cli_execution_error_includes_stdout():
    """CLIExecutionError.__str__() must include stdout when present."""
    err = CLIExecutionError(
        "CLI exited non-zero (2): my-cli",
        code=2,
        stdout="some useful debug output",
        stderr="fatal error occurred",
        cmd=["my-cli"],
    )
    msg = str(err)
    assert "STDERR:" in msg
    assert "fatal error occurred" in msg
    assert "STDOUT:" in msg
    assert "some useful debug output" in msg


def test_cli_execution_error_omits_empty_streams():
    """CLIExecutionError.__str__() omits STDERR/STDOUT sections when empty."""
    err = CLIExecutionError(
        "CLI exited non-zero (1): cmd",
        code=1,
        stdout="",
        stderr="",
        cmd=["cmd"],
    )
    msg = str(err)
    assert "STDERR:" not in msg
    assert "STDOUT:" not in msg
    assert msg == "CLI exited non-zero (1): cmd"
