# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from click.testing import CliRunner
from exgentic import __version__
from exgentic.interfaces.cli.main import cli


def test_cli_version_long_flag():
    """Test that --version flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"exgentic {__version__}" in result.output


def test_cli_version_short_flag():
    """Test that -V flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(cli, ["-V"])
    assert result.exit_code == 0
    assert f"exgentic {__version__}" in result.output
