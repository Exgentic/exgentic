# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from click.testing import CliRunner
from framework import __version__
from framework.interfaces.cli.main import cli


def test_cli_version_long_flag():
    """Test that --version flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"framework {__version__}" in result.output


def test_cli_version_short_flag():
    """Test that -V flag displays version and exits."""
    runner = CliRunner()
    result = runner.invoke(cli, ["-V"])
    assert result.exit_code == 0
    assert f"framework {__version__}" in result.output
