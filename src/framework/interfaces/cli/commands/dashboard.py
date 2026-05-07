# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

import rich_click as click

from ..options import apply_debug_mode


@click.command("dashboard")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
def dashboard_cmd(debug: bool) -> None:
    """Start the experiments graphical dashboard."""
    apply_debug_mode(debug)
    from ...dashboard.app import main as dashboard_main

    dashboard_main()


__all__ = ["dashboard_cmd"]
