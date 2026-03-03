# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import rich_click as click

from ...lib.api import setup_benchmark
from ..options import apply_debug_mode


@click.command("setup")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.argument("benchmark")
def setup_cmd(debug: bool, benchmark: str) -> None:
    """Run a benchmark's setup.sh script."""
    apply_debug_mode(debug)
    try:
        setup_benchmark(benchmark)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


__all__ = ["setup_cmd"]
