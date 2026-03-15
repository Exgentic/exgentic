# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import rich_click as click

from ...lib.api import setup_agent, setup_benchmark
from ..options import apply_debug_mode


@click.command("setup")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option("--benchmark", "benchmark", default=None, help="Benchmark slug name to set up.")
@click.option("--agent", "agent", default=None, help="Agent slug name to set up.")
def setup_cmd(debug: bool, benchmark: str | None, agent: str | None) -> None:
    """Run a benchmark's or agent's setup.sh script."""
    apply_debug_mode(debug)
    if benchmark is not None and agent is not None:
        raise click.UsageError("Specify either --benchmark or --agent, not both.")
    if benchmark is None and agent is None:
        raise click.UsageError("Specify either --benchmark or --agent.")
    try:
        if benchmark is not None:
            setup_benchmark(benchmark)
        else:
            setup_agent(agent)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


__all__ = ["setup_cmd"]
