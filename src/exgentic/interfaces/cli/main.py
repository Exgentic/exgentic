# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import rich_click as click

from ... import __version__
from .commands.analyze import analyse_cmd
from .commands.batch import batch_cmd
from .commands.compare import compare_cmd
from .commands.dashboard import dashboard_cmd
from .commands.evaluate import evaluate_cmd
from .commands.listing import list_cmd
from .commands.run_info import preview_cmd, results_cmd, status_cmd
from .commands.serve import serve_cmd
from .commands.setup import install_cmd, setup_cmd, uninstall_cmd
from .options import apply_debug_mode
from .render import print_banner, should_print_banner

click.rich_click.text_markup = False
click.rich_click.show_arguments = True
click.rich_click.options_table_column_types = [
    "required",
    "opt_short",
    "opt_long",
    "metavar",
    "help",
]
click.rich_click.show_envvar = True
click.rich_click.COMMAND_GROUPS = {
    "exgentic": [
        {
            "name": "Run",
            "commands": [
                "evaluate",
                "batch",
                "status",
                "preview",
                "results",
            ],
        },
        {
            "name": "Analyze",
            "commands": ["compare", "analyse"],
        },
        {
            "name": "Discover",
            "commands": ["list", "install", "uninstall", "setup"],
        },
        {
            "name": "Explore",
            "commands": ["dashboard"],
        },
        {
            "name": "Infrastructure",
            "commands": ["serve"],
        },
    ]
}


def _version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"exgentic {__version__}")
    ctx.exit()


@click.group()
@click.option(
    "--version",
    "-V",
    is_flag=True,
    callback=_version_callback,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
def cli(debug: bool) -> None:
    """Exgentic CLI."""
    apply_debug_mode(debug)


cli.add_command(evaluate_cmd)
cli.add_command(batch_cmd)
cli.add_command(analyse_cmd)
cli.add_command(status_cmd)
cli.add_command(preview_cmd)
cli.add_command(results_cmd)
cli.add_command(compare_cmd)
cli.add_command(list_cmd)
cli.add_command(dashboard_cmd)
cli.add_command(setup_cmd)
cli.add_command(install_cmd)
cli.add_command(uninstall_cmd)
cli.add_command(serve_cmd)


def main() -> None:
    if should_print_banner():
        print_banner()
    cli()


__all__ = [
    "batch_cmd",
    "cli",
    "compare_cmd",
    "dashboard_cmd",
    "evaluate_cmd",
    "install_cmd",
    "list_cmd",
    "main",
    "preview_cmd",
    "results_cmd",
    "serve_cmd",
    "setup_cmd",
    "status_cmd",
    "uninstall_cmd",
]
