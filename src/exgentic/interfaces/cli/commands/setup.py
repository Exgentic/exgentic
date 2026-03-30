# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import rich_click as click

from ..options import apply_debug_mode


def _get_registry_entry(slug: str, kind: str):
    """Look up a RegistryEntry for the given slug and kind ('benchmark' or 'agent')."""
    from ...registry import AGENTS, BENCHMARKS

    registry = BENCHMARKS if kind == "benchmark" else AGENTS
    entry = registry.get(slug)
    if entry is None:
        raise click.ClickException(f"Unknown {kind} slug '{slug}'")
    return entry


@click.command("install")
@click.option("--benchmark", "benchmark", default=None, help="Benchmark slug name to install.")
@click.option("--agent", "agent", default=None, help="Agent slug name to install.")
@click.option("--force", is_flag=True, help="Force reinstall even if already installed.")
@click.option("--docker", is_flag=True, help="Build a Docker image for the environment.")
@click.option("--local", is_flag=True, help="Install into the current Python (no isolation).")
def install_cmd(benchmark: str | None, agent: str | None, force: bool, docker: bool, local: bool) -> None:
    """Install a benchmark or agent environment.

    By default, creates an isolated Python venv with all dependencies.
    Use --docker to build a Docker image, or --local to install into
    the current Python (no isolation).
    """
    from ....environment import EnvType
    from ....environment.instance import get_manager

    if benchmark is not None and agent is not None:
        raise click.UsageError("Specify either --benchmark or --agent, not both.")
    if benchmark is None and agent is None:
        raise click.UsageError("Specify either --benchmark or --agent.")

    if docker and local:
        raise click.UsageError("Specify either --docker or --local, not both.")

    if docker:
        env_type = EnvType.DOCKER
    elif local:
        env_type = EnvType.LOCAL
    else:
        env_type = EnvType.VENV

    mgr = get_manager()

    try:
        if benchmark is not None:
            entry = _get_registry_entry(benchmark, "benchmark")
            name = f"benchmarks/{benchmark}"
        else:
            entry = _get_registry_entry(agent, "agent")
            name = f"agents/{agent}"

        kwargs: dict = {"env_type": env_type, "module_path": entry.module, "force": force}
        if env_type in (EnvType.VENV, EnvType.DOCKER):
            from ....environment.helpers import get_exgentic_install_target

            project_root, packages = get_exgentic_install_target()
            if project_root is not None:
                kwargs["project_root"] = project_root
            if packages:
                kwargs["packages"] = packages
        mgr.install(name, **kwargs)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@click.command("uninstall")
@click.option("--benchmark", "benchmark", default=None, help="Benchmark slug name to uninstall.")
@click.option("--agent", "agent", default=None, help="Agent slug name to uninstall.")
@click.option("--docker", is_flag=True, help="Uninstall Docker environment only.")
@click.option("--local", is_flag=True, help="Uninstall local environment only.")
def uninstall_cmd(benchmark: str | None, agent: str | None, docker: bool, local: bool) -> None:
    """Uninstall a benchmark or agent environment.

    Without flags, removes all environment types for the given name.
    """
    from ....environment import EnvType
    from ....environment.instance import get_manager

    if benchmark is not None and agent is not None:
        raise click.UsageError("Specify either --benchmark or --agent, not both.")
    if benchmark is None and agent is None:
        raise click.UsageError("Specify either --benchmark or --agent.")

    if docker and local:
        raise click.UsageError("Specify either --docker or --local, not both.")

    if docker:
        env_type = EnvType.DOCKER
    elif local:
        env_type = EnvType.LOCAL
    else:
        env_type = None

    mgr = get_manager()

    try:
        if benchmark is not None:
            name = f"benchmarks/{benchmark}"
        else:
            name = f"agents/{agent}"

        if env_type is not None:
            mgr.uninstall(name, env_type=env_type)
        else:
            mgr.uninstall(name)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@click.command("setup")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option("--benchmark", "benchmark", default=None, help="Benchmark slug name to set up.")
@click.option("--agent", "agent", default=None, help="Agent slug name to set up.")
@click.option(
    "--force",
    is_flag=True,
    help="Force reinstall even if already installed.",
)
def setup_cmd(debug: bool, benchmark: str | None, agent: str | None, force: bool) -> None:
    """[Deprecated] Use 'exgentic install' instead."""
    apply_debug_mode(debug)
    click.echo(
        "WARNING: 'exgentic setup' is deprecated. Use 'exgentic install' instead.",
        err=True,
    )

    # Delegate to install logic
    from ....environment import EnvType
    from ....environment.instance import get_manager

    if benchmark is not None and agent is not None:
        raise click.UsageError("Specify either --benchmark or --agent, not both.")
    if benchmark is None and agent is None:
        raise click.UsageError("Specify either --benchmark or --agent.")

    mgr = get_manager()

    try:
        if benchmark is not None:
            entry = _get_registry_entry(benchmark, "benchmark")
            name = f"benchmarks/{benchmark}"
        else:
            entry = _get_registry_entry(agent, "agent")
            name = f"agents/{agent}"

        mgr.install(name, env_type=EnvType.LOCAL, module_path=entry.module, force=force)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


__all__ = ["install_cmd", "setup_cmd", "uninstall_cmd"]
