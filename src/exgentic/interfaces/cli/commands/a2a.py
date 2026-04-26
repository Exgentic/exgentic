# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A2A (Agent-to-Agent) command - expose exgentic agents using the A2A framework."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from textwrap import dedent

import rich_click as click
import uvicorn
from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.routing import Route

from ....core.context import run_scope
from ....observers.logging import get_logger
from ....utils.settings import get_settings
from ...registry import load_agent
from ..options import apply_debug_mode

logger = logging.getLogger(__name__)


@click.command("a2a")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--agent",
    required=True,
    help="Agent slug name (e.g., tool_calling, openai_solo)",
)
@click.option(
    "--mcp",
    required=True,
    help="External MCP server address (e.g., http://localhost:8000/mcp)",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the A2A server to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the A2A server to (default: 8000)",
)
@click.option(
    "--set",
    "set_values",
    multiple=True,
    help="Set agent.* values (e.g., agent.model='openai/gpt-4o')",
)
def a2a_cmd(
    debug: bool,
    agent: str,
    mcp: str,
    host: str,
    port: int,
    set_values: tuple[str, ...],
) -> None:
    r"""Start an A2A (Agent-to-Agent) server using the A2A framework.

    This command:
    1. Connects to an external MCP server to extract available tools
    2. Creates an exgentic agent with those tools as available actions
    3. Exposes the agent using the A2A protocol
    4. Other agents can discover and interact with this agent via A2A

    Example:
        exgentic a2a --agent tool_calling --mcp http://localhost:8000/mcp \\
                     --set agent.model='gpt-4o'

    Note: Requires the 'a2a' optional dependency:
        pip install exgentic[a2a]
    """
    apply_debug_mode(debug)
    settings = get_settings()

    # Load agent class
    try:
        agent_cls = load_agent(agent)
    except Exception as exc:
        raise click.ClickException(f"Failed to load agent '{agent}': {exc}") from exc

    # Parse and apply --set values for agent parameters
    agent_kwargs = {}
    if set_values:
        from ..options import _parse_set_list, _set_nested, _validate_set_keys_for_agent

        set_items = _parse_set_list(set_values)

        # Validate that only agent.* parameters are provided
        for group, path, _ in set_items:
            if group != "agent":
                raise click.ClickException(
                    f"Only agent.* parameters are allowed in a2a command. "
                    f"Got {group}.{'.'.join(path) if path else ''}"
                )

        _validate_set_keys_for_agent(agent, set_items)
        for group, path, value in set_items:
            if group == "agent":
                _set_nested(agent_kwargs, path, value)

    # Setup output directory
    output_dir = Path(settings.output_dir)
    run_id = f"a2a_{agent}"

    # Initialize OTEL tracing if enabled
    if settings.otel_enabled:
        from ....utils.otel import check_otel_collector_health, init_tracing_from_env

        # Verify collector is reachable before starting the server
        healthy, err = check_otel_collector_health()
        if not healthy:
            raise click.ClickException(f"OTEL is enabled but collector is not reachable: {err}")

        init_tracing_from_env(service_name="exgentic-a2a")

        # Instrument ASGI/Starlette to propagate trace context from HTTP headers
        try:
            from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

            click.echo("✓ ASGI instrumentation will be applied to propagate trace context")
        except ImportError:
            click.echo("⚠️  opentelemetry-instrumentation-asgi not installed, trace propagation may not work")

        click.echo(f"✓ OTEL tracing enabled (endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'not set')})")

    # Initialize context using run_scope
    with run_scope(run_id=run_id, output_dir=str(output_dir)):
        # Create log directory
        log_dir = output_dir / "a2a_logs" / agent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        _ = get_logger(__name__)

        click.echo(f"\n🔗 Connecting to external MCP server at {mcp}...")

        # Import here to avoid circular imports
        from ....adapters.agents.mcp_wrapper import extract_mcp_tool_metadata

        # Extract tool metadata (plain dictionaries, serializable)
        try:
            tool_metadata = extract_mcp_tool_metadata(mcp, timeout=30.0)
            tool_names = [t["name"] for t in tool_metadata]
            click.echo("✓ Connected to MCP server")
            click.echo(f"✓ Extracted {len(tool_metadata)} tools: {', '.join(tool_names)}")
        except Exception as exc:
            raise click.ClickException(f"Failed to connect to MCP server at {mcp}: {exc}") from exc

        # Get agent info for the card
        from ...registry import get_agent_entries

        agent_entries = get_agent_entries()
        agent_entry = agent_entries.get(agent)
        agent_display_name = agent_entry.display_name if agent_entry else agent

        # Create Agent Card
        def get_agent_card() -> AgentCard:
            """Returns the Agent Card for the A2A Agent."""
            mcp_section = ""
            if tool_names:
                mcp_section = "\n\nAvailable MCP Tools:\n" + "\n".join(f"- {name}" for name in tool_names)

            capabilities = AgentCapabilities(streaming=True)
            skill = AgentSkill(
                id=f"exgentic_{agent}",
                name=agent_display_name,
                description=f"Exgentic {agent_display_name} agent with MCP tools",
                tags=tool_names,
                examples=[],
            )
            return AgentCard(
                name=f"Exgentic {agent_display_name}",
                description=dedent(
                    f"""\
                    This agent provides assistance using the Exgentic {agent_display_name} agent.
                    Connected to MCP server at {mcp}.{mcp_section}
                    """,
                ),
                url=f"http://{host}:{port}/",
                version="1.0.0",
                default_input_modes=["text"],
                default_output_modes=["text"],
                capabilities=capabilities,
                skills=[skill],
            )

        # Create and start A2A server
        click.echo(f"\n🚀 Starting A2A server for agent '{agent_display_name}'")
        click.echo(f"Log directory: {log_dir}")

        try:
            # Import the executor
            from ....adapters.agents.a2a_executor import ExgenticAgentExecutor

            agent_card = get_agent_card()

            request_handler = DefaultRequestHandler(
                agent_executor=ExgenticAgentExecutor(
                    agent_cls=agent_cls,
                    agent_kwargs=agent_kwargs,
                    mcp_address=mcp,
                    agent_display_name=agent_display_name,
                    tool_metadata=tool_metadata,  # Pass serializable metadata
                ),
                task_store=InMemoryTaskStore(),
            )

            server = A2AStarletteApplication(
                agent_card=agent_card,
                http_handler=request_handler,
            )

            app = server.build()

            # Add the agent-card.json path BEFORE wrapping with middleware
            app.routes.insert(
                0,
                Route(
                    "/.well-known/agent-card.json",
                    server._handle_get_agent_card,
                    methods=["GET"],
                    name="agent_card_new",
                ),
            )

            # Wrap with OpenTelemetry middleware to propagate trace context
            # This must be done AFTER all routes are added
            if settings.otel_enabled:
                try:
                    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

                    app = OpenTelemetryMiddleware(app)
                    click.echo("✓ ASGI middleware applied for trace context propagation")
                except ImportError:
                    pass

            click.echo("\n✓ A2A server started successfully!")
            click.echo(f"  Host: {host}")
            click.echo(f"  Port: {port}")
            click.echo(f"  URL: http://{host}:{port}/")
            click.echo(f"  Agent Card: http://{host}:{port}/.well-known/agent-card.json")
            click.echo("\n📁 Logs:")
            click.echo(f"  Log directory: {log_dir}")
            click.echo(f"  Agent execution logs: {output_dir / run_id}")
            click.echo("\nOther agents can now discover and interact with this agent via A2A protocol.")
            click.echo("Press Ctrl+C to stop the server...")

            # Run the server
            uvicorn.run(app, host=host, port=port)

        except KeyboardInterrupt:
            click.echo("\n\nShutting down A2A server...")
        except Exception as exc:
            raise click.ClickException(f"Failed to start A2A server: {exc}") from exc
        finally:
            click.echo("Server stopped.")


__all__ = ["a2a_cmd"]

# Made with Bob
