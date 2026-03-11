# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path

import rich_click as click

from ....adapters.agents.mcp_server import MCPServer
from ....core.context import run_scope
from ....core.types import SessionIndex
from ....observers.logging import get_logger
from ....utils.settings import get_settings
from ...registry import load_benchmark
from ..options import apply_debug_mode


@click.command("mcp")
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode (sets settings.debug=true and log level to DEBUG)",
)
@click.option(
    "--benchmark",
    required=True,
    help="Benchmark name (e.g., tau2, gsm8k)",
)
@click.option(
    "--task-id",
    "task_ids",
    required=True,
    multiple=True,
    help="Task IDs to load (can be specified multiple times)",
)
@click.option(
    "--subset",
    default=None,
    help="Benchmark subset (optional)",
)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the MCP server to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Port to bind the MCP server to (default: auto-assign)",
)
def mcp_cmd(
    debug: bool,
    benchmark: str,
    task_ids: tuple[str, ...],
    subset: str | None,
    host: str,
    port: int | None,
) -> None:
    """Start an MCP server with benchmark task actions for multiple tasks.

    This command creates an MCP server that exposes the actions available
    for multiple benchmark tasks. Each tool call includes a task_id parameter
    to route the action to the appropriate session.

    Example:
        exgentic mcp --benchmark tau2 --task-id 1 --task-id 2 --task-id 3
    """
    apply_debug_mode(debug)
    settings = get_settings()

    # Load benchmark class
    try:
        benchmark_cls = load_benchmark(benchmark)
    except Exception as exc:
        raise click.ClickException(f"Failed to load benchmark '{benchmark}': {exc}") from exc

    # Create benchmark instance with minimal config
    benchmark_kwargs = {}
    if subset:
        # Get the subset argument name from registry
        from ...registry import get_benchmark_subset_arg

        subset_arg = get_benchmark_subset_arg(benchmark)
        if subset_arg:
            benchmark_kwargs[subset_arg] = subset

    try:
        benchmark_instance = benchmark_cls(**benchmark_kwargs)
    except Exception as exc:
        raise click.ClickException(f"Failed to create benchmark instance: {exc}") from exc

    # Create sessions for all tasks
    output_dir = Path(settings.output_dir)
    run_id = f"mcp_{benchmark}_{'_'.join(task_ids)}"

    # Initialize context using run_scope
    with run_scope(run_id=run_id, output_dir=str(output_dir)):
        sessions = {}
        action_types = None

        click.echo(f"Creating sessions for {len(task_ids)} tasks...")
        for task_id in task_ids:
            session_id = f"mcp_{benchmark}_{task_id}"
            try:
                session_index = SessionIndex(
                    benchmark=benchmark,
                    agent="mcp_server",
                    task_id=task_id,
                    session_id=session_id,
                    output_dir=str(output_dir),
                )
                session = benchmark_instance.create_session(session_index)

                # Start the session to get initial observation
                click.echo(f"  Starting session for task {task_id}...")
                _ = session.start()
                click.echo(f"  ✓ Task {task_id} session started")

                sessions[task_id] = session

                # Get actions from the first session (all tasks should have same actions)
                if action_types is None:
                    action_types = session.actions
                    if not action_types:
                        raise click.ClickException(f"No actions available for task {task_id}")

            except Exception as exc:
                # Clean up any created sessions
                for sess in sessions.values():
                    try:
                        sess.close()
                    except Exception:
                        pass
                raise click.ClickException(f"Failed to create/start session for task {task_id}: {exc}") from exc

        click.echo(f"\n✓ Created {len(sessions)} sessions")

        if action_types is None:
            raise click.ClickException("No actions available from any task")

        click.echo(f"✓ Loaded {len(action_types)} actions from {benchmark}")

        # Convert action types to callable tools that execute via session.step()
        # Each tool will include a task_id parameter to route to the correct session
        tools = []
        for action_type in action_types:
            # Get the schema for this action's arguments
            args_model = action_type.arguments

            # Create a callable function for each action with proper signature
            def make_tool_fn(at, sessions_dict, args_cls):
                # Create a dynamic function with the correct signature
                # by using the args model directly as a parameter
                def tool_fn(task_id: str, **kwargs):
                    """Tool function that executes action via session.step()."""
                    # Validate task_id
                    if task_id not in sessions_dict:
                        return {
                            "error": f"Invalid task_id: {task_id}. Available task IDs: {list(sessions_dict.keys())}"
                        }

                    sess = sessions_dict[task_id]

                    # Create an instance of the arguments model
                    try:
                        args_instance = args_cls(**kwargs)
                    except Exception as e:
                        return {"error": f"Invalid arguments: {e}"}

                    # Create action using the action type's class
                    try:
                        action = at.cls(name=at.name, arguments=args_instance)
                    except Exception as e:
                        return {"error": f"Failed to create action: {e}"}

                    # Execute the action via session.step() with timeout
                    # Use a thread-safe approach since session.step() may block
                    result_container = {}
                    error_container = {}

                    def execute_step():
                        try:
                            observation = sess.step(action)
                            result_container["observation"] = observation
                        except Exception as e:
                            error_container["error"] = e

                    step_thread = threading.Thread(target=execute_step, daemon=True)
                    step_thread.start()
                    step_thread.join(timeout=30.0)  # 30 second timeout

                    if step_thread.is_alive():
                        return {"error": "Action execution timed out after 30 seconds"}

                    if "error" in error_container:
                        return {"error": f"Failed to execute action: {error_container['error']}"}

                    observation = result_container.get("observation")

                    # Return the observation result
                    if observation is None:
                        return {"status": "completed", "message": "Session done"}

                    # Format observation for return
                    result = {
                        "status": "success",
                        "task_id": task_id,
                        "observation": str(observation),
                    }

                    if hasattr(observation, "result"):
                        result["result"] = str(observation.result)

                    return result

                # Set function metadata
                tool_fn.__name__ = at.name
                tool_fn.__doc__ = at.description or f"Execute {at.name} action"

                # Create proper signature from the args model
                import inspect

                # Build parameters - add task_id as first required parameter
                params = [
                    inspect.Parameter(
                        "task_id",
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=str,
                    )
                ]

                for field_name, field_info in args_cls.model_fields.items():
                    annotation = field_info.annotation
                    default = field_info.default if field_info.default is not None else inspect.Parameter.empty
                    if field_info.is_required():
                        default = inspect.Parameter.empty

                    param = inspect.Parameter(
                        field_name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation
                    )
                    params.append(param)

                # Set the signature
                tool_fn.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

                return tool_fn

            tools.append(make_tool_fn(action_type, sessions, args_model))

        # Create log directory
        log_dir = output_dir / "mcp_logs" / f"multi_{benchmark}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = get_logger(__name__)

        # Create and start MCP server
        click.echo(f"\nStarting MCP server for {benchmark} with {len(task_ids)} tasks")
        click.echo(f"Task IDs: {', '.join(task_ids)}")
        click.echo(f"Available actions: {len(tools)}")
        click.echo(f"Log directory: {log_dir}")

        try:
            server = MCPServer(
                host=host,
                port=port,
                tools=tools,
                log_dir=log_dir,
                logger=logger,
                stringify_empty_output=True,
            )

            server.start()

            click.echo("\n✓ MCP server started successfully!")
            click.echo(f"  Host: {server.connect_host}")
            click.echo(f"  Port: {server.port}")
            click.echo(f"  URL: http://{server.connect_host}:{server.port}/mcp")
            click.echo("\nPress Ctrl+C to stop the server...")

            # Keep the server running
            def signal_handler(sig, frame):
                click.echo("\n\nShutting down MCP server...")
                server.stop()
                for task_id, sess in sessions.items():
                    try:
                        click.echo(f"  Closing session for task {task_id}...")
                        sess.close()
                    except Exception:
                        pass
                click.echo("Server stopped.")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Wait for the server thread
            if server.thread:
                server.thread.join()

        except Exception as exc:
            for sess in sessions.values():
                try:
                    sess.close()
                except Exception:
                    pass
            raise click.ClickException(f"Failed to start MCP server: {exc}") from exc


__all__ = ["mcp_cmd"]

# Made with Bob
