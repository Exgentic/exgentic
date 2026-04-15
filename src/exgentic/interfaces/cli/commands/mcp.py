# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import signal
import sys
import threading
import uuid
from pathlib import Path

import rich_click as click

from ....adapters.agents.mcp_server import MCPServer
from ....core.context import run_scope
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
    "--subset",
    default=None,
    help="Benchmark subset (optional)",
)
@click.option(
    "--set",
    "set_values",
    multiple=True,
    help="Set benchmark.* values (e.g., benchmark.user_simulator_model='openai/Azure/gpt-4o')",
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
@click.option(
    "--disable-dns-rebinding-protection",
    is_flag=True,
    help="Disable DNS rebinding protection (allows connections from any host)",
)
def mcp_cmd(
    debug: bool,
    benchmark: str,
    subset: str | None,
    set_values: tuple[str, ...],
    host: str,
    port: int | None,
    disable_dns_rebinding_protection: bool,
) -> None:
    """Start an MCP server with dynamic session management.

    This command creates an MCP server that exposes:
    - list_tasks: Get available tasks from the benchmark
    - create_session: Create a session for a specific task
    - delete_session: Close and delete a session
    - Task-specific actions (once a session is created)

    Example:
        exgentic mcp --benchmark tau2
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

    # Parse and apply --set values for benchmark parameters
    if set_values:
        from ..options import _parse_set_list, _set_nested, _validate_set_keys_for_benchmark

        set_items = _parse_set_list(set_values)

        # Validate that only benchmark.* parameters are provided
        for group, path, _ in set_items:
            if group != "benchmark":
                raise click.ClickException(
                    f"Only benchmark.* parameters are allowed in mcp command. "
                    f"Got {group}.{'.'.join(path) if path else ''}"
                )

        _validate_set_keys_for_benchmark(benchmark, set_items)
        for group, path, value in set_items:
            if group == "benchmark":
                _set_nested(benchmark_kwargs, path, value)

    try:
        benchmark_instance = benchmark_cls(**benchmark_kwargs)
    except Exception as exc:
        raise click.ClickException(f"Failed to create benchmark instance: {exc}") from exc

    # Setup for dynamic session management
    output_dir = Path(settings.output_dir)
    run_id = f"mcp_{benchmark}"

    # Initialize context using run_scope
    with run_scope(run_id=run_id, output_dir=str(output_dir)):
        # Store the context for use in tool functions
        from ....core.context import get_context

        stored_context = get_context()

        # Dictionary to store active sessions (keyed by session_id)
        sessions = {}
        sessions_lock = threading.Lock()
        action_types = []  # Will be populated when first session is created

        # Get available tasks from benchmark evaluator
        try:
            evaluator = benchmark_instance.get_evaluator()
            task_ids = evaluator.list_tasks()
            click.echo(f"✓ Loaded benchmark '{benchmark}' with {len(task_ids)} tasks")
        except Exception as exc:
            raise click.ClickException(f"Failed to get tasks from benchmark: {exc}") from exc

        # Helper function to create a session
        def create_session_for_task(task_id: str) -> dict:
            """Create and start a session for a task. Returns a unique session_id."""
            nonlocal action_types

            # Set the context for this thread
            from ....core.context import set_context

            set_context(stored_context)

            # Generate a unique session_id using UUID
            session_id = str(uuid.uuid4())
            session = None

            try:
                # Create session using benchmark with task_id and session_id
                session = benchmark_instance.get_session(task_id=task_id, session_id=session_id)

                # Start the session to get initial observation
                _ = session.start()

                session_actions = session.actions
                if not session_actions:
                    try:
                        session.close()
                    except Exception:
                        pass
                    return {"error": f"No actions available for task {task_id}"}

                with sessions_lock:
                    sessions[session_id] = session
                    if not action_types:
                        action_types = session_actions

                return {
                    "status": "success",
                    "session_id": session_id,
                    "task_id": task_id,
                    "task": session.task,
                    "context": session.context,
                    "message": f"Session created for task {task_id} with session_id {session_id}",
                }

            except Exception as exc:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass
                with sessions_lock:
                    sessions.pop(session_id, None)
                return {"error": f"Failed to create session for task {task_id}: {exc}"}

        # Helper function to delete a session
        def delete_session_by_id(session_id: str) -> dict:
            """Close and delete a session by its session_id."""
            with sessions_lock:
                session = sessions.pop(session_id, None)

            if session is None:
                return {"error": f"No session found with session_id {session_id}"}

            try:
                task_id = session.task_id if hasattr(session, "task_id") else "unknown"
                session.close()
                return {
                    "status": "success",
                    "session_id": session_id,
                    "task_id": task_id,
                    "message": f"Session {session_id} closed and deleted",
                }
            except Exception as exc:
                return {"error": f"Error closing session {session_id}: {exc}"}

        # Create management tools
        def list_tasks_tool() -> dict:
            """List all available tasks from the benchmark."""
            return {
                "status": "success",
                "benchmark": benchmark,
                "tasks": task_ids,
                "total": len(task_ids),
            }

        def create_session_tool(task_id: str) -> dict:
            """Create a session for a specific task."""
            if task_id not in task_ids:
                return {"error": f"Invalid task_id: {task_id}. Available tasks: {task_ids[:10]}..."}
            return create_session_for_task(task_id)

        def delete_session_tool(session_id: str) -> dict:
            """Close and delete a session by its session_id."""
            return delete_session_by_id(session_id)

        def evaluate_session_tool(session_id: str) -> dict:
            """Evaluate a session and return whether it was successful. Closes the session if not done."""
            with sessions_lock:
                sess = sessions.get(session_id)

            if sess is None:
                return {"error": f"No session found with session_id {session_id}"}

            try:
                # IMPORTANT: Evaluate BEFORE closing!
                # For service runner sessions, closing the session closes the HTTP transport,
                # making it impossible to call score() afterwards.
                score_result = sess.score()

                # Close session if not done yet
                # Wrap in try-except to handle cases where the client is already closed
                if not sess.done():
                    try:
                        sess.close()
                    except Exception as close_exc:
                        # Log but don't fail - session might already be closed by agent
                        logger.warning(f"Error closing session {session_id}: {close_exc}")

                return {
                    "status": "success",
                    "session_id": session_id,
                    "success": score_result.success,
                    "score": score_result.score,
                    "is_finished": score_result.is_finished,
                    "session_metrics": score_result.session_metrics,
                    "session_metadata": score_result.session_metadata,
                }
            except Exception as exc:
                import traceback

                return {"error": f"Failed to evaluate session {session_id}: {exc}\n{traceback.format_exc()}"}

        # Set up function signatures for management tools
        import inspect

        list_tasks_tool.__name__ = "list_tasks"
        list_tasks_tool.__doc__ = "List all available tasks from the benchmark"
        list_tasks_tool.__signature__ = inspect.Signature([])  # type: ignore[attr-defined]

        create_session_tool.__name__ = "create_session"
        create_session_tool.__doc__ = "Create a session for a specific task and return the session_id"
        create_session_tool.__signature__ = inspect.Signature(
            [  # type: ignore[attr-defined]
                inspect.Parameter("task_id", inspect.Parameter.KEYWORD_ONLY, annotation=str)
            ]
        )

        delete_session_tool.__name__ = "delete_session"
        delete_session_tool.__doc__ = "Close and delete a session by its session_id"
        delete_session_tool.__signature__ = inspect.Signature(
            [  # type: ignore[attr-defined]
                inspect.Parameter("session_id", inspect.Parameter.KEYWORD_ONLY, annotation=str)
            ]
        )

        evaluate_session_tool.__name__ = "evaluate_session"
        evaluate_session_tool.__doc__ = "Evaluate a session and return whether it was successful"
        evaluate_session_tool.__signature__ = inspect.Signature(
            [  # type: ignore[attr-defined]
                inspect.Parameter("session_id", inspect.Parameter.KEYWORD_ONLY, annotation=str)
            ]
        )

        # Start with management tools
        tools = [list_tasks_tool, create_session_tool, delete_session_tool, evaluate_session_tool]

        # We'll add action tools dynamically when first session is created
        # For now, create a placeholder that will be populated
        action_tools = []

        def make_action_tool(at, args_cls):
            """Create a tool function for an action type."""

            def tool_fn(session_id: str, **kwargs):
                """Tool function that executes action via session.step()."""
                # Set the context for this thread
                from ....core.context import set_context

                set_context(stored_context)

                with sessions_lock:
                    if session_id not in sessions:
                        return {
                            "error": (
                                f"No session found with session_id {session_id}. "
                                "Create a session first using create_session."
                            )
                        }
                    sess = sessions[session_id]

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
                    "session_id": session_id,
                    "observation": str(observation),
                }

                if hasattr(observation, "result"):
                    result["result"] = str(observation.result)

                return result

            # Set function metadata
            tool_fn.__name__ = at.name
            tool_fn.__doc__ = at.description or f"Execute {at.name} action"

            # Build parameters - add session_id as first required parameter
            params = [
                inspect.Parameter(
                    "session_id",
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

        # Create a dummy session to get action types, then close it
        click.echo("Initializing action types...")
        if task_ids:
            temp_result = create_session_for_task(task_ids[0])
            if "error" in temp_result:
                raise click.ClickException(
                    f"Failed to initialize action types: {temp_result.get('error', 'Unknown error')}"
                )
            if not action_types:
                raise click.ClickException("Failed to initialize action types: No actions available")

            # Create action tools
            for action_type in action_types:
                args_model = action_type.arguments
                action_tools.append(make_action_tool(action_type, args_model))
            click.echo(f"✓ Loaded {len(action_tools)} action types")

            # Close the temporary session
            temp_session_id = temp_result.get("session_id")
            if temp_session_id:
                delete_session_by_id(temp_session_id)

        # Add action tools to the tools list
        tools.extend(action_tools)

        # Create log directory
        log_dir = output_dir / "mcp_logs" / benchmark
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = get_logger(__name__)

        # Create and start MCP server
        click.echo(f"\nStarting MCP server for {benchmark}")
        click.echo(f"Available tasks: {len(task_ids)}")
        click.echo("Management tools: 4 (list_tasks, create_session, delete_session, evaluate_session)")
        click.echo(f"Action tools: {len(action_tools)}")
        click.echo(f"Log directory: {log_dir}")

        try:
            server = MCPServer(
                host=host,
                port=port,
                tools=tools,
                log_dir=log_dir,
                logger=logger,
                stringify_empty_output=True,
                enable_dns_rebinding_protection=not disable_dns_rebinding_protection,
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
                with sessions_lock:
                    for session_id, sess in list(sessions.items()):
                        try:
                            click.echo(f"  Closing session {session_id}...")
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
            with sessions_lock:
                for sess in sessions.values():
                    try:
                        sess.close()
                    except Exception:
                        pass
            raise click.ClickException(f"Failed to start MCP server: {exc}") from exc


__all__ = ["mcp_cmd"]

# Made with Bob
