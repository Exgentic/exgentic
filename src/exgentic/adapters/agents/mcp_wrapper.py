# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""MCP client utilities for connecting to external MCP servers and extracting tools."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, create_model

from ...utils.sync import run_sync


async def _extract_mcp_tools_async(
    mcp_url: str,
    timeout: float = 30.0,
) -> tuple[list[Callable[..., Any]], Any, Any]:
    """Connect to MCP server and extract tools as callable functions.

    Returns:
        Tuple of (tool_functions, session, http_context) where both must be kept alive
        for the tools to work.
    """
    # Keep the HTTP client context alive
    http_context = streamable_http_client(mcp_url)
    read_stream, write_stream, _ = await http_context.__aenter__()

    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()

    try:
        await session.initialize()
        tools_result = await session.list_tools()

        if not tools_result.tools:
            raise ValueError(f"No tools found on MCP server at {mcp_url}")

        # Convert each MCP tool to a Python function
        functions = []
        for tool in tools_result.tools:
            func = _create_mcp_tool_wrapper(session, tool)
            functions.append(func)

        return functions, session, http_context
    except Exception:
        # Clean up on error
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await http_context.__aexit__(None, None, None)
        except Exception:
            pass
        raise


def _create_mcp_tool_wrapper(session: ClientSession, tool: Any) -> Callable[..., Any]:
    """Create a Python function wrapper for an MCP tool.

    The wrapper function will call the MCP server's call_tool method
    when invoked.
    """
    tool_name = tool.name
    tool_description = tool.description or f"Execute {tool_name}"

    # Parse the tool's input schema to create function parameters
    input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Create pydantic model for arguments validation
    field_definitions = {}
    for param_name, param_schema in properties.items():
        param_type = _json_schema_type_to_python(param_schema.get("type", "string"))
        is_required = param_name in required

        if is_required:
            field_definitions[param_name] = (param_type, ...)
        else:
            default_value = param_schema.get("default")
            field_definitions[param_name] = (param_type, default_value)

    # Create a pydantic model for the arguments
    if field_definitions:
        ArgsModel = create_model(  # noqa: N806
            f"{tool_name}_args",
            **field_definitions,
        )
    else:
        # No arguments
        class ArgsModel(BaseModel):
            pass

    def tool_wrapper(**kwargs) -> Any:
        """Wrapper function that calls the MCP server."""
        # Validate arguments
        try:
            args_instance = ArgsModel(**kwargs)
            args_dict = args_instance.model_dump()
        except Exception as e:
            return {"error": f"Invalid arguments for {tool_name}: {e}"}

        # Call the MCP server synchronously
        try:
            result = run_sync(_call_mcp_tool_async(session, tool_name, args_dict))
            return result
        except Exception as e:
            return {"error": f"Failed to execute {tool_name}: {e}"}

    # Set function metadata
    tool_wrapper.__name__ = tool_name
    tool_wrapper.__doc__ = tool_description

    # Build function signature
    params = []
    for param_name, param_schema in properties.items():
        param_type = _json_schema_type_to_python(param_schema.get("type", "string"))
        is_required = param_name in required

        if is_required:
            default = inspect.Parameter.empty
        else:
            default = param_schema.get("default")
            if default is None:
                default = None

        param = inspect.Parameter(
            param_name,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=param_type,
        )
        params.append(param)

    tool_wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

    return tool_wrapper


async def _call_mcp_tool_async(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool asynchronously."""
    result = await session.call_tool(tool_name, arguments)

    # Extract the result content
    if hasattr(result, "content") and result.content:
        # Return the first content item's text
        if len(result.content) > 0:
            first_content = result.content[0]
            if hasattr(first_content, "text"):
                return first_content.text
            return str(first_content)

    return result


def _json_schema_type_to_python(json_type: str) -> type:
    """Convert JSON schema type to Python type."""
    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_mapping.get(json_type, str)


def extract_mcp_tools(mcp_url: str, timeout: float = 30.0) -> tuple[list[Callable[..., Any]], Any, Any]:
    """Synchronous wrapper to extract tools from an MCP server.

    Args:
        mcp_url: URL of the MCP server (e.g., http://localhost:8000/mcp)
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (tool_functions, session, http_context) where both must be kept alive
    """
    return run_sync(_extract_mcp_tools_async(mcp_url, timeout), timeout=timeout + 5.0)


async def _extract_mcp_tool_metadata_async(mcp_url: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Extract only tool metadata (as plain dictionaries) from MCP server.

    This is used to get serializable tool information without creating function wrappers.

    Returns:
        List of tool metadata dictionaries with keys: name, description, inputSchema
    """
    http_context = streamable_http_client(mcp_url)
    read_stream, write_stream, _ = await http_context.__aenter__()

    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()

    try:
        await session.initialize()
        tools_result = await session.list_tools()

        if not tools_result.tools:
            raise ValueError(f"No tools found on MCP server at {mcp_url}")

        # Convert tools to plain dictionaries, excluding admin tools
        admin_tools = {"create_session", "delete_session", "list_tasks", "evaluate_session"}
        tool_metadata = []
        for tool in tools_result.tools:
            # Skip admin tools
            if tool.name in admin_tools:
                continue

            metadata = {
                "name": tool.name,
                "description": tool.description or f"Execute {tool.name}",
                "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            tool_metadata.append(metadata)

        return tool_metadata
    finally:
        # Clean up
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await http_context.__aexit__(None, None, None)
        except Exception:
            pass


def extract_mcp_tool_metadata(mcp_url: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Synchronous wrapper to extract tool metadata from an MCP server.

    Args:
        mcp_url: URL of the MCP server (e.g., http://localhost:8000/mcp)
        timeout: Connection timeout in seconds

    Returns:
        List of tool metadata dictionaries
    """
    return run_sync(_extract_mcp_tool_metadata_async(mcp_url, timeout), timeout=timeout + 5.0)


__all__ = ["extract_mcp_tools", "_extract_mcp_tools_async", "extract_mcp_tool_metadata"]

# Made with Bob
