# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

#!/usr/bin/env python3
"""Call an MCP tool using the official MCP SDK."""

import asyncio
import sys


async def call_mcp_tool_multi_session():
    """Connect to MCP server and call tools from multiple sessions."""
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        print("Error: mcp package not installed")
        print("Install with: pip install mcp")
        return 1

    mcp_url = "http://127.0.0.1:62675/mcp"

    print("=" * 80)
    print("Testing Multi-Session MCP Tool Calls")
    print("=" * 80)
    print(f"\nConnecting to: {mcp_url}")

    try:
        # Connect to MCP server
        async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize
                print("\n1. Initializing session...")
                await session.initialize()
                print("✓ Session initialized")

                # List tools
                print("\n2. Listing available tools...")
                tools_result = await session.list_tools()
                tools = tools_result.tools

                print(f"✓ Found {len(tools)} tools:")
                for i, tool in enumerate(tools[:10], 1):
                    print(f"   {i}. {tool.name}")
                    if tool.description:
                        desc = tool.description[:70]
                        print(f"      {desc}...")

                if len(tools) > 10:
                    print(f"   ... and {len(tools) - 10} more")

                if not tools:
                    print("⚠ No tools available")
                    return 0

                # Test calling message tool with task_id 1
                print("\n3. Testing task_id='1' - Calling message tool")
                arguments_task1 = {"task_id": "1", "content": "Hello from task 1!"}
                print(f"   Arguments: {arguments_task1}")

                result1 = await session.call_tool("message", arguments_task1)
                print("\n✓ Task 1 call successful!")
                print(f"   Result: {result1.content}")

                if result1.isError:
                    print("   ⚠ Tool returned an error")

                # Test calling message tool with task_id 2
                print("\n4. Testing task_id='2' - Calling message tool")
                arguments_task2 = {"task_id": "2", "content": "Hello from task 2!"}
                print(f"   Arguments: {arguments_task2}")

                result2 = await session.call_tool("message", arguments_task2)
                print("\n✓ Task 2 call successful!")
                print(f"   Result: {result2.content}")

                if result2.isError:
                    print("   ⚠ Tool returned an error")

                # Test calling bash tool with different task_ids
                print("\n5. Testing task_id='1' - Calling bash tool")
                bash_args_task1 = {"task_id": "1", "command": 'echo "Task 1 bash command"'}
                print(f"   Arguments: {bash_args_task1}")

                result3 = await session.call_tool("bash", bash_args_task1)
                print("\n✓ Task 1 bash call successful!")
                print(f"   Result: {result3.content}")

                print("\n6. Testing task_id='2' - Calling bash tool")
                bash_args_task2 = {"task_id": "2", "command": 'echo "Task 2 bash command"'}
                print(f"   Arguments: {bash_args_task2}")

                result4 = await session.call_tool("bash", bash_args_task2)
                print("\n✓ Task 2 bash call successful!")
                print(f"   Result: {result4.content}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("Multi-session tool calls complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(call_mcp_tool_multi_session()))

# Made with Bob
