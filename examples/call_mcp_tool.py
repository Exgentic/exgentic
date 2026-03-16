# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

#!/usr/bin/env python3
"""Call an MCP tool using the official MCP SDK with dynamic session management."""

import asyncio
import sys


async def call_mcp_tool_with_session_management():
    """Connect to MCP server and demonstrate dynamic session management."""
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        print("Error: mcp package not installed")
        print("Install with: pip install mcp")
        return 1

    mcp_url = "http://127.0.0.1:62675/mcp"

    print("=" * 80)
    print("Testing MCP Server with Dynamic Session Management")
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
                for i, tool in enumerate(tools, 1):
                    print(f"   {i}. {tool.name}")
                    if tool.description:
                        desc = tool.description[:70]
                        print(f"      {desc}...")

                if not tools:
                    print("⚠ No tools available")
                    return 0

                # Step 3: List available tasks
                print("\n3. Listing available tasks...")
                list_tasks_result = await session.call_tool("list_tasks", {})
                print("✓ Tasks retrieved!")
                print(f"   Result: {list_tasks_result.content}")

                # Step 4: Create session for task 1
                print("\n4. Creating session for task_id='1'...")
                create_result_1 = await session.call_tool("create_session", {"task_id": "1"})
                print("✓ Session created!")
                print(f"   Result: {create_result_1.content}")

                if create_result_1.isError:
                    print("   ⚠ Tool returned an error")
                    return 1

                # Step 5: Create session for task 2
                print("\n5. Creating session for task_id='2'...")
                create_result_2 = await session.call_tool("create_session", {"task_id": "2"})
                print("✓ Session created!")
                print(f"   Result: {create_result_2.content}")

                if create_result_2.isError:
                    print("   ⚠ Tool returned an error")
                    return 1

                # Step 6: Call message tool with task_id 1
                print("\n6. Testing task_id='1' - Calling message tool")
                arguments_task1 = {"task_id": "1", "content": "Hello from task 1!"}
                print(f"   Arguments: {arguments_task1}")

                result1 = await session.call_tool("message", arguments_task1)
                print("\n✓ Task 1 message call successful!")
                print(f"   Result: {result1.content}")

                if result1.isError:
                    print("   ⚠ Tool returned an error")

                # Step 7: Call message tool with task_id 2
                print("\n7. Testing task_id='2' - Calling message tool")
                arguments_task2 = {"task_id": "2", "content": "Hello from task 2!"}
                print(f"   Arguments: {arguments_task2}")

                result2 = await session.call_tool("message", arguments_task2)
                print("\n✓ Task 2 message call successful!")
                print(f"   Result: {result2.content}")

                if result2.isError:
                    print("   ⚠ Tool returned an error")

                # Step 8: Call bash tool with task_id 1
                print("\n8. Testing task_id='1' - Calling bash tool")
                bash_args_task1 = {"task_id": "1", "command": 'echo "Task 1 bash command"'}
                print(f"   Arguments: {bash_args_task1}")

                result3 = await session.call_tool("bash", bash_args_task1)
                print("\n✓ Task 1 bash call successful!")
                print(f"   Result: {result3.content}")

                # Step 9: Call bash tool with task_id 2
                print("\n9. Testing task_id='2' - Calling bash tool")
                bash_args_task2 = {"task_id": "2", "command": 'echo "Task 2 bash command"'}
                print(f"   Arguments: {bash_args_task2}")

                result4 = await session.call_tool("bash", bash_args_task2)
                print("\n✓ Task 2 bash call successful!")
                print(f"   Result: {result4.content}")

                # Step 10: Delete session for task 1
                print("\n10. Deleting session for task_id='1'...")
                delete_result_1 = await session.call_tool("delete_session", {"task_id": "1"})
                print("✓ Session deleted!")
                print(f"   Result: {delete_result_1.content}")

                if delete_result_1.isError:
                    print("   ⚠ Tool returned an error")

                # Step 11: Try to use deleted session (should fail)
                print("\n11. Attempting to use deleted session (should fail)...")
                try:
                    fail_result = await session.call_tool("message", {"task_id": "1", "content": "This should fail"})
                    print(f"   Result: {fail_result.content}")
                    if "error" in str(fail_result.content).lower() or "no session" in str(fail_result.content).lower():
                        print("   ✓ Correctly rejected - session was deleted")
                except Exception as e:
                    print(f"   ✓ Correctly rejected with error: {e}")

                # Step 12: Delete session for task 2
                print("\n12. Deleting session for task_id='2'...")
                delete_result_2 = await session.call_tool("delete_session", {"task_id": "2"})
                print("✓ Session deleted!")
                print(f"   Result: {delete_result_2.content}")

                if delete_result_2.isError:
                    print("   ⚠ Tool returned an error")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("Dynamic session management test complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(call_mcp_tool_with_session_management()))

# Made with Bob
