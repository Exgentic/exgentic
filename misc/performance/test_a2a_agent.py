# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
"""A2A Agent Test Harness for MCP Server Tasks.

This test harness measures performance and memory consumption while using an
A2A agent to solve tasks from an MCP server.

Usage:
    python test_a2a_agent.py --mcp-url URL --a2a-url URL [options]

Options:
    --mcp-url URL           MCP server URL (default: http://127.0.0.1:8000/mcp)
    --a2a-url URL           A2A agent server URL (default: http://127.0.0.1:8001)
    --cleanup               Delete sessions after completion (default: False)
    --delay SECS            Delay between task executions in seconds (default: 1.0)
    --limit NUM             Limit number of tasks to test (default: all tasks)
    --server-pid PID        PID of A2A server process to monitor (default: auto-detect)
    --timeout SECS          Timeout for each task execution in seconds (default: 300)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psutil


def find_a2a_server_pid(port: int = 8001) -> Optional[int]:
    """Find the PID of the running A2A server process by port.

    Args:
        port: Port number the A2A server is listening on

    Returns:
        PID of the server process, or None if not found
    """
    try:
        # First try using lsof to find process listening on the port
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            pid = int(result.stdout.strip().split("\n")[0])
            # Verify it's an exgentic or uvicorn process
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline())
                if "exgentic" in cmdline or "uvicorn" in cmdline or "a2a" in cmdline:
                    return pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Fallback: search through all processes
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and ("exgentic" in " ".join(cmdline) or "uvicorn" in " ".join(cmdline)):
                    # Check if this process is listening on the port
                    for conn in proc.connections():
                        if conn.status == "LISTEN" and conn.laddr.port == port:
                            return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue

        return None
    except Exception as e:
        print(f"Error finding A2A server PID: {e}")
        return None


class MemoryMonitor:
    """Monitor memory usage of a specific process and system."""

    def __init__(self, pid: Optional[int] = None):
        """Initialize memory monitor.

        Args:
            pid: Process ID to monitor. If None, monitors current process.
        """
        if pid is None:
            self.process = psutil.Process(os.getpid())
            self.monitoring_self = True
        else:
            try:
                self.process = psutil.Process(pid)
                self.monitoring_self = False
            except psutil.NoSuchProcess as err:
                raise ValueError(f"No process found with PID {pid}") from err

        self.pid = self.process.pid
        self.measurements: List[Dict[str, Any]] = []

    def measure(self, label: str) -> Dict[str, float]:
        """Take a memory measurement and store it."""
        # Process memory (parent)
        mem_info = self.process.memory_info()
        process_rss_mb = mem_info.rss / 1024 / 1024  # RSS in MB
        process_vms_mb = mem_info.vms / 1024 / 1024  # VMS in MB

        # Get all child processes and their memory
        children_rss_mb = 0
        children_vms_mb = 0
        child_count = 0

        try:
            children = self.process.children(recursive=True)
            for child in children:
                try:
                    child_mem = child.memory_info()
                    children_rss_mb += child_mem.rss / 1024 / 1024
                    children_vms_mb += child_mem.vms / 1024 / 1024
                    child_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Total memory (parent + all children)
        total_rss_mb = process_rss_mb + children_rss_mb
        total_vms_mb = process_vms_mb + children_vms_mb

        # System memory
        sys_mem = psutil.virtual_memory()
        system_used_mb = sys_mem.used / 1024 / 1024
        system_percent = sys_mem.percent

        measurement = {
            "label": label,
            "timestamp": time.time(),
            "process_rss_mb": round(process_rss_mb, 2),
            "process_vms_mb": round(process_vms_mb, 2),
            "children_rss_mb": round(children_rss_mb, 2),
            "children_vms_mb": round(children_vms_mb, 2),
            "child_count": child_count,
            "total_rss_mb": round(total_rss_mb, 2),
            "total_vms_mb": round(total_vms_mb, 2),
            "system_used_mb": round(system_used_mb, 2),
            "system_percent": round(system_percent, 2),
        }

        self.measurements.append(measurement)
        return measurement

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of memory measurements."""
        if not self.measurements:
            return {}

        process_rss_values = [m["process_rss_mb"] for m in self.measurements]
        process_vms_values = [m["process_vms_mb"] for m in self.measurements]
        total_rss_values = [m["total_rss_mb"] for m in self.measurements]
        total_vms_values = [m["total_vms_mb"] for m in self.measurements]
        child_counts = [m["child_count"] for m in self.measurements]

        return {
            "total_measurements": len(self.measurements),
            "process_rss": {
                "initial_mb": process_rss_values[0],
                "final_mb": process_rss_values[-1],
                "delta_mb": round(process_rss_values[-1] - process_rss_values[0], 2),
                "max_mb": max(process_rss_values),
                "min_mb": min(process_rss_values),
                "avg_mb": round(sum(process_rss_values) / len(process_rss_values), 2),
            },
            "process_vms": {
                "initial_mb": process_vms_values[0],
                "final_mb": process_vms_values[-1],
                "delta_mb": round(process_vms_values[-1] - process_vms_values[0], 2),
                "max_mb": max(process_vms_values),
                "min_mb": min(process_vms_values),
                "avg_mb": round(sum(process_vms_values) / len(process_vms_values), 2),
            },
            "total_rss": {
                "initial_mb": total_rss_values[0],
                "final_mb": total_rss_values[-1],
                "delta_mb": round(total_rss_values[-1] - total_rss_values[0], 2),
                "max_mb": max(total_rss_values),
                "min_mb": min(total_rss_values),
                "avg_mb": round(sum(total_rss_values) / len(total_rss_values), 2),
            },
            "total_vms": {
                "initial_mb": total_vms_values[0],
                "final_mb": total_vms_values[-1],
                "delta_mb": round(total_vms_values[-1] - total_vms_values[0], 2),
                "max_mb": max(total_vms_values),
                "min_mb": min(total_vms_values),
                "avg_mb": round(sum(total_vms_values) / len(total_vms_values), 2),
            },
            "child_processes": {
                "initial_count": child_counts[0],
                "final_count": child_counts[-1],
                "max_count": max(child_counts),
                "avg_count": round(sum(child_counts) / len(child_counts), 1),
            },
        }

    def print_measurement(self, measurement: Dict[str, Any]):
        """Print a single measurement."""
        print(f"   Parent: RSS={measurement['process_rss_mb']:.2f}MB, " f"VMS={measurement['process_vms_mb']:.2f}MB")
        print(
            f"   Children ({measurement['child_count']}): RSS={measurement['children_rss_mb']:.2f}MB, "
            f"VMS={measurement['children_vms_mb']:.2f}MB"
        )
        print(
            f"   TOTAL: RSS={measurement['total_rss_mb']:.2f}MB, "
            f"VMS={measurement['total_vms_mb']:.2f}MB, "
            f"System={measurement['system_percent']:.1f}%"
        )

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        if not summary:
            print("No measurements taken")
            return

        print("\n" + "=" * 80)
        print("MEMORY CONSUMPTION SUMMARY")
        print("=" * 80)

        rss = summary["process_rss"]
        vms = summary["process_vms"]
        total_rss = summary["total_rss"]
        total_vms = summary["total_vms"]
        children = summary["child_processes"]

        print(f"\nTotal Measurements: {summary['total_measurements']}")

        print("\n" + "-" * 80)
        print("PARENT PROCESS ONLY")
        print("-" * 80)

        print("\nProcess RSS (Resident Set Size):")
        print(f"  Initial:  {rss['initial_mb']:.2f} MB")
        print(f"  Final:    {rss['final_mb']:.2f} MB")
        print(f"  Delta:    {rss['delta_mb']:+.2f} MB")
        print(f"  Max:      {rss['max_mb']:.2f} MB")
        print(f"  Min:      {rss['min_mb']:.2f} MB")
        print(f"  Average:  {rss['avg_mb']:.2f} MB")

        print("\nProcess VMS (Virtual Memory Size):")
        print(f"  Initial:  {vms['initial_mb']:.2f} MB")
        print(f"  Final:    {vms['final_mb']:.2f} MB")
        print(f"  Delta:    {vms['delta_mb']:+.2f} MB")
        print(f"  Max:      {vms['max_mb']:.2f} MB")
        print(f"  Min:      {vms['min_mb']:.2f} MB")
        print(f"  Average:  {vms['avg_mb']:.2f} MB")

        print("\n" + "-" * 80)
        print("TOTAL (PARENT + ALL CHILDREN)")
        print("-" * 80)

        print("\nChild Processes:")
        print(f"  Initial Count:  {children['initial_count']}")
        print(f"  Final Count:    {children['final_count']}")
        print(f"  Max Count:      {children['max_count']}")
        print(f"  Average Count:  {children['avg_count']:.1f}")

        print("\nTotal RSS (Resident Set Size):")
        print(f"  Initial:  {total_rss['initial_mb']:.2f} MB")
        print(f"  Final:    {total_rss['final_mb']:.2f} MB")
        print(f"  Delta:    {total_rss['delta_mb']:+.2f} MB")
        print(f"  Max:      {total_rss['max_mb']:.2f} MB")
        print(f"  Min:      {total_rss['min_mb']:.2f} MB")
        print(f"  Average:  {total_rss['avg_mb']:.2f} MB")

        print("\nTotal VMS (Virtual Memory Size):")
        print(f"  Initial:  {total_vms['initial_mb']:.2f} MB")
        print(f"  Final:    {total_vms['final_mb']:.2f} MB")
        print(f"  Delta:    {total_vms['delta_mb']:+.2f} MB")
        print(f"  Max:      {total_vms['max_mb']:.2f} MB")
        print(f"  Min:      {total_vms['min_mb']:.2f} MB")
        print(f"  Average:  {total_vms['avg_mb']:.2f} MB")

        # Memory leak detection (based on total RSS)
        print("\n" + "-" * 80)
        if total_rss["delta_mb"] > 100:
            print(f"⚠️  WARNING: Significant memory increase detected: {total_rss['delta_mb']:+.2f} MB")
        elif total_rss["delta_mb"] > 10:
            print(f"⚠️  NOTICE: Moderate memory increase: {total_rss['delta_mb']:+.2f} MB")
        elif total_rss["delta_mb"] > 0:
            print(f"✓ Memory increase within normal range: {total_rss['delta_mb']:+.2f} MB")
        else:
            print("✓ No memory increase detected (or memory decreased)")


async def call_a2a_agent(a2a_url: str, task_input: str, timeout: float = 300.0, debug: bool = False) -> Dict[str, Any]:
    """Call the A2A agent to solve a task.

    Args:
        a2a_url: URL of the A2A agent server
        task_input: Task description/input
        timeout: Timeout in seconds
        debug: Enable debug output

    Returns:
        Dictionary with result information
    """
    try:
        from a2a.client import ClientFactory, create_text_message_object
        from a2a.client.errors import A2AClientTimeoutError
        from a2a.types import Role, TextPart
        import json
    except ImportError:
        print("Error: a2a package not installed")
        print("Install with: pip install a2a")
        raise

    start_time = time.time()
    
    try:
        # Create httpx client with custom timeout
        import httpx
        httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
        
        # Create client config with custom httpx client
        from a2a.client import ClientConfig
        client_config = ClientConfig(httpx_client=httpx_client)
        
        # Connect to A2A agent
        if debug:
            print(f"   [DEBUG] Connecting to A2A agent at {a2a_url} (timeout={timeout}s)")
        client = await ClientFactory.connect(a2a_url, client_config=client_config)
        
        # Create a text message with explicit role
        if debug:
            print(f"   [DEBUG] Creating message with role=user, content length={len(task_input)}")
        message = create_text_message_object(role=Role.user, content=task_input)
        
        if debug:
            print(f"   [DEBUG] Message created: role={message.role}, message_id={message.message_id}")
            print(f"   [DEBUG] Message parts: {len(message.parts)}")
        
        # Send message and collect results
        result_text = ""
        task_id = None
        event_count = 0
        
        if debug:
            print(f"   [DEBUG] Sending message to agent...")
        
        async for response in client.send_message(message):
            event_count += 1
            if debug:
                print(f"   [DEBUG] Received response #{event_count}: type={type(response).__name__}")
            
            if isinstance(response, tuple):
                # This is a (Task, Event) tuple
                task, event = response
                if task_id is None:
                    task_id = task.id
                    if debug:
                        print(f"   [DEBUG] Task ID: {task_id}")
                
                if debug and event:
                    # For status updates, just print the text
                    if hasattr(event, 'kind') and event.kind == 'status-update':
                        if hasattr(event, 'status') and hasattr(event.status, 'message'):
                            msg = event.status.message
                            if hasattr(msg, 'parts'):
                                texts = []
                                for part in msg.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        texts.append(part.root.text)
                                if texts:
                                    print(f"   [DEBUG] Status update: {' '.join(texts)}")
                                else:
                                    print(f"   [DEBUG] Status update (no text)")
                            else:
                                print(f"   [DEBUG] Status update: {event.status.state}")
                        else:
                            print(f"   [DEBUG] Status update: {event.status.state if hasattr(event, 'status') else 'unknown'}")
                    else:
                        # For other events, pretty print the full details
                        event_dict = event.model_dump() if hasattr(event, 'model_dump') else str(event)
                        print(f"   [DEBUG] Event details ({event.kind if hasattr(event, 'kind') else 'unknown'}):")
                        print(f"   {json.dumps(event_dict, indent=6, default=str)}")
                
                # Check for artifact updates
                if event and hasattr(event, 'artifact') and event.artifact:
                    if debug:
                        print(f"   [DEBUG] Event has artifact with {len(event.artifact.parts)} parts")
                    for part in event.artifact.parts:
                        # Part is a wrapper, actual data is in part.root
                        if hasattr(part, 'root') and isinstance(part.root, TextPart):
                            text = part.root.text
                            result_text += text
                            if debug:
                                print(f"   [DEBUG] Extracted text from artifact: {text[:100]}...")
            else:
                # This is a Message response
                if debug:
                    # Pretty print the message
                    msg_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                    print(f"   [DEBUG] Message response details:")
                    print(f"   {json.dumps(msg_dict, indent=6, default=str)}")
                
                if hasattr(response, 'parts'):
                    if debug:
                        print(f"   [DEBUG] Message response with {len(response.parts)} parts")
                    for part in response.parts:
                        # Part is a wrapper, actual data is in part.root
                        if hasattr(part, 'root') and isinstance(part.root, TextPart):
                            text = part.root.text
                            result_text += text
                            if debug:
                                print(f"   [DEBUG] Extracted text from message: {text[:100]}...")
        
        elapsed_time = time.time() - start_time
        
        if debug:
            print(f"   [DEBUG] Completed in {elapsed_time:.2f}s, received {event_count} events")
            print(f"   [DEBUG] Result length: {len(result_text)} characters")
        
        # Close httpx client
        await httpx_client.aclose()
        
        return {
            "success": True,
            "result": result_text,
            "elapsed_time": elapsed_time,
            "error": None,
            "task_id": task_id,
        }
    except A2AClientTimeoutError as e:
        elapsed_time = time.time() - start_time
        error_msg = f"A2A Timeout: {str(e)}"
        if debug:
            print(f"   [DEBUG] A2A Timeout Error: {error_msg}")
        # Close httpx client if it exists
        if 'httpx_client' in locals():
            await httpx_client.aclose()
        return {
            "success": False,
            "result": None,
            "elapsed_time": elapsed_time,
            "error": error_msg,
            "task_id": task_id if 'task_id' in locals() else None,
        }
    except asyncio.TimeoutError:
        elapsed_time = time.time() - start_time
        # Close httpx client if it exists
        if 'httpx_client' in locals():
            await httpx_client.aclose()
        return {
            "success": False,
            "result": None,
            "elapsed_time": elapsed_time,
            "error": f"Asyncio Timeout after {timeout}s",
            "task_id": None,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        if debug:
            print(f"   [DEBUG] Error: {error_msg}")
            import traceback
            traceback.print_exc()
        # Close httpx client if it exists
        if 'httpx_client' in locals():
            await httpx_client.aclose()
        return {
            "success": False,
            "result": None,
            "elapsed_time": elapsed_time,
            "error": error_msg,
            "task_id": None,
        }


async def test_a2a_agent(
    mcp_url: str,
    a2a_url: str,
    cleanup: bool = False,
    delay: float = 1.0,
    limit: int = 0,
    server_pid: Optional[int] = None,
    timeout: float = 300.0,
):
    """Test A2A agent by solving tasks from MCP server.

    Args:
        mcp_url: URL of the MCP server
        a2a_url: URL of the A2A agent server
        cleanup: Whether to delete sessions after completion
        delay: Delay between task executions in seconds
        limit: Limit number of tasks to test (0 = all tasks)
        server_pid: PID of A2A server process to monitor (None = auto-detect)
        timeout: Timeout for each task execution in seconds
    """
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        print("Error: mcp package not installed")
        print("Install with: pip install mcp")
        return 1

    # Auto-detect or use provided server PID
    if server_pid is None:
        # Extract port from A2A URL
        parsed_url = urlparse(a2a_url)
        port = parsed_url.port or 8001

        print(f"Auto-detecting A2A server process on port {port}...")
        server_pid = find_a2a_server_pid(port=port)
        if server_pid is None:
            print(f"❌ Could not find A2A server process listening on port {port}")
            print("   Please ensure the A2A server is running or provide --server-pid")
            return 1
        print(f"✓ Found A2A server process: PID {server_pid}")
    else:
        print(f"Using provided A2A server PID: {server_pid}")

    # Initialize memory monitor for the server process
    try:
        monitor = MemoryMonitor(pid=server_pid)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    created_sessions: List[Dict[str, str]] = []
    task_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("A2A AGENT TEST HARNESS")
    print("=" * 80)
    print(f"\nMCP Server URL: {mcp_url}")
    print(f"A2A Agent URL: {a2a_url}")
    print(f"Monitoring Process: PID {monitor.pid}")
    print(f"Cleanup after completion: {cleanup}")
    print(f"Delay between executions: {delay}s")
    print(f"Task limit: {limit if limit > 0 else 'all tasks'}")
    print(f"Task timeout: {timeout}s")

    # Initial memory measurement
    print("\n" + "-" * 80)
    print("INITIAL STATE")
    print("-" * 80)
    initial_mem = monitor.measure("initial")
    monitor.print_measurement(initial_mem)

    try:
        # Connect to MCP server
        print("\n" + "-" * 80)
        print("CONNECTING TO MCP SERVER")
        print("-" * 80)

        async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize
                print("Initializing session...")
                await session.initialize()
                print("✓ Session initialized")

                connection_mem = monitor.measure("after_connection")
                monitor.print_measurement(connection_mem)

                # List available tasks
                print("\n" + "-" * 80)
                print("FETCHING AVAILABLE TASKS")
                print("-" * 80)

                list_tasks_result = await session.call_tool("list_tasks", {})

                if list_tasks_result.isError:
                    print("❌ Error listing tasks")
                    print(f"   Result: {list_tasks_result.content}")
                    return 1

                # Parse task list
                tasks_data = json.loads(list_tasks_result.content[0].text)
                task_ids = tasks_data.get("tasks", tasks_data.get("task_ids", []))

                # Apply limit if specified
                if limit > 0 and len(task_ids) > limit:
                    task_ids = task_ids[:limit]
                    print(f"✓ Found {len(tasks_data.get('tasks', []))} tasks (limiting to {limit})")
                else:
                    print(f"✓ Found {len(task_ids)} tasks")
                print(f"   Task IDs: {task_ids[:10]}{'...' if len(task_ids) > 10 else ''}")

                tasks_mem = monitor.measure("after_list_tasks")
                monitor.print_measurement(tasks_mem)

                # Process tasks sequentially
                print("\n" + "-" * 80)
                print("PROCESSING TASKS WITH A2A AGENT")
                print("-" * 80)

                for i, task_id in enumerate(task_ids, 1):
                    print(f"\n[{i}/{len(task_ids)}] Processing task_id='{task_id}'...")

                    try:
                        # Create session in MCP server
                        create_result = await session.call_tool("create_session", {"task_id": str(task_id)})

                        if create_result.isError:
                            print(f"   ❌ Error creating session for task {task_id}")
                            print(f"   Result: {create_result.content}")
                            continue

                        # Extract session info
                        session_data = json.loads(create_result.content[0].text)
                        session_id = session_data.get("session_id")
                        task_input = session_data.get("task", "")
                        context = session_data.get("context", {})
                        
                        created_sessions.append({
                            "session_id": session_id,
                            "task_id": task_id,
                        })

                        print(f"   ✓ Session created: {session_id}")
                        print(f"   Task: {task_input}")
                        if context:
                            print(f"   Context: {context}")

                        # Build enhanced task input with context and session_id instructions
                        prompt_parts = [task_input]
                        
                        # Add context if available
                        if context:
                            prompt_parts.append("\nContext:")
                            for key, value in context.items():
                                prompt_parts.append(f"- {key}: {value}")
                        
                        # Add session_id instructions
                        prompt_parts.append(f"""

IMPORTANT: Use session id "{session_id}" in all your interactions with the benchmark tools.

When calling any benchmark-related tools or APIs, you MUST include the session_id parameter with the value "{session_id}". This ensures your actions are properly tracked and evaluated within the correct benchmark session.

If you are asked to submit an answer, make sure you call the submit MCP tool.""")
                        
                        enhanced_task_input = "\n".join(prompt_parts)

                        # Call A2A agent to solve the task
                        print(f"   🤖 Calling A2A agent...")
                        result = await call_a2a_agent(a2a_url, enhanced_task_input, timeout=timeout, debug=True)

                        if result["success"]:
                            print(f"   ✓ Task completed in {result['elapsed_time']:.2f}s")
                            result_preview = result["result"][:200] if result["result"] else "No result"
                            print(f"   Result: {result_preview}{'...' if len(result.get('result', '')) > 200 else ''}")
                            
                            # Evaluate the session
                            print(f"   📊 Evaluating session...")
                            try:
                                eval_result = await session.call_tool("evaluate_session", {"session_id": session_id})
                                
                                if eval_result.isError:
                                    print(f"   ⚠️  Evaluation error: {eval_result.content}")
                                    eval_data = None
                                else:
                                    eval_data = json.loads(eval_result.content[0].text)
                                    print(f"   Evaluation Results:")
                                    print(f"     Success: {eval_data.get('success', 'N/A')}")
                                    print(f"     Score: {eval_data.get('score', 'N/A')}")
                                    print(f"     Finished: {eval_data.get('is_finished', 'N/A')}")
                                    if eval_data.get('session_metrics'):
                                        print(f"     Metrics: {eval_data['session_metrics']}")
                            except Exception as e:
                                print(f"   ⚠️  Evaluation exception: {e}")
                                eval_data = None
                        else:
                            print(f"   ❌ Task failed: {result['error']}")
                            eval_data = None

                        task_results.append({
                            "task_id": task_id,
                            "session_id": session_id,
                            "evaluation": eval_data,
                            **result,
                        })

                        # Measure memory after each task
                        mem = monitor.measure(f"after_task_{i}")
                        monitor.print_measurement(mem)

                        # Delay between tasks
                        if delay > 0 and i < len(task_ids):
                            await asyncio.sleep(delay)

                    except Exception as e:
                        print(f"   ❌ Exception processing task {task_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                print(f"\n✓ Processed {len(task_results)} tasks")

                # Print task results summary
                print("\n" + "-" * 80)
                print("TASK RESULTS SUMMARY")
                print("-" * 80)
                successful = sum(1 for r in task_results if r["success"])
                failed = len(task_results) - successful
                total_time = sum(r["elapsed_time"] for r in task_results)
                avg_time = total_time / len(task_results) if task_results else 0

                print(f"Total tasks: {len(task_results)}")
                print(f"Successful: {successful}")
                print(f"Failed: {failed}")
                print(f"Total time: {total_time:.2f}s")
                print(f"Average time per task: {avg_time:.2f}s")
                
                # Evaluation statistics
                evaluated = sum(1 for r in task_results if r.get("evaluation"))
                eval_successful = sum(1 for r in task_results if r.get("evaluation") and r["evaluation"].get("success"))
                
                if evaluated > 0:
                    print(f"\nEvaluation Results:")
                    print(f"  Evaluated: {evaluated}/{len(task_results)}")
                    print(f"  Eval Success: {eval_successful}/{evaluated}")
                    
                    # Show scores if available
                    scores = [r["evaluation"].get("score") for r in task_results
                             if r.get("evaluation") and r["evaluation"].get("score") is not None]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        print(f"  Average Score: {avg_score:.2f}")
                        print(f"  Score Range: {min(scores):.2f} - {max(scores):.2f}")

                # Final memory measurement
                print("\n" + "-" * 80)
                print("FINAL STATE (All Tasks Processed)")
                print("-" * 80)
                final_mem = monitor.measure("final_all_processed")
                monitor.print_measurement(final_mem)

                # Cleanup if requested
                if cleanup and created_sessions:
                    print("\n" + "-" * 80)
                    print("CLEANUP: DELETING SESSIONS")
                    print("-" * 80)

                    for i, session_info in enumerate(created_sessions, 1):
                        session_id = session_info["session_id"]
                        print(f"\n[{i}/{len(created_sessions)}] Deleting session {session_id}...")

                        try:
                            delete_result = await session.call_tool("delete_session", {"session_id": session_id})

                            if delete_result.isError:
                                print("   ❌ Error deleting session")
                                print(f"   Result: {delete_result.content}")
                            else:
                                print("   ✓ Session deleted")

                            # Measure memory after every 10 deletions
                            if i % 10 == 0 or i == len(created_sessions):
                                mem = monitor.measure(f"after_delete_{i}")
                                monitor.print_measurement(mem)

                        except Exception as e:
                            print(f"   ❌ Exception deleting session: {e}")

                    # Final memory after cleanup
                    print("\n" + "-" * 80)
                    print("FINAL STATE (After Cleanup)")
                    print("-" * 80)
                    cleanup_mem = monitor.measure("final_after_cleanup")
                    monitor.print_measurement(cleanup_mem)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print summary
    monitor.print_summary()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="A2A agent test harness for MCP server tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP server URL (default: http://127.0.0.1:8000/mcp)",
    )
    parser.add_argument(
        "--a2a-url",
        default="http://127.0.0.1:9000",
        help="A2A agent server URL (default: http://127.0.0.1:9001)",
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete sessions after completion")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between task executions in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of tasks to test (0 = all tasks, default: 0)"
    )
    parser.add_argument(
        "--server-pid", type=int, default=None, help="PID of A2A server process to monitor (default: auto-detect)"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Timeout for each task execution in seconds (default: 300)"
    )

    args = parser.parse_args()

    return asyncio.run(
        test_a2a_agent(
            args.mcp_url,
            args.a2a_url,
            args.cleanup,
            args.delay,
            args.limit,
            args.server_pid,
            args.timeout,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
