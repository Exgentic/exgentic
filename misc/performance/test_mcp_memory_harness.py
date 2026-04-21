# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
"""Memory Test Harness for MCP Server.

This test harness measures memory consumption while sequentially creating
sessions for all available tasks in the MCP server.

Usage:
    python test_mcp_memory_harness.py [--url URL] [--cleanup]

Options:
    --url URL           MCP server URL (default: http://127.0.0.1:8000/mcp)
    --cleanup           Delete sessions after creation (default: False)
    --delay SECS        Delay between session creations in seconds (default: 0.5)
    --max-sessions NUM  Maximum parallel sessions to keep active (default: unlimited)
    --limit NUM         Limit number of tasks to test (default: all tasks)
    --server-pid PID    PID of MCP server process to monitor (default: auto-detect)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import psutil


def find_mcp_server_pid(port: int = 8000) -> Optional[int]:
    """Find the PID of the running MCP server process by port.

    Args:
        port: Port number the MCP server is listening on

    Returns:
        PID of the server process, or None if not found
    """
    try:
        # First try using lsof to find process listening on the port
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            pid = int(result.stdout.strip().split("\n")[0])
            # Verify it's an exgentic process
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline())
                if "exgentic" in cmdline and "mcp" in cmdline:
                    return pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Fallback: search through all processes
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "exgentic" in " ".join(cmdline) and "mcp" in " ".join(cmdline):
                    # Check if this process is listening on the port
                    for conn in proc.connections():
                        if conn.status == "LISTEN" and conn.laddr.port == port:
                            return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue

        return None
    except Exception as e:
        print(f"Error finding MCP server PID: {e}")
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


async def test_mcp_memory(
    mcp_url: str,
    cleanup: bool = False,
    delay: float = 0.5,
    max_sessions: int = 0,
    limit: int = 0,
    server_pid: Optional[int] = None,
):
    """Test MCP server memory consumption by creating sessions for all tasks.

    Args:
        mcp_url: URL of the MCP server
        cleanup: Whether to delete sessions after creation
        delay: Delay between session creations in seconds
        max_sessions: Maximum number of parallel sessions to keep active (0 = unlimited)
        limit: Limit number of tasks to test (0 = all tasks)
        server_pid: PID of MCP server process to monitor (None = auto-detect)
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
        # Extract port from URL
        from urllib.parse import urlparse

        parsed_url = urlparse(mcp_url)
        port = parsed_url.port or 8000

        print(f"Auto-detecting MCP server process on port {port}...")
        server_pid = find_mcp_server_pid(port=port)
        if server_pid is None:
            print(f"❌ Could not find MCP server process listening on port {port}")
            print("   Please ensure the MCP server is running or provide --server-pid")
            return 1
        print(f"✓ Found MCP server process: PID {server_pid}")
    else:
        print(f"Using provided MCP server PID: {server_pid}")

    # Initialize memory monitor for the server process
    try:
        monitor = MemoryMonitor(pid=server_pid)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    created_sessions: List[str] = []

    print("\n" + "=" * 80)
    print("MCP SERVER MEMORY TEST HARNESS")
    print("=" * 80)
    print(f"\nMCP Server URL: {mcp_url}")
    print(f"Monitoring Process: PID {monitor.pid}")
    print(f"Cleanup after creation: {cleanup}")
    print(f"Delay between creations: {delay}s")
    print(f"Max parallel sessions: {max_sessions if max_sessions > 0 else 'unlimited'}")
    print(f"Task limit: {limit if limit > 0 else 'all tasks'}")

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

                # Create sessions sequentially
                print("\n" + "-" * 80)
                print("CREATING SESSIONS SEQUENTIALLY")
                print("-" * 80)

                for i, task_id in enumerate(task_ids, 1):
                    print(f"\n[{i}/{len(task_ids)}] Creating session for task_id='{task_id}'...")

                    try:
                        create_result = await session.call_tool("create_session", {"task_id": str(task_id)})

                        if create_result.isError:
                            print(f"   ❌ Error creating session for task {task_id}")
                            print(f"   Result: {create_result.content}")
                            continue

                        # Extract session_id
                        session_data = json.loads(create_result.content[0].text)
                        print(session_data)
                        session_id = session_data.get("session_id")
                        created_sessions.append(session_id)

                        print(f"   ✓ Session created: {session_id}")

                        # Measure memory after every session creation
                        mem = monitor.measure(f"after_session_{i}")
                        monitor.print_measurement(mem)

                        # If max_sessions is set and we've reached the limit, delete oldest session
                        if max_sessions > 0 and len(created_sessions) > max_sessions:
                            oldest_session = created_sessions.pop(0)
                            print(f"   ⚠ Max sessions ({max_sessions}) reached, deleting oldest: {oldest_session}")
                            try:
                                delete_result = await session.call_tool(
                                    "delete_session", {"session_id": oldest_session}
                                )
                                if delete_result.isError:
                                    print("   ❌ Error deleting session")
                                else:
                                    print("   ✓ Session deleted")
                            except Exception as e:
                                print(f"   ❌ Exception deleting session: {e}")

                        # Delay between creations
                        if delay > 0 and i < len(task_ids):
                            await asyncio.sleep(delay)

                    except Exception as e:
                        print(f"   ❌ Exception creating session for task {task_id}: {e}")
                        continue

                print(f"\n✓ Created {len(created_sessions)} sessions successfully")

                # Final memory measurement
                print("\n" + "-" * 80)
                print("FINAL STATE (All Sessions Created)")
                print("-" * 80)
                final_mem = monitor.measure("final_all_created")
                monitor.print_measurement(final_mem)

                # Cleanup if requested
                if cleanup and created_sessions:
                    print("\n" + "-" * 80)
                    print("CLEANUP: DELETING SESSIONS")
                    print("-" * 80)

                    for i, session_id in enumerate(created_sessions, 1):
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
        description="Memory test harness for MCP server", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/mcp", help="MCP server URL (default: http://127.0.0.1:8000/mcp)"
    )
    parser.add_argument("--cleanup", action="store_true", help="Delete sessions after creation")
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between session creations in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Maximum number of parallel sessions to keep active (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of tasks to test (0 = all tasks, default: 0)"
    )
    parser.add_argument(
        "--server-pid", type=int, default=None, help="PID of MCP server process to monitor (default: auto-detect)"
    )

    args = parser.parse_args()

    return asyncio.run(
        test_mcp_memory(args.url, args.cleanup, args.delay, args.max_sessions, args.limit, args.server_pid)
    )


if __name__ == "__main__":
    sys.exit(main())
