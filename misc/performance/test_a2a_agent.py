# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
"""A2A Agent Test Harness for MCP Server Tasks.

This test harness measures performance and memory consumption while using an
A2A agent to solve tasks from an MCP server.

Usage:
    python test_a2a_agent.py --mcp-url URL --a2a-url URL [options]

Options:
    --mcp-url URL           MCP server URL (default: http://127.0.0.1:8000/mcp)
    --a2a-url URL           A2A agent server URL (default: http://127.0.0.1:9000)
    --cleanup               Delete sessions after completion (default: False)
    --delay SECS            Delay between task executions in seconds (default: 1.0)
    --limit NUM             Limit number of tasks to test (default: all tasks)
    --server-pid PID        PID of A2A server process to monitor (default: auto-detect)
    --timeout SECS          Timeout for each task execution in seconds (default: 300)
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psutil

# Setup logger
logger = logging.getLogger(__name__)


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


async def call_a2a_agent(a2a_url: str, task_input: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Call the A2A agent to solve a task.

    Args:
        a2a_url: URL of the A2A agent server
        task_input: Task description/input
        timeout: Timeout in seconds for the A2A client (default: 600)

    Returns:
        Dictionary with result information
    """
    from a2a.client import ClientConfig, ClientFactory, create_text_message_object
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.client.errors import A2AClientTimeoutError
    from a2a.types import Role, TextPart

    start_time = time.time()
    httpx_client = None
    
    try:
        import httpx

        # Use a long timeout for A2A client to handle long-running tasks
        httpx_client = httpx.AsyncClient(timeout=timeout)
        client_config = ClientConfig(httpx_client=httpx_client)
        
        # Fetch the agent card manually and override the URL
        logger.debug(f"Resolving Agent Card at {a2a_url} (timeout={timeout}s)")
        
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=a2a_url)
        card = await resolver.get_agent_card()
        
        # Override the URL in the card to use the actual server URL
        logger.debug(f"Agent card original URL: '{card.url}'")
        logger.debug(f"Overriding agent card URL to: '{a2a_url}'")
        card.url = a2a_url
        
        # Create the client using the modified card
        logger.debug("Creating client with overridden URL")
        client = ClientFactory(client_config).create(card=card)
        
        # Create a text message with explicit role
        logger.debug(f"Creating message with role=user, content length={len(task_input)}")
        message = create_text_message_object(role=Role.user, content=task_input)
        
        logger.debug(f"Message created: role={message.role}, message_id={message.message_id}")
        logger.debug(f"Message parts: {len(message.parts)}")
        
        # Send message and collect results
        result_text = ""
        task_id = None
        event_count = 0
        
        logger.debug("Sending message to agent...")
        
        async for response in client.send_message(message):
            event_count += 1
            logger.debug(f"Received response #{event_count}: type={type(response).__name__}")
            
            if isinstance(response, tuple):
                # This is a (Task, Event) tuple
                task, event = response
                if task_id is None:
                    task_id = task.id
                    logger.debug(f"Task ID: {task_id}")
                
                if event:
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
                                    logger.debug(f"Status update: {' '.join(texts)}")
                                else:
                                    logger.debug("Status update (no text)")
                            else:
                                logger.debug(f"Status update: {event.status.state}")
                        else:
                            logger.debug(f"Status update: {event.status.state if hasattr(event, 'status') else 'unknown'}")
                    else:
                        # For other events, pretty print the full details
                        event_dict = event.model_dump() if hasattr(event, 'model_dump') else str(event)
                        logger.debug(f"Event details ({event.kind if hasattr(event, 'kind') else 'unknown'}):")
                        logger.debug(f"{json.dumps(event_dict, indent=6, default=str)}")
                
                # Check for artifact updates
                if event and hasattr(event, 'artifact') and event.artifact:
                    logger.debug(f"Event has artifact with {len(event.artifact.parts)} parts")
                    for part in event.artifact.parts:
                        # Part is a wrapper, actual data is in part.root
                        if hasattr(part, 'root') and isinstance(part.root, TextPart):
                            text = part.root.text
                            result_text += text
                            logger.debug(f"Extracted text from artifact: {text[:100]}...")
            else:
                # This is a Message response
                msg_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                logger.debug("Message response details:")
                logger.debug(f"{json.dumps(msg_dict, indent=6, default=str)}")
                
                if hasattr(response, 'parts'):
                    logger.debug(f"Message response with {len(response.parts)} parts")
                    for part in response.parts:
                        # Part is a wrapper, actual data is in part.root
                        if hasattr(part, 'root') and isinstance(part.root, TextPart):
                            text = part.root.text
                            result_text += text
                            logger.debug(f"Extracted text from message: {text[:100]}...")
        
        elapsed_time = time.time() - start_time
        
        logger.debug(f"Completed in {elapsed_time:.2f}s, received {event_count} events")
        logger.debug(f"Result length: {len(result_text)} characters")
        
        return {
            "success": True,
            "result": result_text,
            "elapsed_time": elapsed_time,
            "error": None,
            "task_id": task_id,
        }
    except A2AClientTimeoutError as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "result": None,
            "elapsed_time": elapsed_time,
            "error": f"A2A Timeout: {str(e)}",
            "task_id": None,
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.exception(f"Error calling A2A agent: {type(e).__name__}: {str(e)}")
        return {
            "success": False,
            "result": None,
            "elapsed_time": elapsed_time,
            "error": f"{type(e).__name__}: {str(e)}",
            "task_id": None,
        }
    finally:
        if httpx_client:
            try:
                await httpx_client.aclose()
            except Exception:
                pass


async def fetch_tasks(mcp_session) -> List[str]:
    """Fetch available tasks from MCP server.
    
    Args:
        mcp_session: MCP client session
        
    Returns:
        List of task IDs
    """
    list_tasks_result = await mcp_session.call_tool("list_tasks", {})
    
    if list_tasks_result.isError:
        raise RuntimeError(f"Error listing tasks: {list_tasks_result.content}")
    
    tasks_data = json.loads(list_tasks_result.content[0].text)
    task_ids = tasks_data.get("tasks", tasks_data.get("task_ids", []))
    
    logger.info(f"Found {len(task_ids)} tasks")
    
    return task_ids


async def create_mcp_session(mcp_session, task_id: str) -> Dict[str, Any]:
    """Create a session in the MCP server for a task.
    
    Args:
        mcp_session: MCP client session
        task_id: Task ID to create session for
        
    Returns:
        Dictionary with session information
    """
    create_result = await mcp_session.call_tool("create_session", {"task_id": str(task_id)})
    
    if create_result.isError:
        raise RuntimeError(f"Error creating session: {create_result.content}")
    
    session_data = json.loads(create_result.content[0].text)
    logger.debug(f"Session created: {session_data.get('session_id')}")
    
    return session_data


def build_enhanced_task_input(task_input: str, session_id: str, context: Dict[str, Any]) -> str:
    """Build enhanced task input with context and session_id instructions.
    
    Args:
        task_input: Original task description
        session_id: Session ID to include
        context: Additional context information
        
    Returns:
        Enhanced task input string
    """
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
    
    return "\n".join(prompt_parts)


async def evaluate_mcp_session(mcp_session, session_id: str) -> Optional[Dict[str, Any]]:
    """Evaluate a session in the MCP server.
    
    Args:
        mcp_session: MCP client session
        session_id: Session ID to evaluate
        
    Returns:
        Evaluation data dictionary, or None if evaluation failed
    """
    try:
        # Small delay to ensure A2A agent has finished
        await asyncio.sleep(0.5)
        
        eval_result = await mcp_session.call_tool("evaluate_session", {"session_id": session_id})
        
        if eval_result.isError:
            error_content = eval_result.content[0].text if eval_result.content else "Unknown error"
            logger.warning(f"Evaluation error: {error_content}")
            print(f"   ⚠️  Evaluation error: {error_content}")
            return None
        
        eval_text = eval_result.content[0].text
        logger.debug(f"Raw evaluation response: {eval_text}")
        
        eval_data = json.loads(eval_text)
        
        # Check if there's an error in the response
        if "error" in eval_data:
            logger.warning(f"Evaluation returned error: {eval_data['error']}")
            print(f"   ⚠️  Evaluation returned error: {eval_data['error']}")
            return None
        
        print(f"   Evaluation Results:")
        print(f"     Success: {eval_data.get('success', 'N/A')}")
        print(f"     Score: {eval_data.get('score', 'N/A')}")
        print(f"     Finished: {eval_data.get('is_finished', 'N/A')}")
        
        return eval_data
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse evaluation response as JSON: {e}")
        print(f"   ⚠️  Failed to parse evaluation response as JSON: {e}")
        return None
    except Exception as e:
        logger.exception(f"Evaluation exception: {type(e).__name__}: {e}")
        print(f"   ⚠️  Evaluation exception: {type(e).__name__}: {e}")
        return None


async def delete_mcp_session(mcp_session, session_id: str) -> bool:
    """Delete a session in the MCP server.
    
    Args:
        mcp_session: MCP client session
        session_id: Session ID to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        delete_result = await mcp_session.call_tool("delete_session", {"session_id": session_id})
        
        if delete_result.isError:
            logger.error(f"Error deleting session: {delete_result.content}")
            print(f"   ❌ Error deleting session: {delete_result.content}")
            return False
        
        logger.debug("Session deleted")
        print(f"   ✓ Session deleted")
        return True
        
    except Exception as e:
        logger.exception(f"Exception deleting session: {e}")
        print(f"   ❌ Exception deleting session: {e}")
        return False


def print_task_results_summary(task_results: List[Dict[str, Any]]):
    """Print summary of task results.
    
    Args:
        task_results: List of task result dictionaries
    """
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
        timeout: Timeout for each task execution in seconds (unused, kept for compatibility)
        debug: Enable debug output
    """
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        print("Error: mcp package not installed")
        print("Install with: pip install mcp")
        return 1

    # Auto-detect or use provided server PID
    monitor = None
    if server_pid is None:
        parsed_url = urlparse(a2a_url)
        port = parsed_url.port or 8001

        print(f"Auto-detecting A2A server process on port {port}...")
        server_pid = find_a2a_server_pid(port=port)
        if server_pid is None:
            print(f"⚠️  Could not find A2A server process listening on port {port}")
            print("   Memory monitoring will be disabled")
        else:
            print(f"✓ Found A2A server process: PID {server_pid}")
    else:
        print(f"Using provided A2A server PID: {server_pid}")

    # Initialize memory monitor for the server process (if PID available)
    if server_pid is not None:
        try:
            monitor = MemoryMonitor(pid=server_pid)
            print(f"✓ Memory monitoring enabled for PID {server_pid}")
        except ValueError as e:
            print(f"⚠️  Could not monitor PID {server_pid}: {e}")
            monitor = None

    created_sessions: List[Dict[str, str]] = []
    task_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("A2A AGENT TEST HARNESS")
    print("=" * 80)
    print(f"\nMCP Server URL: {mcp_url}")
    print(f"A2A Agent URL: {a2a_url}")
    print(f"Monitoring Process: {'PID ' + str(monitor.pid) if monitor else 'Disabled'}")
    print(f"Cleanup after completion: {cleanup}")
    print(f"Delay between executions: {delay}s")
    print(f"Task limit: {limit if limit > 0 else 'all tasks'}")

    # Initial memory measurement
    if monitor:
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
            async with ClientSession(read_stream, write_stream) as mcp_session:
                # Initialize
                print("Initializing session...")
                await mcp_session.initialize()
                print("✓ Session initialized")

                if monitor:
                    connection_mem = monitor.measure("after_connection")
                    monitor.print_measurement(connection_mem)

                # Fetch available tasks
                print("\n" + "-" * 80)
                print("FETCHING AVAILABLE TASKS")
                print("-" * 80)

                task_ids = await fetch_tasks(mcp_session)

                # Apply limit if specified
                if limit > 0 and len(task_ids) > limit:
                    task_ids = task_ids[:limit]
                    print(f"✓ Limiting to {limit} tasks")

                if monitor:
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
                        session_data = await create_mcp_session(mcp_session, task_id)
                        session_id = session_data.get("session_id")
                        task_input = session_data.get("task", "")
                        context = session_data.get("context", {})
                        
                        created_sessions.append({
                            "session_id": session_id,
                            "task_id": task_id,
                        })

                        print(f"   Task: {task_input[:100]}...")
                        
                        # Build enhanced task input
                        enhanced_task_input = build_enhanced_task_input(task_input, session_id, context)
                        
                        # Call A2A agent to solve the task
                        print(f"   🤖 Calling A2A agent...")
                        result = await call_a2a_agent(a2a_url, enhanced_task_input, timeout=timeout)

                        if result["success"]:
                            print(f"   ✓ Task completed in {result['elapsed_time']:.2f}s")
                            result_preview = result["result"][:200] if result["result"] else "No result"
                            print(f"   Result: {result_preview}...")
                            
                            # Evaluate the session
                            print(f"   📊 Evaluating session...")
                            eval_data = await evaluate_mcp_session(mcp_session, session_id)
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
                        if monitor:
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
                print_task_results_summary(task_results)

                # Final memory measurement
                if monitor:
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

                        await delete_mcp_session(mcp_session, session_id)

                        # Measure memory after every 10 deletions
                        if monitor and (i % 10 == 0 or i == len(created_sessions)):
                            mem = monitor.measure(f"after_delete_{i}")
                            monitor.print_measurement(mem)

                    # Final memory after cleanup
                    if monitor:
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
    finally:
        pass

    # Print summary
    if monitor:
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
        help="A2A agent server URL (default: http://127.0.0.1:9000)",
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
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )

    args = parser.parse_args()

    # Configure logging for this module only based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

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
