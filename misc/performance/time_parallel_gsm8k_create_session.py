# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
"""Time parallel create_session calls against an MCP server.

This script connects to an MCP server, lists available tasks, selects a number of
task IDs, and invokes `create_session` in parallel to measure contention and total
wall-clock performance.

Example:
    python misc/performance/time_parallel_gsm8k_create_session.py \
        --mcp-url http://127.0.0.1:8000/mcp \
        --count 5 \
        --cleanup
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CreateSessionResult:
    task_id: str
    success: bool
    elapsed_seconds: float
    session_id: str | None
    payload: dict[str, Any] | None
    error: str | None


async def list_tasks(mcp_session) -> list[str]:
    result = await mcp_session.call_tool("list_tasks", {})
    if result.isError:
        raise RuntimeError(f"list_tasks failed: {result.content}")

    if not result.content:
        raise RuntimeError("list_tasks returned no content")

    payload = json.loads(result.content[0].text)
    tasks = payload.get("tasks", payload.get("task_ids", []))
    if not isinstance(tasks, list):
        raise RuntimeError(f"Unexpected list_tasks payload: {payload}")
    return [str(task_id) for task_id in tasks]


async def create_session_for_task(mcp_session, task_id: str) -> CreateSessionResult:
    started_at = time.perf_counter()
    try:
        result = await mcp_session.call_tool("create_session", {"task_id": task_id})
        elapsed = time.perf_counter() - started_at

        if result.isError:
            return CreateSessionResult(
                task_id=task_id,
                success=False,
                elapsed_seconds=elapsed,
                session_id=None,
                payload=None,
                error=str(result.content),
            )

        if not result.content:
            return CreateSessionResult(
                task_id=task_id,
                success=False,
                elapsed_seconds=elapsed,
                session_id=None,
                payload=None,
                error="create_session returned no content",
            )

        payload = json.loads(result.content[0].text)
        session_id = payload.get("session_id")
        error = payload.get("error")
        return CreateSessionResult(
            task_id=task_id,
            success=error is None and session_id is not None,
            elapsed_seconds=elapsed,
            session_id=str(session_id) if session_id is not None else None,
            payload=payload,
            error=str(error) if error is not None else None,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        return CreateSessionResult(
            task_id=task_id,
            success=False,
            elapsed_seconds=elapsed,
            session_id=None,
            payload=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def delete_session(mcp_session, session_id: str) -> None:
    result = await mcp_session.call_tool("delete_session", {"session_id": session_id})
    if result.isError:
        raise RuntimeError(f"delete_session failed for {session_id}: {result.content}")


async def run(args: argparse.Namespace) -> int:
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        print("Error: mcp package not installed", file=sys.stderr)
        print("Install with the project environment that includes the MCP client.", file=sys.stderr)
        return 1

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        async with streamable_http_client(args.mcp_url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as mcp_session:
                print(f"Connecting to MCP server: {args.mcp_url}")
                await mcp_session.initialize()
                print("Initialized MCP session")

                task_ids = await list_tasks(mcp_session)
                if not task_ids:
                    print("No tasks available")
                    return 1

                selected_task_ids = task_ids[: args.count]
                print(f"Selected {len(selected_task_ids)} task(s): {selected_task_ids}")

                started_at = time.perf_counter()
                results = await asyncio.gather(
                    *(create_session_for_task(mcp_session, task_id) for task_id in selected_task_ids)
                )
                total_elapsed = time.perf_counter() - started_at

                print("\ncreate_session results")
                print("-" * 80)
                for index, result in enumerate(results, start=1):
                    status = "ok" if result.success else "error"
                    print(
                        f"[{index}] task_id={result.task_id} status={status} "
                        f"elapsed={result.elapsed_seconds:.3f}s "
                        f"session_id={result.session_id or '-'}"
                    )
                    if result.error:
                        print(f"    error: {result.error}")

                success_count = sum(1 for result in results if result.success)
                failure_count = len(results) - success_count
                per_call_times = [result.elapsed_seconds for result in results]

                print("\nsummary")
                print("-" * 80)
                print(f"parallel calls:      {len(results)}")
                print(f"successful:          {success_count}")
                print(f"failed:              {failure_count}")
                print(f"total wall time:     {total_elapsed:.3f}s")
                print(f"min call time:       {min(per_call_times):.3f}s")
                print(f"max call time:       {max(per_call_times):.3f}s")
                print(f"avg call time:       {sum(per_call_times) / len(per_call_times):.3f}s")

                if args.cleanup:
                    session_ids = [result.session_id for result in results if result.session_id]
                    if session_ids:
                        print("\ncleanup")
                        print("-" * 80)
                        cleanup_started_at = time.perf_counter()
                        await asyncio.gather(*(delete_session(mcp_session, session_id) for session_id in session_ids))
                        cleanup_elapsed = time.perf_counter() - cleanup_started_at
                        print(f"deleted {len(session_ids)} session(s) in {cleanup_elapsed:.3f}s")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple MCP create_session calls in parallel and time them.",
    )
    parser.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP server URL (default: http://127.0.0.1:8000/mcp)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of task IDs to use for parallel create_session calls (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP client timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete created sessions after timing run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("--count must be > 0", file=sys.stderr)
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

# Made with Bob
