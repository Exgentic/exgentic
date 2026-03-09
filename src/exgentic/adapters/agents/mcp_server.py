# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import functools
import inspect
import logging
import socket
import threading
import time
from typing import Any, Callable, Iterable, Optional

import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp.server import FastMCP
from pydantic import BaseModel

from ...observers.logging import (
    configure_library_file_logging,
    configure_uvicorn_file_logging,
    get_logger,
)
from ...utils.sync import run_sync


class MCPServerConfig(BaseModel):
    http_timeout_seconds: float | None = None  # None means no timeout
    sse_read_timeout_seconds: float | None = None  # None means no timeout
    http_connect_timeout_seconds: float | None = None  # None means no timeout
    headers: dict[str, str] | None = None
    terminate_on_close: bool = True


class MCPServer:
    _MAX_SAFE_SCHEMA_INT = 2_147_483_647

    def __init__(
        self,
        mcp: FastMCP | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        tools: Iterable[Callable[..., Any]] | None = None,
        log_dir,
        logger: logging.Logger,
        stringify_empty_output: bool = False,
    ) -> None:
        self._mcp = mcp or self._build_fastmcp()
        self._host = host or "0.0.0.0"
        self._port = port
        self._log_dir = log_dir
        self._logger = logger
        self._stringify_empty_output = stringify_empty_output
        self._mcp_log_dir = self._log_dir / "mcp"
        self._mcp_log_dir.mkdir(parents=True, exist_ok=True)
        self._server_logger = get_logger(
            f"MCPServer_{id(self)}",
            str(self._mcp_log_dir / "server.log"),
        )
        ts = self._mcp.settings.transport_security
        ts.allowed_hosts = [
            *ts.allowed_hosts,
            "host.containers.internal:*",
            "host.docker.internal:*",
        ]
        ts.allowed_origins = [
            *ts.allowed_origins,
            "http://host.containers.internal:*",
            "http://host.docker.internal:*",
        ]

        tool_names: list[str] = []
        if tools:
            for fn in tools:
                tool_fn = self._wrap_tool(fn) if self._stringify_empty_output else fn
                tool = self._mcp._tool_manager.add_tool(tool_fn)
                tool.parameters = self._clamp_schema_ints(tool.parameters)
                tool_names.append(fn.__name__)
        self._log_tool_summary(tool_names)

        if self._logger is not None:
            self._logger.info("MCP logs at %s", self._mcp_log_dir)

        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._started = threading.Event()

    @classmethod
    def _clamp_schema_ints(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls._clamp_schema_ints(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._clamp_schema_ints(v) for v in obj]
        if isinstance(obj, int) and not isinstance(obj, bool):
            if obj > cls._MAX_SAFE_SCHEMA_INT:
                return cls._MAX_SAFE_SCHEMA_INT
        if isinstance(obj, float):
            if obj > cls._MAX_SAFE_SCHEMA_INT:
                return float(cls._MAX_SAFE_SCHEMA_INT)
        return obj

    @staticmethod
    def stringify_empty_output(result: Any) -> Any:
        if result is None:
            return "null"
        if result == []:
            return "[]"
        return result

    @classmethod
    def _wrap_tool(cls, fn: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                return cls.stringify_empty_output(result)

        else:

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                return cls.stringify_empty_output(result)

        wrapper.__signature__ = signature  # type: ignore[attr-defined]
        return wrapper

    def _build_fastmcp(self) -> FastMCP:
        root = logging.getLogger()
        null_handler: logging.Handler | None = None
        if not root.handlers:
            null_handler = logging.NullHandler()
            root.addHandler(null_handler)
        try:
            return FastMCP()
        finally:
            if null_handler is not None:
                try:
                    root.removeHandler(null_handler)
                except Exception:
                    pass

    @property
    def server(self) -> Optional[uvicorn.Server]:
        return self._server

    @property
    def thread(self) -> Optional[threading.Thread]:
        return self._thread

    @property
    def mcp(self) -> FastMCP:
        return self._mcp

    @property
    def host(self) -> str:
        return self._host

    @property
    def connect_host(self) -> str:
        # Use 127.0.0.1 for connecting, even if server binds to 0.0.0.0.
        return "127.0.0.1" if self._host == "0.0.0.0" else self._host

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("MCP server port not assigned yet.")
        return self._port

    def start(
        self,
        timeout: float = 60.0,
        *,
        tcp_timeout: float = 60.0,
        ping_timeout: float = 60.0,
    ) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._ensure_socket()

        self._started.clear()
        t = threading.Thread(
            target=self._thread_entry,
            name=f"mcp-native:{self._port}",
            daemon=True,
        )
        self._thread = t
        t.start()

        if not self._started.wait(timeout=timeout):
            raise RuntimeError("MCP uvicorn thread did not signal startup within timeout.")

        try:
            started_at = time.time()
            # Use 127.0.0.1 for connecting, even if server binds to 0.0.0.0
            connect_host = "127.0.0.1" if self._host == "0.0.0.0" else self._host
            wait_for_tcp(connect_host, self.port, timeout=tcp_timeout)
            tcp_elapsed = time.time() - started_at
            ping_started = time.time()
            run_sync(
                wait_for_mcp_ping_async(connect_host, self.port, timeout=ping_timeout),
                timeout=ping_timeout + 5.0,
            )
            ping_elapsed = time.time() - ping_started
            self._server_logger.info(
                "MCP readiness OK (tcp=%.2fs, ping=%.2fs)",
                tcp_elapsed,
                ping_elapsed,
            )
        except BaseException as exc:
            self.stop(error=exc, raise_on_timeout=False)
            raise

    def __enter__(self) -> "MCPServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(error=exc, raise_on_timeout=True)

    def stop(
        self,
        timeout: float = 60.0,
        *,
        error: BaseException | None = None,
        raise_on_timeout: bool = True,
    ) -> None:
        if self._server is not None:
            self._server.should_exit = True

        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                message = "MCP uvicorn server thread did not exit cleanly"
                if raise_on_timeout and error is None:
                    raise RuntimeError(message)
                self._server_logger.warning(message)
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _thread_entry(self) -> None:
        thread_id = threading.get_ident()
        cleanup_uvicorn = configure_uvicorn_file_logging(
            self._mcp_log_dir / "uvicorn.log",
            thread_id=thread_id,
        )
        cleanup_mcp = configure_library_file_logging(
            self._mcp_log_dir / "server.log",
            logger_names=["mcp", "mcp.server", "mcp.client"],
            thread_id=thread_id,
        )

        server = self._build_server()

        self._server_logger.info(
            "Starting MCP server on %s:%s (path=%s)",
            self._host,
            self.port,
            self._mcp.settings.streamable_http_path,
        )
        self._started.set()

        try:
            sock = self._sock
            if sock is not None:
                self._server_logger.info("Using pre-bound socket on %s:%s", self._host, self.port)
                server.run(sockets=[sock])
            else:
                server.run()
        finally:
            self._server_logger.info("MCP server stopped")
            cleanup_mcp()
            cleanup_uvicorn()

    def _ensure_socket(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._host, self._port or 0))
            sock.listen(1)
        except Exception:
            sock.close()
            raise
        self._sock = sock
        self._port = int(sock.getsockname()[1])

    def _build_server(self) -> uvicorn.Server:
        app = self._mcp.streamable_http_app()
        config = uvicorn.Config(app, host=self._host, port=self.port, log_config=None)
        server = uvicorn.Server(config)
        self._server = server
        return server

    def _log_tool_summary(self, tool_names: list[str]) -> None:
        count = len(tool_names)
        if count == 0:
            self._server_logger.info("Registered 0 MCP tools")
            return
        preview = ", ".join(tool_names[:6])
        if count > 6:
            preview = f"{preview}, +{count - 6} more"
        self._server_logger.info("Registered %s MCP tool(s): %s", count, preview)


def wait_for_tcp(host: str, port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise TimeoutError(f"Server at {host}:{port} did not open TCP port within {timeout}s. " f"Last error: {last_err!r}")


async def wait_for_mcp_ping_async(host: str, port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err: Optional[BaseException] = None
    url = f"http://{host}:{port}/mcp"

    import asyncio  # local import to keep module sync-first

    while time.time() < deadline:
        try:
            async with streamable_http_client(url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    await session.send_ping()
                    return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.1)

    raise TimeoutError(f"MCP ping at {host}:{port} failed within {timeout}s. " f"Last error: {last_err!r}")
