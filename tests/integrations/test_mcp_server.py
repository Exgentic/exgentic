# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import logging
import time
from datetime import timedelta

import httpx
import pytest
from exgentic.adapters.agents import mcp_server as mcp_srv
from exgentic.utils.sync import run_sync
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError


class FakeMCP:
    def __init__(self, *args, **kwargs) -> None:
        fake_ts = type(
            "FakeTransportSecurity",
            (),
            {"allowed_hosts": [], "allowed_origins": []},
        )()
        self.settings = type(
            "FakeSettings",
            (),
            {"streamable_http_path": "/mcp", "transport_security": fake_ts},
        )()
        return

    def add_tool(self, fn, **kwargs) -> None:
        return None

    def streamable_http_app(self):
        return object()


class FakeConfig:
    def __init__(self, app, host, port, log_config=None) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.log_config = log_config


class FakeServer:
    def __init__(self, config) -> None:
        self.config = config
        self.should_exit = False

    def run(self, *args, **kwargs) -> None:
        while not self.should_exit:
            time.sleep(0.01)


def _patch_uvicorn(monkeypatch):
    monkeypatch.setattr(mcp_srv.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(mcp_srv.uvicorn, "Server", lambda cfg: FakeServer(cfg))
    monkeypatch.setattr(mcp_srv, "FastMCP", FakeMCP)
    monkeypatch.setattr(mcp_srv, "wait_for_tcp", lambda *args, **kwargs: None)

    def _noop_run_sync(coro, timeout=None):
        coro.close()
        return

    monkeypatch.setattr(mcp_srv, "run_sync", _noop_run_sync)
    monkeypatch.setattr(mcp_srv.socket, "socket", _fake_socket_factory())


def _fake_socket_factory():
    class FakeSocket:
        _next_port = 10000

        def __init__(self, *args, **kwargs) -> None:
            self._port = None

        def setsockopt(self, *args, **kwargs) -> None:
            return None

        def bind(self, addr) -> None:
            _, port = addr
            if port == 0:
                FakeSocket._next_port += 1
                port = FakeSocket._next_port
            self._port = port

        def listen(self, backlog) -> None:
            return None

        def getsockname(self):
            return ("127.0.0.1", self._port)

        def close(self) -> None:
            return None

    return FakeSocket


def test_mcp_server_start_stop(tmp_path, monkeypatch):
    _patch_uvicorn(monkeypatch)
    logger = logging.getLogger("test.mcp_server")
    server = mcp_srv.MCPServer(
        FakeMCP(),
        host="127.0.0.1",
        port=12345,
        log_dir=tmp_path,
        logger=logger,
    )

    server.start(timeout=1.0)
    assert server.server is not None
    assert server.thread is not None and server.thread.is_alive()

    server.stop(timeout=1.0)
    assert server.server.should_exit is True
    assert server.thread is None or not server.thread.is_alive()

    server.stop(timeout=1.0)


def test_mcp_server_start_stop_idempotent(tmp_path, monkeypatch):
    _patch_uvicorn(monkeypatch)
    logger = logging.getLogger("test.mcp_server.idempotent")
    server = mcp_srv.MCPServer(
        FakeMCP(),
        log_dir=tmp_path,
        logger=logger,
    )

    server.start(timeout=1.0)
    first_thread = server.thread
    server.start(timeout=1.0)
    assert server.thread is first_thread

    server.stop(timeout=1.0)
    server.stop(timeout=1.0)


def test_mcp_server_start_tcp_failure_stops(tmp_path, monkeypatch):
    _patch_uvicorn(monkeypatch)
    logger = logging.getLogger("test.mcp_server.tcp_fail")
    server = mcp_srv.MCPServer(
        FakeMCP(),
        log_dir=tmp_path,
        logger=logger,
    )

    def _raise_tcp(*args, **kwargs):
        raise TimeoutError("tcp timeout")

    monkeypatch.setattr(mcp_srv, "wait_for_tcp", _raise_tcp)

    with pytest.raises(TimeoutError, match="tcp timeout"):
        server.start(timeout=1.0)

    assert server.thread is None or not server.thread.is_alive()


def test_mcp_server_start_ping_failure_stops(tmp_path, monkeypatch):
    _patch_uvicorn(monkeypatch)
    logger = logging.getLogger("test.mcp_server.ping_fail")
    server = mcp_srv.MCPServer(
        FakeMCP(),
        log_dir=tmp_path,
        logger=logger,
    )

    def _raise_timeout(coro, timeout=None):
        coro.close()
        raise TimeoutError("ping timeout")

    monkeypatch.setattr(mcp_srv, "run_sync", _raise_timeout)

    with pytest.raises(TimeoutError, match="ping timeout"):
        server.start(timeout=1.0)

    assert server.thread is None or not server.thread.is_alive()


def test_mcp_server_allocates_unique_ports(tmp_path, monkeypatch):
    _patch_uvicorn(monkeypatch)
    logger = logging.getLogger("test.mcp_server.ports")

    ports = set()
    servers = []
    for _ in range(50):
        server = mcp_srv.MCPServer(
            FakeMCP(),
            log_dir=tmp_path,
            logger=logger,
        )
        servers.append(server)
        server.start(timeout=1.0)
        ports.add(server.port)

    assert len(ports) == len(servers)
    for server in servers:
        server.stop(timeout=1.0)


def test_mcp_server_start_timeout(tmp_path, monkeypatch):
    _patch_uvicorn(monkeypatch)
    logger = logging.getLogger("test.mcp_server.timeout")
    server = mcp_srv.MCPServer(
        FakeMCP(),
        host="127.0.0.1",
        port=12345,
        log_dir=tmp_path,
        logger=logger,
    )

    def _no_start():
        time.sleep(0.2)

    monkeypatch.setattr(server, "_thread_entry", _no_start)

    with pytest.raises(RuntimeError, match="did not signal startup"):
        server.start(timeout=0.05)

    if server.thread and server.thread.is_alive():
        server.thread.join(timeout=0.5)


async def _call_tool(app, name: str, arguments: dict, read_timeout_seconds=None):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        base_url = "http://127.0.0.1:9999"
        async with httpx.AsyncClient(
            transport=transport,
            base_url=base_url,
        ) as client:
            async with streamable_http_client(
                f"{base_url}/mcp",
                http_client=client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await session.call_tool(
                        name,
                        arguments=arguments,
                        read_timeout_seconds=read_timeout_seconds,
                    )


def test_mcp_server_unknown_tool_returns_mcp_error(tmp_path):
    logger = logging.getLogger("test.mcp_server.unknown_tool")

    def echo(text: str) -> str:
        return text

    server = mcp_srv.MCPServer(
        tools=[echo],
        log_dir=tmp_path,
        logger=logger,
    )
    app = server.mcp.streamable_http_app()

    result = run_sync(_call_tool(app, "does_not_exist", {}))
    assert result.isError is True
    assert any(block.text == "Unknown tool: does_not_exist" for block in result.content)


def test_mcp_server_invalid_tool_argument_name(tmp_path):
    logger = logging.getLogger("test.mcp_server.bad_arg_name")

    def echo(text: str) -> str:
        return text

    server = mcp_srv.MCPServer(
        tools=[echo],
        log_dir=tmp_path,
        logger=logger,
    )
    app = server.mcp.streamable_http_app()

    result = run_sync(_call_tool(app, "echo", {"wrong": "hi"}))
    assert result.isError is True
    assert any(
        "Error executing tool echo:" in block.text and "validation error" in block.text.lower()
        for block in result.content
    )


def test_mcp_server_invalid_tool_argument_type(tmp_path):
    logger = logging.getLogger("test.mcp_server.bad_arg_type")

    def echo(text: str) -> str:
        return text

    server = mcp_srv.MCPServer(
        tools=[echo],
        log_dir=tmp_path,
        logger=logger,
    )
    app = server.mcp.streamable_http_app()

    result = run_sync(_call_tool(app, "echo", {"text": 123}))
    assert result.isError is True
    assert any(
        "Error executing tool echo:" in block.text and "validation error" in block.text.lower()
        for block in result.content
    )


def test_mcp_server_call_tool_timeout(tmp_path):
    logger = logging.getLogger("test.mcp_server.call_timeout")

    def slow(text: str) -> str:
        time.sleep(0.2)
        return text

    server = mcp_srv.MCPServer(
        tools=[slow],
        log_dir=tmp_path,
        logger=logger,
    )
    app = server.mcp.streamable_http_app()

    timeout = timedelta(milliseconds=10)
    exc: BaseException | None = None
    try:
        run_sync(_call_tool(app, "slow", {"text": "hi"}, timeout))
    except BaseExceptionGroup as err:
        exc = err
    except McpError as err:
        exc = err

    assert exc is not None

    def _find_mcp_error(err: BaseException) -> McpError | None:
        if isinstance(err, McpError):
            return err
        if isinstance(err, BaseExceptionGroup):
            for sub in err.exceptions:
                found = _find_mcp_error(sub)
                if found is not None:
                    return found
        return None

    mcp_error = _find_mcp_error(exc)
    assert mcp_error is not None
    assert "Timed out while waiting for response" in str(mcp_error)
