# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Liveness-check regression tests for HTTPTransport.

HTTPTransport takes an optional ``is_alive`` callable and checks it
before every RPC. If the callable returns False, an immediate
``ConnectionError`` is raised, so ``run_session``'s existing
step/react error paths kick in and the session fails cleanly instead
of hanging on a dead socket for the full transport timeout.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from exgentic.adapters.runners.service import HTTPTransport


class _EchoHandler(BaseHTTPRequestHandler):
    """Minimal stub server: any POST returns 200 with a fixed RPCResponse payload."""

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = b'{"status":"ok","result":null}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return  # silence stderr spam in tests


@pytest.fixture
def stub_server():
    """Bring up a short-lived HTTP server on an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _url(server: HTTPServer) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Regression: dead peer must fail fast
# ---------------------------------------------------------------------------


def test_is_alive_none_preserves_legacy_behavior(stub_server):
    """No ``is_alive`` supplied → transport behaves exactly as before."""
    transport = HTTPTransport(_url(stub_server), timeout=2.0)
    try:
        # Must succeed against a live stub.
        assert transport.call("anything") is None
    finally:
        transport.close()


def test_is_alive_true_allows_calls(stub_server):
    """``is_alive`` returning True is a no-op guard."""
    transport = HTTPTransport(_url(stub_server), timeout=2.0, is_alive=lambda: True)
    try:
        assert transport.call("anything") is None
    finally:
        transport.close()


def test_is_alive_false_raises_connection_error_on_call():
    """Dead peer: ``is_alive`` returns False → call raises ConnectionError immediately.

    Critically, the URL here points at a port nothing is listening on,
    so the test would *also* fail if we fell through to httpx — but the
    guard must fire *before* we get there, so the test must not hang on
    httpx's connect timeout.
    """
    transport = HTTPTransport("http://127.0.0.1:1", timeout=30.0, is_alive=lambda: False)
    t0 = time.monotonic()
    with pytest.raises(ConnectionError, match="not alive"):
        transport.call("doesnt_matter")
    # Must fail fast — not wait on the 30s transport timeout.
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"dead-peer guard should fail fast, took {elapsed:.2f}s"
    transport.close()


def test_is_alive_false_raises_on_get():
    transport = HTTPTransport("http://127.0.0.1:1", timeout=30.0, is_alive=lambda: False)
    with pytest.raises(ConnectionError, match="not alive"):
        _ = transport.get("some_attr")
    transport.close()


def test_is_alive_false_raises_on_set():
    transport = HTTPTransport("http://127.0.0.1:1", timeout=30.0, is_alive=lambda: False)
    with pytest.raises(ConnectionError, match="not alive"):
        transport.set("some_attr", 42)
    transport.close()


def test_is_alive_checked_per_call(stub_server):
    """Flag flip mid-session: transport stops serving once the peer dies."""
    alive = {"v": True}
    transport = HTTPTransport(_url(stub_server), timeout=2.0, is_alive=lambda: alive["v"])
    try:
        # First call succeeds.
        assert transport.call("anything") is None
        # Peer "dies".
        alive["v"] = False
        with pytest.raises(ConnectionError, match="not alive"):
            transport.call("anything_else")
    finally:
        transport.close()


def test_is_alive_exception_treated_as_dead():
    """Any error inside ``is_alive`` should be treated conservatively as dead.

    The caller is a black-box process-watcher; if it can't report, we
    can't safely assume the peer is alive. Defer to ConnectionError
    rather than risk a hang.
    """

    def explode() -> bool:
        raise RuntimeError("watcher broke")

    transport = HTTPTransport("http://127.0.0.1:1", timeout=30.0, is_alive=explode)
    with pytest.raises(ConnectionError, match="liveness check"):
        transport.call("anything")
    transport.close()
