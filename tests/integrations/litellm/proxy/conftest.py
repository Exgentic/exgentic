# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/v1/models"):
            payload = {"object": "list", "data": [{"id": "openai/gpt-4o-mini"}]}
            self._write_json(payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.endswith("/v1/chat/completions"):
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(body or "{}")
            FakeOpenAIHandler.request_count += 1
            model = data.get("model", "openai/gpt-4o-mini")
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            self._write_json(payload)
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _write_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_openai_server() -> ThreadingHTTPServer:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    except PermissionError as exc:
        pytest.skip(f"Socket binding not permitted: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
