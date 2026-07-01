# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the A2A agent adapter."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any
from unittest.mock import patch

import pytest

from exgentic.agents.a2a.a2a_agent import A2AAgent
from exgentic.agents.a2a.a2a_instance import A2AAgentInstance, A2AError
from exgentic.core.types import MessageAction, Observation, SingleObservation


# ---------------------------------------------------------------------------
# Fixtures: lightweight HTTP server that mimics an A2A agent
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Test Agent",
    "description": "A test A2A agent",
    "version": "1.0.0",
    "protocolVersion": "1.0",
    "url": "http://localhost/a2a",
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [{"id": "echo", "name": "Echo", "description": "Echoes input"}],
}


def _make_a2a_response(text: str, request_id: str = "1") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "id": "task-1",
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "artifactId": "art-1",
                    "parts": [{"kind": "text", "text": text}],
                }
            ],
        },
    }


def _make_a2a_error(code: int, message: str, request_id: str = "1") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


class _A2AHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves an agent card and echoes messages."""

    response_override: dict[str, Any] | None = None

    def do_GET(self):  # noqa: N802
        if self.path == "/.well-known/agent-card.json":
            body = json.dumps(AGENT_CARD).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.write(body)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path == "/message:send":
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            request = json.loads(raw)

            if self.response_override is not None:
                body = json.dumps(self.response_override).encode()
            else:
                # Echo back the user message text
                text = request["params"]["message"]["parts"][0]["text"]
                body = json.dumps(
                    _make_a2a_response(f"echo: {text}", request.get("id", "1"))
                ).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.write(body)
        else:
            self.send_error(404)

    def write(self, data: bytes):
        self.wfile.write(data)

    def log_message(self, format, *args):
        """Suppress request logging during tests."""
        pass


@pytest.fixture()
def a2a_server():
    """Start a local A2A test server and yield its base URL."""
    _A2AHandler.response_override = None
    server = HTTPServer(("127.0.0.1", 0), _A2AHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# A2AAgent (config class) tests
# ---------------------------------------------------------------------------


class TestA2AAgent:
    def test_slug_and_display_name(self):
        agent = A2AAgent(url="https://example.com")
        assert agent.slug_name == "a2a"
        assert agent.display_name == "A2A Agent"

    def test_model_name(self):
        agent = A2AAgent(url="https://example.com")
        assert agent.model_name == "a2a"
        assert agent.get_models_names() == ["a2a"]

    def test_instance_class_ref(self):
        ref = A2AAgent._get_instance_class_ref()
        assert ref == "exgentic.agents.a2a.a2a_instance:A2AAgentInstance"

    def test_instance_kwargs(self):
        agent = A2AAgent(url="https://example.com", timeout=30.0)
        kwargs = agent._get_instance_kwargs(session_id="s1")
        assert kwargs == {
            "session_id": "s1",
            "url": "https://example.com",
            "timeout": 30.0,
        }


# ---------------------------------------------------------------------------
# A2AAgentInstance tests
# ---------------------------------------------------------------------------


class TestA2AAgentInstance:
    def test_discover_agent_card(self, a2a_server):
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        card = instance._discover_agent_card()
        assert card["name"] == "Test Agent"
        assert card["version"] == "1.0.0"
        # Second call returns cached card
        assert instance._discover_agent_card() is card

    def test_send_message_echo(self, a2a_server):
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        response = instance._send_message("hello world")
        assert response == "echo: hello world"

    def test_send_message_rpc_error(self, a2a_server):
        _A2AHandler.response_override = _make_a2a_error(-32600, "Invalid request")
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        with pytest.raises(A2AError, match="Invalid request"):
            instance._send_message("test")

    def test_send_message_no_artifacts(self, a2a_server):
        _A2AHandler.response_override = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"id": "task-1", "status": {"state": "completed"}, "artifacts": []},
        }
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        with pytest.raises(A2AError, match="No text parts"):
            instance._send_message("test")

    def test_react_returns_message_action(self, a2a_server):
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        instance.start(task="What is 1+1?", context={}, actions=[])

        obs = SingleObservation(result="initial observation")
        action = instance.react(obs)

        assert isinstance(action, MessageAction)
        assert "echo: What is 1+1?" in action.arguments.content

    def test_react_includes_context(self, a2a_server):
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        instance.start(
            task="Solve this.",
            context={"instructions": "Be concise"},
            actions=[],
        )

        action = instance.react(SingleObservation(result=""))
        assert isinstance(action, MessageAction)
        # The prompt should include context tags
        assert "Be concise" in action.arguments.content
        assert "Solve this." in action.arguments.content

    def test_react_returns_none_on_second_call(self, a2a_server):
        instance = A2AAgentInstance(session_id="s1", url=a2a_server)
        instance.start(task="test", context={}, actions=[])

        action1 = instance.react(SingleObservation(result=""))
        assert action1 is not None

        action2 = instance.react(SingleObservation(result=""))
        assert action2 is None

    def test_extract_response_text_multiple_artifacts(self):
        rpc_response = {
            "result": {
                "artifacts": [
                    {"parts": [{"kind": "text", "text": "line 1"}]},
                    {"parts": [{"kind": "text", "text": "line 2"}]},
                ]
            }
        }
        text = A2AAgentInstance._extract_response_text(rpc_response)
        assert text == "line 1\nline 2"

    def test_extract_response_text_skips_non_text_parts(self):
        rpc_response = {
            "result": {
                "artifacts": [
                    {
                        "parts": [
                            {"kind": "image", "data": "..."},
                            {"kind": "text", "text": "answer"},
                        ]
                    }
                ]
            }
        }
        text = A2AAgentInstance._extract_response_text(rpc_response)
        assert text == "answer"

    def test_url_trailing_slash_stripped(self):
        instance = A2AAgentInstance(session_id="s1", url="https://example.com/")
        assert instance.url == "https://example.com"

    def test_close_is_noop(self):
        instance = A2AAgentInstance(session_id="s1", url="https://example.com")
        instance.close()  # Should not raise
