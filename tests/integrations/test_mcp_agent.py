# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import threading
import time
from typing import Literal

import pytest
from exgentic.adapters.actions.functions import action_type_to_function
from exgentic.adapters.agents import mcp_agent as mcp
from exgentic.core.types import (
    ActionType,
    MultiObservation,
    ParallelAction,
    SingleAction,
    SingleObservation,
)
from pydantic import BaseModel


class FakeMCPServer:
    def __init__(self, mcp=None, *args, **kwargs) -> None:
        tools = kwargs.get("tools") or []
        if mcp is None:
            mcp = FakeMCP()
        for fn in tools:
            mcp.tool(fn)
        self.mcp = mcp
        self.started = False
        self.stopped = False
        self.stop_calls = []
        self.host = "127.0.0.1"
        self.connect_host = "127.0.0.1"
        self.port = 12345

    def start(self, timeout: float = 5.0) -> None:
        self.started = True

    def stop(
        self,
        timeout: float = 10.0,
        *,
        error: BaseException | None = None,
        raise_on_timeout: bool = True,
    ) -> None:
        self.stopped = True
        self.stop_calls.append((timeout, error, raise_on_timeout))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop(error=exc, raise_on_timeout=True)


class FakeMCP:
    def __init__(self) -> None:
        self.tools: list = []

    def tool(self, fn):
        self.tools.append(fn)


class DummyMCPAgent(mcp.MCPAgent):
    def run_mcp_agent(self, mcp_host: str, mcp_port: int) -> str:
        self._last_mcp = (mcp_host, mcp_port)
        return "ok"


class ErrorMCPAgent(mcp.MCPAgent):
    def run_mcp_agent(self, mcp_host: str, mcp_port: int) -> str:
        raise ValueError("boom")


def _patch_mcp(monkeypatch):
    monkeypatch.setattr(mcp, "MCPServer", FakeMCPServer)


@pytest.fixture
def env(tmp_path):
    from exgentic.core.context import run_scope

    with run_scope(run_id="test_run", output_dir=str(tmp_path)):
        yield tmp_path


def test_run_code_agent_success_stops_server(env, monkeypatch):
    _patch_mcp(monkeypatch)
    agent = DummyMCPAgent("session")
    agent.task = "task"
    agent.context = {}
    agent.actions = []

    result = agent.run_code_agent([lambda: None])

    assert result == "ok"
    assert isinstance(agent.mcp, FakeMCP)
    assert agent.mcp.tools
    assert agent._mcp_server is not None
    assert agent._mcp_server.started is True
    assert agent._mcp_server.stopped is True


def test_run_code_agent_propagates_error_and_cleans_up(env, monkeypatch):
    _patch_mcp(monkeypatch)
    agent = ErrorMCPAgent("session")
    agent.task = "task"
    agent.context = {}
    agent.actions = []

    with pytest.raises(ValueError, match="boom"):
        agent.run_code_agent([lambda: None])

    assert agent._mcp_server is not None
    assert agent._mcp_server.started is True
    assert agent._mcp_server.stopped is True


def test_run_code_agent_ping_timeout_still_cleans_up(env, monkeypatch):
    _patch_mcp(monkeypatch)

    def _raise_timeout(self, timeout: float = 5.0) -> None:
        raise TimeoutError("ping timeout")

    monkeypatch.setattr(FakeMCPServer, "start", _raise_timeout)
    agent = DummyMCPAgent("session")
    agent.task = "task"
    agent.context = {}
    agent.actions = []

    with pytest.raises(TimeoutError, match="ping timeout"):
        agent.run_code_agent([lambda: None])

    assert agent._mcp_server is not None
    assert agent._mcp_server.started is False
    assert agent._mcp_server.stopped is False


class ToolArgs(BaseModel):
    value: int


class ToolA(SingleAction):
    name: Literal["tool.a"] = "tool.a"
    arguments: ToolArgs


class ToolB(SingleAction):
    name: Literal["tool.b"] = "tool.b"
    arguments: ToolArgs


class ParallelToolMCPAgent(mcp.MCPAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results: list = []

    def run_mcp_agent(self, mcp_host: str, mcp_port: int) -> str:
        tools = list(self.mcp.tools)
        results = []

        def _call(fn, value):
            results.append(fn(value=value))

        t1 = threading.Thread(target=_call, args=(tools[0], 1))
        t2 = threading.Thread(target=_call, args=(tools[1], 2))
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        if t1.is_alive() or t2.is_alive():
            raise RuntimeError("Tool calls did not complete")
        self.results = results
        return "ok"


def test_mcp_agent_parallel_tool_calls_return_parallel_action(env, monkeypatch):
    _patch_mcp(monkeypatch)

    actions = [
        ActionType(name="tool.a", description="tool a", cls=ToolA),
        ActionType(name="tool.b", description="tool b", cls=ToolB),
    ]
    agent = ParallelToolMCPAgent("session")
    agent.task = "task"
    agent.context = {}
    agent.actions = actions
    functions = [action_type_to_function(act, agent.execute) for act in actions]

    worker = threading.Thread(target=agent.run_code_agent, args=(functions,))
    worker.start()

    deadline = time.time() + 1.0
    with agent._condition:
        while len(agent._pending_actions) < 2 and time.time() < deadline:
            agent._condition.wait(timeout=0.05)
        pending = list(agent._pending_actions)

    observations = [
        SingleObservation(invoking_actions=[action], result={"ok": action.arguments.value})
        for action in pending
        if isinstance(action, SingleAction)
    ]
    act = agent.react(MultiObservation(observations=observations))
    assert isinstance(act, ParallelAction)
    worker.join(timeout=1.0)
    if worker.is_alive():
        agent.close()
        pytest.fail("MCP agent did not finish after parallel tool calls")
    agent.close()

    assert len(agent.results) == 2
    assert all(isinstance(r, dict) and r.get("ok") in (1, 2) for r in agent.results)
