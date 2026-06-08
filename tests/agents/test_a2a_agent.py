# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the A2A agent adapter.

Validates:
- A2A message format construction
- Response parsing (structured JSON and plain text fallback)
- Agent config and registry integration
- Action schema serialization
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exgentic.core.types import ActionType, MessageAction, SingleAction
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class DummyArgs(BaseModel):
    query: str


class DummyAction(SingleAction):
    name: str = "search"
    arguments: DummyArgs


SEARCH_ACTION_TYPE = ActionType(
    name="search",
    description="Search for information",
    cls=DummyAction,
)


class FinishArgs(BaseModel):
    answer: str


class FinishAction(SingleAction):
    name: str = "finish"
    arguments: FinishArgs


FINISH_ACTION_TYPE = ActionType(
    name="finish",
    description="Submit your final answer",
    cls=FinishAction,
    is_finish=True,
)

ALL_ACTIONS = [SEARCH_ACTION_TYPE, FINISH_ACTION_TYPE]


# ---------------------------------------------------------------------------
# Unit tests – response parsing
# ---------------------------------------------------------------------------


class TestParseAgentResponse:
    """Test _parse_agent_response from the adapter module."""

    def _parse(self, text: str):
        from exgentic.adapters.agents.a2a_agent import _parse_agent_response

        return _parse_agent_response(text, ALL_ACTIONS)

    def test_valid_json_action(self):
        resp = json.dumps({"action": "search", "arguments": {"query": "hello"}})
        action = self._parse(resp)
        assert action is not None
        assert isinstance(action, SingleAction)
        assert action.name == "search"

    def test_valid_json_finish(self):
        resp = json.dumps({"action": "finish", "arguments": {"answer": "42"}})
        action = self._parse(resp)
        assert action is not None
        assert action.name == "finish"

    def test_json_in_code_block(self):
        resp = '```json\n{"action": "search", "arguments": {"query": "test"}}\n```'
        action = self._parse(resp)
        assert action is not None
        assert action.name == "search"

    def test_plain_text_fallback(self):
        action = self._parse("I think the answer is 42")
        assert action is not None
        assert isinstance(action, MessageAction)
        assert action.arguments.content == "I think the answer is 42"

    def test_empty_response(self):
        action = self._parse("")
        assert action is None

    def test_unknown_action_name(self):
        resp = json.dumps({"action": "fly_to_moon", "arguments": {}})
        action = self._parse(resp)
        assert isinstance(action, MessageAction)

    def test_json_no_action_key(self):
        resp = json.dumps({"result": "something"})
        action = self._parse(resp)
        assert isinstance(action, MessageAction)


# ---------------------------------------------------------------------------
# Unit tests – action schema serialization
# ---------------------------------------------------------------------------


class TestActionTypesToSchema:
    def test_serializes_action_types(self):
        from exgentic.adapters.agents.a2a_agent import _action_types_to_schema

        schemas = _action_types_to_schema(ALL_ACTIONS)
        assert len(schemas) == 2
        assert schemas[0]["name"] == "search"
        assert schemas[0]["description"] == "Search for information"
        assert "parameters" in schemas[0]
        assert schemas[1]["name"] == "finish"


# ---------------------------------------------------------------------------
# Unit tests – system prompt construction
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_includes_task_and_context(self):
        from exgentic.adapters.agents.a2a_agent import _build_system_prompt

        prompt = _build_system_prompt(
            task="Find the capital of France",
            context={"instructions": "Be concise"},
            action_schemas=[{"name": "search", "description": "Search"}],
        )
        assert "Find the capital of France" in prompt
        assert "<instructions>" in prompt
        assert "Be concise" in prompt
        assert '"search"' in prompt

    def test_empty_context(self):
        from exgentic.adapters.agents.a2a_agent import _build_system_prompt

        prompt = _build_system_prompt(
            task="Do something",
            context={},
            action_schemas=[],
        )
        assert "Do something" in prompt


# ---------------------------------------------------------------------------
# Unit tests – observation to text conversion
# ---------------------------------------------------------------------------


class TestObservationToText:
    def _make_instance(self):
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance

        with patch.object(A2AAgentInstance, "__init__", lambda self, *a, **kw: None):
            inst = A2AAgentInstance.__new__(A2AAgentInstance)
            inst._actions = ALL_ACTIONS
            inst._step_count = 0
            inst.max_steps = 150
            # Set up logger
            inst._logger = MagicMock()
        return inst

    def test_none_observation(self):
        inst = self._make_instance()
        text = inst._observation_to_text(None)
        assert text is not None
        assert "No observation" in text

    def test_empty_observation(self):
        from exgentic.core.types import EmptyObservation

        inst = self._make_instance()
        obs = EmptyObservation()
        text = inst._observation_to_text(obs)
        assert "no output" in text.lower() or "empty" in text.lower()


# ---------------------------------------------------------------------------
# Integration – A2AAgent config & registry
# ---------------------------------------------------------------------------


class TestA2AAgentConfig:
    def test_slug_and_display_name(self):
        from exgentic.agents.a2a.agent import A2AAgent

        assert A2AAgent.slug_name == "a2a"
        assert A2AAgent.display_name == "A2A Agent"

    def test_instance_class_ref(self):
        from exgentic.agents.a2a.agent import A2AAgent

        ref = A2AAgent._get_instance_class_ref()
        assert ref == "exgentic.adapters.agents.a2a_agent:A2AAgentInstance"

    def test_instance_kwargs(self):
        from exgentic.agents.a2a.agent import A2AAgent

        agent = A2AAgent(agent_url="http://localhost:8080")
        kwargs = agent._get_instance_kwargs(session_id="test-session")
        assert kwargs["session_id"] == "test-session"
        assert kwargs["agent_url"] == "http://localhost:8080"
        assert kwargs["max_steps"] == 150

    def test_custom_max_steps(self):
        from exgentic.agents.a2a.agent import A2AAgent

        agent = A2AAgent(agent_url="http://localhost:8080", max_steps=50)
        kwargs = agent._get_instance_kwargs(session_id="s1")
        assert kwargs["max_steps"] == 50


class TestA2ARegistryEntry:
    def test_agent_in_registry(self):
        from exgentic.interfaces.registry import AGENTS

        assert "a2a" in AGENTS
        entry = AGENTS["a2a"]
        assert entry.slug_name == "a2a"
        assert entry.display_name == "A2A Agent"
        assert entry.kind == "agent"

    def test_load_agent_class(self):
        from exgentic.interfaces.registry import load_agent

        cls = load_agent("a2a")
        assert cls.slug_name == "a2a"
        assert cls.display_name == "A2A Agent"


# ---------------------------------------------------------------------------
# Unit tests – A2AAgentInstance construction
# ---------------------------------------------------------------------------


class TestA2AAgentInstance:
    def test_construction(self):
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance

        inst = A2AAgentInstance(
            session_id="test-session",
            agent_url="http://localhost:8080",
            max_steps=100,
        )
        assert inst.session_id == "test-session"
        assert inst._agent_url == "http://localhost:8080"
        assert inst.max_steps == 100
        assert inst._client is None
        assert inst._task_id is None

    def test_max_steps_exceeded_returns_none(self):
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance

        inst = A2AAgentInstance(
            session_id="test-session",
            agent_url="http://localhost:8080",
            max_steps=1,
        )
        inst._actions = ALL_ACTIONS
        inst._step_count = 2  # Already past max

        result = inst.react(None)
        assert result is None

    def test_close_without_client(self):
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance

        inst = A2AAgentInstance(
            session_id="test-session",
            agent_url="http://localhost:8080",
        )
        # Should not raise
        inst.close()
