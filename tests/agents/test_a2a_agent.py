# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for the A2A agent adapter.

Validates:
- A2A message format construction
- Response parsing (structured JSON and plain text fallback)
- Agent config and registry integration
- Action schema serialization
- Text extraction from A2A messages
- Multi-turn support via context_id
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

    def test_whitespace_only_response(self):
        action = self._parse("   \n\t  ")
        assert action is None

    def test_unknown_action_name(self):
        resp = json.dumps({"action": "fly_to_moon", "arguments": {}})
        action = self._parse(resp)
        assert isinstance(action, MessageAction)

    def test_json_no_action_key(self):
        resp = json.dumps({"result": "something"})
        action = self._parse(resp)
        assert isinstance(action, MessageAction)

    def test_json_with_name_key(self):
        """Supports 'name' as an alternative to 'action'."""
        resp = json.dumps({"name": "search", "arguments": {"query": "test"}})
        action = self._parse(resp)
        assert action is not None
        assert action.name == "search"

    def test_json_with_params_key(self):
        """Supports 'params' as an alternative to 'arguments'."""
        resp = json.dumps({"action": "finish", "params": {"answer": "42"}})
        action = self._parse(resp)
        assert action is not None
        assert action.name == "finish"

    def test_json_array_falls_back_to_message(self):
        """A JSON array (not dict) should fall back to message."""
        resp = json.dumps([1, 2, 3])
        action = self._parse(resp)
        assert isinstance(action, MessageAction)

    def test_code_block_with_language_tag(self):
        """Code block with ```json tag."""
        resp = "Here is my response:\n```json\n" + json.dumps(
            {"action": "finish", "arguments": {"answer": "done"}}
        ) + "\n```\nEnd."
        action = self._parse(resp)
        assert action is not None
        assert action.name == "finish"

    def test_code_block_without_language_tag(self):
        """Code block without language specifier."""
        resp = "```\n" + json.dumps(
            {"action": "search", "arguments": {"query": "x"}}
        ) + "\n```"
        action = self._parse(resp)
        assert action is not None
        assert action.name == "search"


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

    def test_action_without_arguments_model(self):
        """ActionType where arguments has no model_json_schema."""
        from exgentic.adapters.agents.a2a_agent import _action_types_to_schema

        class MinimalAction(SingleAction):
            name: str = "noop"
            arguments: dict = {}

        at = ActionType(
            name="noop", description="Do nothing", cls=MinimalAction
        )
        schemas = _action_types_to_schema([at])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "noop"
        # No 'parameters' key since dict doesn't have model_json_schema
        assert "parameters" not in schemas[0]


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

    def test_includes_json_response_instruction(self):
        from exgentic.adapters.agents.a2a_agent import _build_system_prompt

        prompt = _build_system_prompt(
            task="Test",
            context={},
            action_schemas=[{"name": "finish"}],
        )
        assert '"action"' in prompt
        assert '"arguments"' in prompt


# ---------------------------------------------------------------------------
# Unit tests – A2A message text extraction
# ---------------------------------------------------------------------------


class TestExtractTextFromMessage:
    def test_text_parts(self):
        from a2a.types import Message, Part, Role
        from exgentic.adapters.agents.a2a_agent import _extract_text_from_message

        msg = Message(
            role=Role.ROLE_AGENT,
            parts=[
                Part(text="Hello"),
                Part(text="World"),
            ],
            message_id="test-1",
        )
        text = _extract_text_from_message(msg)
        assert text == "Hello\nWorld"

    def test_empty_parts(self):
        from a2a.types import Message, Role
        from exgentic.adapters.agents.a2a_agent import _extract_text_from_message

        msg = Message(
            role=Role.ROLE_AGENT,
            parts=[],
            message_id="test-2",
        )
        text = _extract_text_from_message(msg)
        assert text == ""

    def test_single_text_part(self):
        from a2a.types import Message, Part, Role
        from exgentic.adapters.agents.a2a_agent import _extract_text_from_message

        msg = Message(
            role=Role.ROLE_AGENT,
            parts=[Part(text="Just one")],
            message_id="test-3",
        )
        assert _extract_text_from_message(msg) == "Just one"


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

    def test_single_observation_with_action(self):
        from exgentic.core.types import SingleObservation

        inst = self._make_instance()
        action = DummyAction(arguments=DummyArgs(query="test"))
        obs = SingleObservation(
            invoking_actions=[action],
            result="Search result: found 5 items",
        )
        text = inst._observation_to_text(obs)
        assert "search" in text.lower()
        assert "Search result: found 5 items" in text
        # Should include JSON action instruction
        assert '"action"' in text

    def test_single_observation_without_action(self):
        from exgentic.core.types import SingleObservation

        inst = self._make_instance()
        obs = SingleObservation(
            invoking_actions=[],
            result="Some result",
        )
        text = inst._observation_to_text(obs)
        assert "Some result" in text


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
        assert kwargs["timeout"] == 300.0

    def test_custom_max_steps(self):
        from exgentic.agents.a2a.agent import A2AAgent

        agent = A2AAgent(agent_url="http://localhost:8080", max_steps=50)
        kwargs = agent._get_instance_kwargs(session_id="s1")
        assert kwargs["max_steps"] == 50

    def test_model_name(self):
        from exgentic.agents.a2a.agent import A2AAgent

        agent = A2AAgent(agent_url="http://localhost:8080")
        assert agent.model_name == "a2a-external"


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
        inst.close()

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
        inst.close()

    def test_close_without_client(self):
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance

        inst = A2AAgentInstance(
            session_id="test-session",
            agent_url="http://localhost:8080",
        )
        # Should not raise
        inst.close()

    def test_context_id_is_uuid(self):
        """Each instance gets a unique context_id for session continuity."""
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance
        import uuid

        inst = A2AAgentInstance(
            session_id="test-session",
            agent_url="http://localhost:8080",
        )
        # Should be a valid UUID
        uuid.UUID(inst._context_id)
        inst.close()

    def test_two_instances_different_context_ids(self):
        """Two instances should have different context IDs."""
        from exgentic.adapters.agents.a2a_agent import A2AAgentInstance

        inst1 = A2AAgentInstance(
            session_id="s1", agent_url="http://localhost:8080"
        )
        inst2 = A2AAgentInstance(
            session_id="s2", agent_url="http://localhost:8080"
        )
        assert inst1._context_id != inst2._context_id
        inst1.close()
        inst2.close()


# ---------------------------------------------------------------------------
# Unit tests – _AsyncBridge
# ---------------------------------------------------------------------------


class TestAsyncBridge:
    def test_run_simple_coroutine(self):
        from exgentic.adapters.agents.a2a_agent import _AsyncBridge
        import asyncio

        bridge = _AsyncBridge()

        async def add(a, b):
            return a + b

        result = bridge.run(add(2, 3))
        assert result == 5
        bridge.shutdown()

    def test_multiple_calls_same_bridge(self):
        """Multiple calls on the same bridge use the same event loop."""
        from exgentic.adapters.agents.a2a_agent import _AsyncBridge

        bridge = _AsyncBridge()
        results = []

        async def get_loop_id():
            import asyncio
            return id(asyncio.get_running_loop())

        for _ in range(3):
            results.append(bridge.run(get_loop_id()))

        # All calls should use the same event loop
        assert len(set(results)) == 1
        bridge.shutdown()
