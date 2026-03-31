# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Integration test: run_session() calls all observer lifecycle hooks in order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from exgentic.core.context import Context, set_context
from exgentic.core.orchestrator.observer import Observer
from exgentic.core.orchestrator.session import run_session
from exgentic.core.orchestrator.tracker import Tracker
from exgentic.core.types import Action, Observation

# ---------------------------------------------------------------------------
# Recording observer — logs every hook call
# ---------------------------------------------------------------------------


class RecordingObserver(Observer):
    """Observer that records (method_name, …) for each hook invocation."""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def on_session_creation(self, session):
        self.calls.append("on_session_creation")

    def on_session_start(self, session, agent, observation):
        self.calls.append("on_session_start")

    def on_react_success(self, session, action):
        self.calls.append("on_react_success")

    def on_step_success(self, session, observation):
        self.calls.append("on_step_success")

    def on_react_error(self, session, error):
        self.calls.append("on_react_error")

    def on_step_error(self, session, error):
        self.calls.append("on_step_error")

    def on_session_error(self, session, error):
        self.calls.append("on_session_error")

    def on_session_success(self, session, score, agent):
        self.calls.append("on_session_success")

    def on_session_scoring(self, session):
        self.calls.append("on_session_scoring")


# ---------------------------------------------------------------------------
# Minimal mock objects — use real Action/Observation base classes so
# CoreController (always added by Tracker) accepts them.
# ---------------------------------------------------------------------------


class StubAction(Action):
    """Minimal concrete Action for testing."""

    def to_action_list(self):
        return [self]


class StubObservation(Observation):
    """Minimal concrete Observation for testing."""

    text: str = "hello"

    def to_observation_list(self):
        return [self]

    def is_empty(self) -> bool:
        return False


@dataclass
class MockScore:
    success: bool = True
    score: float = 1.0
    is_finished: bool | None = None
    session_metadata: dict | None = None


class MockAgentInstance:
    agent_id = "agent-001"

    def start(self, *, task, context, actions):
        pass

    def react(self, observation):
        return StubAction()

    def close(self):
        pass

    def get_cost(self):
        return {"total": 0.0}


class MockAgent:
    def get_instance(self, session_id):
        return MockAgentInstance()


class MockSession:
    session_id = "sess-001"
    task_id = "task-001"
    task = "Test task"
    actions: ClassVar[list] = [StubAction()]
    context: ClassVar[dict] = {"key": "value"}

    def __init__(self):
        self._step = 0

    def start(self):
        return StubObservation()

    def done(self):
        return True  # end immediately → BenchmarkTerminationError path

    def step(self, action):
        return StubObservation()

    def score(self):
        return MockScore()

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctx(tmp_path):
    c = Context(run_id="test-run", output_dir=str(tmp_path), cache_dir=str(tmp_path))
    set_context(c)
    yield c
    set_context(None)


def test_observer_lifecycle_order(ctx, tmp_path):
    """run_session() calls observer hooks in the correct order."""
    observer = RecordingObserver()
    run_session(
        session_config=MagicMock(),
        session=MockSession(),
        agent=MockAgent(),
        tracker=Tracker(observers=[observer], use_defaults=False),
    )

    assert observer.calls == [
        "on_session_creation",
        "on_session_start",
        "on_session_scoring",
        "on_session_success",
    ]


def test_observer_lifecycle_with_steps(ctx, tmp_path):
    """run_session() calls react/step hooks when the session runs multiple steps."""

    class MultiStepSession(MockSession):
        def __init__(self):
            self._steps = 0

        def done(self):
            # Run for 2 steps, then mark done
            return self._steps >= 2

        def step(self, action):
            self._steps += 1
            return StubObservation()

    observer = RecordingObserver()
    run_session(
        session_config=MagicMock(),
        session=MultiStepSession(),
        agent=MockAgent(),
        tracker=Tracker(observers=[observer], use_defaults=False),
    )

    assert observer.calls == [
        "on_session_creation",
        "on_session_start",
        "on_react_success",
        "on_step_success",
        "on_react_success",
        "on_step_success",
        "on_session_scoring",
        "on_session_success",
    ]


def test_on_session_creation_called_before_start(ctx, tmp_path):
    """on_session_creation must be called before on_session_start."""
    observer = RecordingObserver()
    run_session(
        session_config=MagicMock(),
        session=MockSession(),
        agent=MockAgent(),
        tracker=Tracker(observers=[observer], use_defaults=False),
    )

    creation_idx = observer.calls.index("on_session_creation")
    start_idx = observer.calls.index("on_session_start")
    assert creation_idx < start_idx, "on_session_creation must precede on_session_start"
