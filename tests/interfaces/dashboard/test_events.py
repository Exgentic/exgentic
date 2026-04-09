# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from unittest.mock import MagicMock, patch

import pytest
from exgentic.core.orchestrator.termination import (
    AgentError,
    BenchmarkError,
    SessionCancelError,
)
from exgentic.interfaces.dashboard.views.runtime import process_new_events
from exgentic.interfaces.dashboard.views.state import RunState
from exgentic.observers.handlers.dashboard_events import DashboardEventsObserver


def _make_score(success=True, score=1.0, is_finished=True):
    mock = MagicMock()
    mock.success = success
    mock.score = score
    mock.is_finished = is_finished
    mock.model_dump.return_value = {
        "success": success,
        "score": score,
        "is_finished": is_finished,
    }
    return mock


def _make_session(session_id="sess-1"):
    mock = MagicMock()
    mock.session_id = session_id
    cost_report = MagicMock()
    cost_report.total_cost = 0.05
    mock.get_cost.return_value = cost_report
    return mock


def _make_agent():
    mock = MagicMock()
    cost_report = MagicMock()
    cost_report.total_cost = 0.10
    mock.get_cost.return_value = cost_report
    return mock


def _drain_events(observer, max_events=100):
    events = []
    while not observer.events.empty() and len(events) < max_events:
        events.append(observer.events.get_nowait())
    return events


class TestDashboardEventsObserver:
    def test_init(self):
        obs = DashboardEventsObserver()
        assert obs.events.empty()

    @patch("exgentic.observers.handlers.dashboard_events.get_context")
    def test_on_run_start_emits_run_meta(self, mock_ctx):
        mock_ctx.return_value.run_id = "run-123"
        obs = DashboardEventsObserver()
        obs.on_run_start(MagicMock())
        events = _drain_events(obs)
        run_meta = [e for e in events if e["type"] == "run_meta"]
        assert len(run_meta) == 1
        assert run_meta[0]["run_id"] == "run-123"

    @patch("exgentic.observers.handlers.dashboard_events.get_context")
    def test_run_meta_emitted_only_once(self, mock_ctx):
        mock_ctx.return_value.run_id = "run-123"
        obs = DashboardEventsObserver()
        obs.on_run_start(MagicMock())
        obs.on_run_start(MagicMock())
        events = _drain_events(obs)
        run_meta = [e for e in events if e["type"] == "run_meta"]
        assert len(run_meta) == 1

    def test_on_session_start(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        agent = _make_agent()
        obs.on_session_start(session, agent, None)
        events = _drain_events(obs)
        started = [e for e in events if e["type"] == "session_started"]
        assert len(started) == 1
        assert started[0]["session_id"] == "s1"

    def test_on_session_success(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        agent = _make_agent()
        obs.on_session_start(session, agent, None)
        _drain_events(obs)

        score = _make_score(success=True, score=0.9, is_finished=True)
        obs.on_session_success(session, score, agent)
        events = _drain_events(obs)
        finished = [e for e in events if e["type"] == "session_finished"]
        assert len(finished) == 1
        assert finished[0]["success"] is True
        assert finished[0]["score"] == 0.9
        assert finished[0]["agent_cost"] == pytest.approx(0.10)
        assert finished[0]["benchmark_cost"] == pytest.approx(0.05)

    def test_on_session_error_agent(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        agent = _make_agent()
        obs.on_session_start(session, agent, None)
        _drain_events(obs)

        error = AgentError(ValueError("boom"))
        obs.on_session_error(session, error)
        events = _drain_events(obs)
        finished = [e for e in events if e["type"] == "session_finished"]
        assert len(finished) == 1
        assert finished[0]["error_source"] == "agent"
        assert finished[0]["success"] is False

    def test_on_session_error_benchmark(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        obs.on_session_start(session, _make_agent(), None)
        _drain_events(obs)

        error = BenchmarkError(RuntimeError("fail"))
        obs.on_session_error(session, error)
        events = _drain_events(obs)
        finished = [e for e in events if e["type"] == "session_finished"]
        assert finished[0]["error_source"] == "benchmark"

    def test_on_session_error_cancelled(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        obs.on_session_start(session, _make_agent(), None)
        _drain_events(obs)

        error = SessionCancelError()
        obs.on_session_error(session, error)
        events = _drain_events(obs)
        finished = [e for e in events if e["type"] == "session_finished"]
        assert finished[0]["error_source"] == "cancelled"

    def test_on_react_success_emits_step(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        obs.on_session_start(session, _make_agent(), None)
        _drain_events(obs)

        action = MagicMock()
        obs.on_react_success(session, action)
        obs._flush_pending()
        events = _drain_events(obs)
        steps = [e for e in events if e["type"] == "step"]
        assert len(steps) == 1
        assert steps[0]["session_id"] == "s1"
        assert steps[0]["n"] == 1

    def test_on_step_success_emits_observation(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        obs.on_session_start(session, _make_agent(), None)
        _drain_events(obs)

        obs.on_step_success(session, "some observation")
        obs._flush_pending()
        events = _drain_events(obs)
        observations = [e for e in events if e["type"] == "observation"]
        assert len(observations) == 1
        assert observations[0]["session_id"] == "s1"

    def test_step_counter_increments(self):
        obs = DashboardEventsObserver()
        session = _make_session("s1")
        obs.on_session_start(session, _make_agent(), None)
        _drain_events(obs)

        for _ in range(3):
            obs.on_react_success(session, MagicMock())
        obs._flush_pending()
        events = _drain_events(obs)
        steps = [e for e in events if e["type"] == "step"]
        step_nums = [e["n"] for e in steps]
        assert step_nums == [1, 2, 3]

    @patch("exgentic.observers.handlers.dashboard_events.get_context")
    def test_on_run_success_emits_saved(self, mock_ctx):
        mock_ctx.return_value.run_id = "run-1"
        mock_ctx.return_value.output_dir = "/tmp/test"
        obs = DashboardEventsObserver()
        results = MagicMock()
        results.model_dump.return_value = {"score": 0.5}

        with patch("exgentic.observers.handlers.dashboard_events.RunPaths") as mock_paths:
            mock_paths.from_context.return_value.results = "/tmp/test/run-1/results.json"
            obs.on_run_success(results, MagicMock())

        events = _drain_events(obs)
        saved = [e for e in events if e["type"] == "saved"]
        assert len(saved) == 1


class TestProcessNewEvents:
    """Tests for process_new_events in runtime.py."""

    def _make_state_with_events(self, events):
        """Create a RunState with a tracker that has pre-loaded events."""
        state = RunState()
        tracker = DashboardEventsObserver()
        for evt in events:
            tracker.events.put_nowait(evt)
        state.tracker = tracker
        return state

    def test_session_started(self):
        state = self._make_state_with_events(
            [
                {"type": "session_started", "session_id": "s1", "ts": 0},
            ]
        )
        changed = process_new_events(state)
        assert changed is True
        assert "s1" in state.sessions
        assert state.sessions["s1"]["status"] == "running"

    def test_step_increments(self):
        state = self._make_state_with_events(
            [
                {"type": "session_started", "session_id": "s1", "ts": 0},
                {"type": "step", "session_id": "s1", "n": 1, "ts": 1},
                {"type": "step", "session_id": "s1", "n": 2, "ts": 2},
            ]
        )
        process_new_events(state)
        assert state.sessions["s1"]["steps"] == 2

    def test_session_finished_success(self):
        state = self._make_state_with_events(
            [
                {"type": "session_started", "session_id": "s1", "ts": 0},
                {
                    "type": "session_finished",
                    "session_id": "s1",
                    "success": True,
                    "is_finished": True,
                    "score": 1.0,
                    "steps": 5,
                    "details": {},
                    "ts": 1,
                },
            ]
        )
        process_new_events(state)
        assert state.sessions["s1"]["status"] == "success"
        assert state.sessions["s1"]["score"] == 1.0
        # No error trajectory entry for successful sessions
        assert "s1" not in state.turns

    def test_session_finished_unsuccessful_no_error_trajectory(self):
        """Unsuccessful sessions should not get error trajectory entries.

        Finished-but-not-successful sessions aren't errors.
        """
        state = self._make_state_with_events(
            [
                {"type": "session_started", "session_id": "s1", "ts": 0},
                {
                    "type": "session_finished",
                    "session_id": "s1",
                    "success": False,
                    "is_finished": True,
                    "score": 0.0,
                    "steps": 3,
                    "details": {"success": False, "score": 0.0, "is_finished": True},
                    "ts": 1,
                },
            ]
        )
        process_new_events(state)
        assert state.sessions["s1"]["status"] == "unsuccessful"
        # No error_source means no error trajectory entry
        assert "s1" not in state.turns

    def test_session_finished_with_error_gets_trajectory(self):
        """Sessions with an error_source should get an error trajectory entry."""
        state = self._make_state_with_events(
            [
                {"type": "session_started", "session_id": "s1", "ts": 0},
                {
                    "type": "session_finished",
                    "session_id": "s1",
                    "success": False,
                    "is_finished": None,
                    "score": None,
                    "steps": 2,
                    "error_source": "agent",
                    "details": {"error": "something broke", "error_source": "agent"},
                    "ts": 1,
                },
            ]
        )
        process_new_events(state)
        assert state.sessions["s1"]["status"] == "agent error"
        assert "s1" in state.turns
        error_entries = [t for t in state.turns["s1"] if t["type"] == "error"]
        assert len(error_entries) == 1
        assert error_entries[0]["content"] == "something broke"

    def test_observation_stored_in_turns(self):
        state = self._make_state_with_events(
            [
                {"type": "session_started", "session_id": "s1", "ts": 0},
                {"type": "observation", "session_id": "s1", "step": 1, "observation": {"result": "ok"}, "ts": 1},
            ]
        )
        process_new_events(state)
        assert "s1" in state.turns
        assert state.turns["s1"][0]["type"] == "observation"

    def test_no_tracker_returns_false(self):
        state = RunState()
        assert process_new_events(state) is False
