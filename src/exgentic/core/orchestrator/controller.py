# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import threading
from dataclasses import dataclass

from ..types import Action, Observation
from .cleanup import close_aiohttp_sessions_silently
from .termination import (
    AgentError,
    AgentTerminationError,
    BenchmarkError,
    BenchmarkTerminationError,
    InvalidActionError,
    InvalidObservationError,
)


class Controller:
    def on_run_success(self, results, run_config) -> None:
        return None

    def on_run_error(self, error) -> None:
        return None

    def on_react_success(self, session, action) -> None:
        return None

    def on_step_success(self, session, observation) -> None:
        return None

    def on_react_error(self, session, error) -> None:
        return None

    def on_step_error(self, session, error) -> None:
        return None


class CoreController(Controller):
    def on_react_success(self, session, action) -> None:
        if action is None:
            raise BenchmarkTerminationError()
        if not isinstance(action, Action):
            raise AgentError(InvalidActionError(action))

    def on_step_success(self, session, observation) -> None:
        if observation is None:
            raise AgentTerminationError()
        if not isinstance(observation, Observation):
            raise BenchmarkError(InvalidObservationError(observation))


@dataclass
class _LimitState:
    steps: int = 0
    actions: int = 0


class LimitController(Controller):
    def __init__(self, *, max_steps: int, max_actions: int) -> None:
        self._max_steps = max_steps
        self._max_actions = max_actions
        self._lock = threading.Lock()
        self._counts: dict[str, _LimitState] = {}

    def on_react_success(self, session, action) -> None:
        if not isinstance(action, Action):
            return
        session_id = session.session_id
        action_count = len(action.to_action_list())
        with self._lock:
            state = self._counts.get(session_id)
            if state is None:
                state = _LimitState()
                self._counts[session_id] = state
            state.steps += 1
            state.actions += action_count

    def on_step_success(self, session, observation) -> None:
        if observation is None:
            return
        session_id = session.session_id
        with self._lock:
            state = self._counts.get(session_id)
        if state is None:
            return
        if state.steps >= self._max_steps:
            from .termination import SessionLimitReachedError

            raise SessionLimitReachedError(
                reason="max_steps",
                max_steps=self._max_steps,
                max_actions=self._max_actions,
                steps=state.steps,
                actions=state.actions,
            )
        if state.actions >= self._max_actions:
            from .termination import SessionLimitReachedError

            raise SessionLimitReachedError(
                reason="max_actions",
                max_steps=self._max_steps,
                max_actions=self._max_actions,
                steps=state.steps,
                actions=state.actions,
            )


class CleanupController(Controller):
    def on_run_success(self, results, run_config) -> None:
        close_aiohttp_sessions_silently()

    def on_run_error(self, error) -> None:
        close_aiohttp_sessions_silently()
