# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class SessionState:
    started_at: float
    steps: int = 0


class SessionLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter = 0
        self._numbers: dict[str, int] = {}
        self._states: dict[str, SessionState] = {}

    def register(self, session_id: str) -> int:
        with self._lock:
            number = self._numbers.get(session_id)
            if number is None:
                self._counter += 1
                number = self._counter
                self._numbers[session_id] = number
            self._states[session_id] = SessionState(started_at=time.time())
            return number

    def mark_reuse(self, session_id: str) -> int:
        with self._lock:
            number = self._numbers.get(session_id)
            if number is None:
                self._counter += 1
                number = self._counter
                self._numbers[session_id] = number
            return number

    def increment_steps(self, session_id: str, count: int = 1) -> int:
        with self._lock:
            state = self._states.get(session_id)
            if state is None:
                if session_id not in self._numbers:
                    self._counter += 1
                    self._numbers[session_id] = self._counter
                state = SessionState(started_at=time.time())
                self._states[session_id] = state
            state.steps += count
            return state.steps

    def get_steps(self, session_id: str) -> int:
        with self._lock:
            state = self._states.get(session_id)
            return state.steps if state is not None else 0

    def get_number(self, session_id: str) -> int:
        with self._lock:
            return self._numbers.get(session_id, 0)

    def pop_state(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._states.pop(session_id, None)
