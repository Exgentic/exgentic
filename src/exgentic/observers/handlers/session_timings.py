# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Per-session LLM call timing accumulator.

Single source of truth for LLM call wall-time data exposed in
`SessionResults`. Written by the LiteLLM trace logger on every call;
read by the results observer when finalizing a session.
"""

from __future__ import annotations

import threading


class SessionLLMTimings:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, list[float]] = {}

    def record(self, session_id: str, duration_seconds: float) -> None:
        if not session_id or duration_seconds < 0:
            return
        with self._lock:
            self._durations.setdefault(session_id, []).append(duration_seconds)

    def pop(self, session_id: str) -> list[float]:
        with self._lock:
            return self._durations.pop(session_id, [])


_INSTANCE = SessionLLMTimings()


def get_session_llm_timings() -> SessionLLMTimings:
    return _INSTANCE
