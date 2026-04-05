# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Evaluator ABC — benchmark evaluation logic (task discovery, session config, aggregation).

An Evaluator is created via ``Benchmark.get_evaluator()`` for container
isolation, keeping heavy dependencies off the host.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..utils.paths import SessionPaths, get_run_paths
from .types import BenchmarkResults, SessionIndex


class Evaluator(ABC):
    """Benchmark evaluation logic — task discovery, session config, aggregation.

    Runs in the same isolation level as the benchmark's runner (can be containerized).
    Returns only simple serializable data across the transport boundary.
    """

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return available task identifiers for this benchmark."""
        ...

    @abstractmethod
    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        """Return kwargs for constructing the Session for a given task.

        The orchestrator will call::

            benchmark.get_session(**session_kwargs)
        """
        ...

    @abstractmethod
    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        """Aggregate results for the specified task sessions."""
        ...

    def get_sessions_paths(self, sessions: list[SessionIndex]) -> list[SessionPaths]:
        """Return ``SessionPaths`` for each session index."""
        run_paths = get_run_paths()
        return [run_paths.session(s.session_id) for s in sessions]

    def close(self) -> None:
        """Optional cleanup hook."""
        return
