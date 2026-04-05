# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Evaluator ABC — run-level benchmark operations (task discovery + aggregation).

An Evaluator is created via ``Benchmark.get_evaluator()`` for container
isolation, keeping heavy dependencies off the host.  It handles run-level
concerns only: listing tasks for planning and aggregating session results.
Per-task work happens in the ``Session`` class, which self-loads its own
task data given only ``task_id``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..utils.paths import SessionPaths, get_run_paths
from .types import BenchmarkResults, SessionIndex


class Evaluator(ABC):
    """Run-level benchmark operations — task discovery + session aggregation.

    Runs in the same isolation level as the benchmark's runner (can be containerized).
    Returns only simple serializable data across the transport boundary.
    """

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return available task identifiers for this benchmark."""
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
