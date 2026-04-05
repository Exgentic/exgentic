# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""ReplayBenchmark — replays a full recorded session (agent + environment).

Pairs with ReplayAgent + ReplaySession to test the full execution loop
without needing any benchmark dependencies installed.

Usage (from tests)::

    benchmark = ReplayBenchmark(recording_dir="path/to/recording")
    agent = ReplayAgent(recording="path/to/recording")
    results = evaluate(benchmark=benchmark, agent=agent)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from ...core.benchmark import Benchmark
from ...core.evaluator import Evaluator
from ...core.types import BenchmarkResults, SessionIndex
from .replay_session import ReplaySession


class ReplayEvaluator(Evaluator):
    """Evaluator that returns session kwargs for ReplaySession."""

    def __init__(self, recording_dir: str) -> None:
        self._recording_dir = recording_dir

    def list_tasks(self) -> list[str]:
        recording = Path(self._recording_dir)
        # Try to get task_id from session.json
        manifest_path = recording / "session.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            task_id = manifest.get("task_id", "0")
            return [str(task_id)]
        return ["0"]

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        paths = self.get_sessions_paths(sessions)
        scores = []
        for p in paths:
            results_path = p.benchmark_results
            if results_path.exists():
                data = json.loads(results_path.read_text())
                score = data.get("score")
                if score is not None:
                    scores.append(float(score))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResults(
            benchmark_name="replay",
            total_tasks=len(sessions),
            score=avg_score,
        )


class ReplayBenchmark(Benchmark):
    """Benchmark that replays recorded sessions from a directory."""

    display_name: ClassVar[str] = "Replay Benchmark"
    slug_name: ClassVar[str] = "replay"
    recording_dir: str

    @classmethod
    def _get_evaluator_class(cls):
        return ReplayEvaluator

    @classmethod
    def _get_session_class(cls):
        return ReplaySession

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {"recording_dir": self.recording_dir}

    def _get_session_kwargs(self) -> dict[str, Any]:
        return {"recording_dir": self.recording_dir}

    def runner_kwargs(self) -> dict[str, Any]:
        kw = super().runner_kwargs()
        if self.resolve_runner() == "docker":
            recording_dir = str(Path(self.recording_dir).resolve())
            kw.setdefault("volumes", {})[recording_dir] = recording_dir
        return kw
