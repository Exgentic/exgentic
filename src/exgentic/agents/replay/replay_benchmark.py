# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""ReplayBenchmark — replays a full recorded session (agent + environment).

Pairs with ReplayAgent + ReplaySession to test the full execution loop
without needing any benchmark dependencies installed.

Usage (from tests)::

    from exgentic.agents.replay.replay_benchmark import ReplayBenchmark
    benchmark = ReplayBenchmark(recording_dir="path/to/recording")
    agent = ReplayAgent(recording="path/to/recording")
    results = evaluate(benchmark=benchmark, agent=agent)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Type

from ...core.benchmark import Benchmark
from ...core.evaluator import Evaluator
from ...core.types import BenchmarkResults, SessionIndex
from .replay_session import ReplaySession


class ReplayEvaluator(Evaluator):
    """Evaluator that returns session kwargs for ReplaySession."""

    def __init__(self, recording_dir: str) -> None:
        self._recording_dir = recording_dir

    def list_tasks(self) -> List[str]:
        recording = Path(self._recording_dir)
        # Try to get task_id from session.json
        manifest_path = recording / "session.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            task_id = manifest.get("task_id", "0")
            return [str(task_id)]
        return ["0"]

    def get_session_kwargs(self, index: SessionIndex) -> Dict[str, Any]:
        return {
            "recording_dir": self._recording_dir,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: List[SessionIndex]) -> BenchmarkResults:
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
    evaluator_class: ClassVar[Type[Evaluator]] = ReplayEvaluator
    session_class: ClassVar[Type[ReplaySession]] = ReplaySession

    recording_dir: str

    def get_evaluator_kwargs(self) -> Dict[str, Any]:
        return {"recording_dir": self.recording_dir}
