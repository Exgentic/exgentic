# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from typing import Any, ClassVar

from ..core.benchmark import Benchmark
from ..core.evaluator import Evaluator
from ..core.session import Session
from ..core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    SessionIndex,
    SessionScore,
    SingleObservation,
)
from .agent import (
    BAD_ACTION_TYPE,
    FINISH_ACTION_TYPE,
    GOOD_ACTION_TYPE,
    BadAction,
    FinishAction,
    GoodAction,
)


class TestSession(Session):
    __test__ = False

    def __init__(
        self,
        *,
        task_id: str,
        session_id: str,
        stop_on_step: bool,
        invalid_observation: bool,
    ) -> None:
        self._session_id = session_id
        self._task_id = task_id
        self._stop_on_step = stop_on_step
        self._invalid_observation = invalid_observation
        self._done = False
        self._good = 0
        self._bad = 0
        self._steps = 0
        super().__init__()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        return f"Task {self._task_id}"

    @property
    def context(self) -> dict[str, Any]:
        return {"task_id": self._task_id}

    @property
    def actions(self) -> list[ActionType]:
        return [GOOD_ACTION_TYPE, BAD_ACTION_TYPE, FINISH_ACTION_TYPE]

    def start(self) -> SingleObservation:
        return SingleObservation(result="start")

    def step(self, action: Action) -> SingleObservation | None:
        self._steps += 1
        if self._invalid_observation:
            return "invalid-observation"  # type: ignore[return-value]
        if isinstance(action, GoodAction):
            self._good += 1
        elif isinstance(action, BadAction):
            self._bad += 1
        elif isinstance(action, FinishAction):
            self._done = True
            return SingleObservation(result="finish")
        if self._stop_on_step:
            return None
        return SingleObservation(result="step")

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        total = self._good + self._bad
        score = float(self._good / total) if total > 0 else 0.0
        success = bool(self._done and self._bad == 0 and total > 0)
        result = SessionScore(
            score=score,
            success=success,
            is_finished=self._done,
            session_metrics={"good": self._good, "bad": self._bad, "total": total},
            session_metadata={"steps": self._steps},
        )
        self.save_standard_results(result)
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "task_id": self._task_id,
            "stop_on_step": self._stop_on_step,
        }

    def close(self) -> None:
        return None


class TestEvaluator(Evaluator):
    __test__ = False

    def __init__(
        self,
        *,
        tasks: list[str] | None = None,
        stop_on_step: bool = False,
        invalid_observation: bool = False,
    ) -> None:
        self._tasks = tasks or ["task-1", "task-2", "task-3"]
        self._stop_on_step = stop_on_step
        self._invalid_observation = invalid_observation

    def list_tasks(self) -> list[str]:
        return list(self._tasks)

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        return {
            "task_id": str(index.task_id),
            "session_id": index.session_id,
            "stop_on_step": self._stop_on_step,
            "invalid_observation": self._invalid_observation,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores: list[float] = []
        for paths in self.get_sessions_paths(sessions):
            if not paths.results.exists():
                continue
            payload = json.loads(paths.results.read_text(encoding="utf-8"))
            try:
                score = float(payload["score"])
            except Exception:
                continue
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResults(
            benchmark_name="test_benchmark",
            total_tasks=len(sessions),
            score=avg,
            metrics={},
        )


class TestBenchmark(Benchmark):
    __test__ = False
    display_name: ClassVar[str] = "Test Benchmark"
    slug_name: ClassVar[str] = "test_benchmark"
    runner: str | None = "direct"  # Run in-process; override via benchmark_kwargs for runner tests
    tasks: list[str] = ["task-1", "task-2", "task-3"]  # noqa: RUF012

    @classmethod
    def get_evaluator_class(cls):
        return TestEvaluator

    @classmethod
    def get_session_class(cls):
        return TestSession

    stop_on_step: bool = False
    invalid_observation: bool = False

    def get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "stop_on_step": self.stop_on_step,
            "invalid_observation": self.invalid_observation,
        }
