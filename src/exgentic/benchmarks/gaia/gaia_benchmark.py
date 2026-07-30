# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import logging
import re
import string
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.actions import ActionsHandler, extract_argument
from ...core.benchmark import Benchmark
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    EmptyObservation,
    FinishAction,
    Observation,
    SessionIndex,
    SessionScore,
    SingleAction,
    SingleObservation,
)
from ...observers.logging import get_logger
from ...utils.paths import get_run_paths
from ...utils.settings import RunnerName

_run_logger: logging.Logger | None = None


def _get_run_logger() -> logging.Logger:
    """Benchmark-level logger that writes into the run's run log."""
    global _run_logger
    if _run_logger is None:
        log_path = get_run_paths().tracker
        _run_logger = get_logger(__name__, str(log_path))
    return _run_logger


# ── Answer normalization ─────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    """Normalize an answer string for exact-match comparison.

    Applies the standard GAIA normalization: lowercase, strip articles,
    remove punctuation, and collapse whitespace.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return True if normalized prediction matches normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ── GAIA dataset helpers ─────────────────────────────────────────────

# Number of validation tasks per subset.
GAIA_TASK_COUNTS: dict[str, int] = {
    "2023_all": 165,
    "2023_level1": 53,
    "2023_level2": 86,
    "2023_level3": 26,
}


def _load_gaia_row(task_id: str, subset: str) -> dict[str, Any]:
    """Load a single GAIA row by index from the validation split."""
    from datasets import load_dataset

    idx = int(task_id)
    row = load_dataset(
        "gaia-benchmark/GAIA",
        subset,
        split=f"validation[{idx}:{idx + 1}]",
        trust_remote_code=True,
    )[0]
    return dict(row)


# ── Actions ──────────────────────────────────────────────────────────


class GAIAFinishArgs(BaseModel):
    answer: str = Field(
        ...,
        description="Final answer to the question. Provide a concise, exact answer.",
    )


class GAIAFinishAction(FinishAction):
    name: Literal["submit"] = "submit"
    arguments: GAIAFinishArgs


# ── Session ──────────────────────────────────────────────────────────


class GAIASession(Session):
    """Session for GAIA benchmark evaluation.

    Each session loads one GAIA question from HuggingFace, presents it
    to the agent, collects the final answer, and scores by exact match.
    """

    _question: str
    _done: bool

    def __init__(
        self,
        task_id: str,
        subset: str = "2023_all",
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id

        row = _load_gaia_row(task_id, subset)
        self._question = row["Question"]
        self._gold_answer = row.get("Final answer", "")
        self._level = row.get("Level", None)
        self._task_id = int(task_id)
        self._done = False
        self._final_answer: str | None = None

        self._registry = ActionsHandler(logger=self.logger)
        self._registry.add_action(
            name="submit",
            description="Submit the final answer and complete the task.",
            action_cls=GAIAFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )
        super().__init__()

    @property
    def task(self) -> str:
        return (
            "Answer the following question. You may need to reason through multiple steps, "
            "search for information, or process data to arrive at the answer.\n"
            "When you are finished, submit your final answer by calling `submit`.\n"
            "Provide a concise, exact answer — do not include explanations or units unless "
            "they are part of the answer itself.\n"
            "\n"
            f"Question:\n\n{self._question}"
        )

    @property
    def context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        if self._level is not None:
            ctx["level"] = self._level
        return ctx

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    @property
    def task_id(self) -> str:
        return str(self._task_id)

    def _to_observation(self, raw: Any, invoking_actions: list[SingleAction] | None = None) -> Observation:
        return SingleObservation(invoking_actions=invoking_actions or [], result=raw)

    def start(self) -> Observation | None:
        return EmptyObservation()

    def step(self, action: Action) -> Observation | None:
        if action is None:
            self._done = True

        if self._done:
            return None

        observation = self._registry.execute(action)
        return observation

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        if self._final_answer is None:
            return SessionScore(score=0.0, success=False, is_finished=False)

        match = exact_match(self._final_answer, self._gold_answer)
        score = 1.0 if match else 0.0
        self.logger.info(
            f"Gold: {self._gold_answer!r}  Prediction: {self._final_answer!r}  " f"Level: {self._level}  Score: {score}"
        )
        return SessionScore(score=score, success=match, is_finished=True)

    def close(self):
        super().close()
        sc = self.score()
        payload: dict[str, Any] = {
            "score": sc.score,
            "success": bool(sc.success),
        }
        if self._level is not None:
            payload["level"] = self._level
        self.save_results(payload)

    # ── Action handlers ──────────────────────────────────────────────

    def _handle_finish(self, action: SingleAction) -> None:
        self.logger.info(f"Received final answer: {action}")
        answer = extract_argument(action.arguments, "answer", None)
        self._final_answer = answer
        self._done = True
        return


# ── Evaluator ────────────────────────────────────────────────────────


class GAIAEvaluator(Evaluator):
    """Evaluator for GAIA — task discovery and aggregation."""

    def __init__(self, subset: str = "2023_all") -> None:
        self._subset = subset

    def list_tasks(self) -> list[str]:
        total = GAIA_TASK_COUNTS.get(self._subset)
        if total is not None:
            return [str(i) for i in range(total)]
        # Fallback: load the dataset to determine the count.
        from datasets import load_dataset

        ds = load_dataset(
            "gaia-benchmark/GAIA",
            self._subset,
            split="validation",
            trust_remote_code=True,
        )
        return [str(i) for i in range(len(ds))]

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        run_logger = _get_run_logger()
        scores: list[float] = []
        level_scores: dict[str, list[float]] = {}

        for paths in self.get_sessions_paths(sessions):
            fp = paths.benchmark_results
            try:
                with open(fp, encoding="utf-8-sig") as f:
                    payload = json.load(f)
                s = float(payload["score"])
                scores.append(s)
            except FileNotFoundError as err:
                raise FileNotFoundError(
                    f"Missing benchmark result for session" f" '{paths.session_id}' at {fp}"
                ) from err
            except Exception:
                run_logger.exception(
                    "Failed to load benchmark result for session %s at %s",
                    paths.session_id,
                    fp,
                )
                raise

            # Collect per-level scores if available.
            level = payload.get("level")
            if level is not None:
                level_key = f"level_{level}"
                level_scores.setdefault(level_key, []).append(s)

        avg = sum(scores) / len(scores) if scores else 0.0

        # Build per-level accuracy metrics.
        metrics: dict[str, Any] = {}
        for level_key, lvl_scores in sorted(level_scores.items()):
            metrics[f"{level_key}_accuracy"] = sum(lvl_scores) / len(lvl_scores) if lvl_scores else 0.0
            metrics[f"{level_key}_total"] = len(lvl_scores)

        return BenchmarkResults(
            benchmark_name="gaia",
            total_tasks=len(sessions),
            score=avg,
            metrics=metrics,
        )


# ── Benchmark config ─────────────────────────────────────────────────


class GAIABenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "GAIA"
    slug_name: ClassVar[str] = "gaia"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _get_evaluator_class(cls):
        return GAIAEvaluator

    @classmethod
    def _get_session_class(cls):
        return GAIASession

    subset: Literal["2023_all", "2023_level1", "2023_level2", "2023_level3"] = "2023_all"
    runner: RunnerName | None = None

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {"subset": self.subset}

    def _get_session_kwargs(self) -> dict[str, Any]:
        return {"subset": self.subset}
