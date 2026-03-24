# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import ast
import json
import logging
import operator as op
import re
from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

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
from ...utils.settings import ExgenticSettings, RunnerName, get_settings

GSM8K_TOTAL_TASKS = 1319

_run_logger: logging.Logger | None = None


def _get_run_logger() -> logging.Logger:
    """Benchmark-level logger that writes into the run's run log."""
    global _run_logger
    if _run_logger is None:
        log_path = get_run_paths().tracker
        _run_logger = get_logger(__name__, str(log_path))
    return _run_logger


def _parse_int(s: str | None) -> int | None:
    if s is None:
        return None
    s = str(s).strip()
    # allow "42\n" etc.
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    return None


_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_ALLOWED_DESC = "numbers (ints/decimals), + - * /, parentheses, unary +/-. No names, functions, **, %, comparisons."


def safe_evaluate(expression: str):
    expr = (expression or "").strip()
    if not expr:
        return f"Invalid expression: empty. Allowed: {_ALLOWED_DESC} Got: {expression!r}"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return f"Invalid syntax. Allowed: {_ALLOWED_DESC} Got: {expression!r}"

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            if isinstance(n.op, ast.Div) and ev(n.right) == 0:
                return f"Invalid op: division by zero. Got: {expression!r}"
            return _ALLOWED_BINOPS[type(n.op)](ev(n.left), ev(n.right))
        return f"Invalid element: {type(n).__name__}. Allowed: {_ALLOWED_DESC} Got: {expression!r}"

    try:
        out = ev(tree)
        return (
            out
            if isinstance(out, (int, float, str))
            else f"Did not evaluate to a number. Allowed: {_ALLOWED_DESC} Got: {expression!r}"
        )
    except Exception as e:
        return f"Error evaluating. Allowed: {_ALLOWED_DESC} Got: {expression!r}. Error: {type(e).__name__}: {e}"


class GSM8kCalculateExpressionArgs(BaseModel):
    expression: str = Field(..., description="Arithmetic expression using + - * / and parentheses.")


class GSM8kCalculateExpressionAction(SingleAction):
    name: Literal["calculate_expression"] = "calculate_expression"
    arguments: GSM8kCalculateExpressionArgs


class GSM8kFinishArgs(BaseModel):
    answer: str | int = Field(..., description="Final answer as a single integer (string or int).")

    @field_validator("answer", mode="before")
    @classmethod
    def coerce_int_to_str(cls, v: Any) -> str:
        # Allow agents to pass raw integers; store as string for downstream checks.
        if isinstance(v, int):
            return str(v)
        return v

    @field_validator("answer")
    @classmethod
    def must_look_like_int(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"[+-]?\d+", v):
            raise ValueError("Answer must be a single integer.")
        return v


class GSM8kFinishAction(FinishAction):
    name: Literal["submit"] = "submit"
    arguments: GSM8kFinishArgs


class GSM8kSession(Session):
    """Session for GSM8k benchmark evaluation."""

    _question: str
    _done: bool

    def __init__(
        self,
        settings: ExgenticSettings,
        include_calculator_tool: bool,
        instance: dict[str, Any],
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id
        self._question = instance["question"]
        self._answer = instance["answer"]
        self._task_id = instance["task_id"]
        self._done = False
        self._gold_answer = self._answer.split("####")[-1].strip()
        self._final_answer = None
        self._registry = ActionsHandler(logger=self.logger)
        # Define Actions directly (single source of truth)

        if include_calculator_tool:
            self._registry.add_action(
                name="calculate_expression",
                description=(
                    "Evaluate a mathematical expression using only"
                    " numbers and basic operators"
                    " (+, -, *, /, parentheses)."
                ),
                action_cls=GSM8kCalculateExpressionAction,
                handler=self._handle_calculate_expression,
            )

        self._registry.add_action(
            name="submit",
            description="Submit final answer and complete the task.",
            action_cls=GSM8kFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )
        super().__init__()

    @property
    def task(self) -> str:
        return (
            "Solve the following math word problem using basic arithmetic.\n"
            "You may perform intermediate calculations if helpful.\n"
            "When you are finished, submit the final answer as a single integer by calling `submit`.\n"
            "\n"
            "Do not include units, words, or explanations in the final answer.\n"
            "Your response will be graded only on whether the final integer exactly matches the correct answer.\n"
            "\n"
            f"Question:\n\n{self._question}"
        )

    @property
    def context(self) -> dict[str, Any]:
        return {}

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    @property
    def task_id(self) -> str:
        return str(self._task_id)

    def _to_observation(self, raw: Any, invoking_actions: list[SingleAction] | None = None) -> Observation:
        return SingleObservation(invoking_actions=invoking_actions or [], result=raw)

    def start(self) -> Observation | None:
        # Empty initial observation; question is carried in the task string.
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
        gold = _parse_int(self._gold_answer)
        pred = _parse_int(self._final_answer)
        score = 1.0 if (gold is not None and pred is not None and gold == pred) else 0.0
        self.logger.info(f"Gold: {self._gold_answer} Prediction: {self._final_answer} Score: {score}")
        # Finished only when the benchmark finish action stores a final answer.
        finished = self._final_answer is not None
        success = score == 1.0
        return SessionScore(score=float(score), success=success, is_finished=finished)

    def close(self):
        super().close()
        # Persist minimal results for aggregation
        sc = self.score()
        self.save_standard_results(sc)

    # Action handlers ------------------------------------------------------------
    def _handle_calculate_expression(self, action: SingleAction) -> Any:
        self.logger.info(f"Received expression: {action}")
        expression = extract_argument(action.arguments, "expression", "")
        result = safe_evaluate(expression)
        self.logger.info(f"Calculated result: {result}")
        return result

    def _handle_finish(self, action: SingleAction) -> None:
        self.logger.info(f"Received final answer: {action}")
        answer = extract_argument(action.arguments, "answer", None)
        self._final_answer = answer
        self._done = True
        return


# ── Evaluator ────────────────────────────────────────────────────────


class GSM8kEvaluator(Evaluator):
    """Evaluator for GSM8k — task discovery, session kwargs, aggregation."""

    def __init__(self, subset: str = "main", include_calculator_tool: bool = True) -> None:
        self._subset = subset
        self._include_calculator_tool = include_calculator_tool
        self._dataset = None

    def _ensure_dataset(self) -> None:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset("gsm8k", "main")["test"]

    def list_tasks(self) -> list[str]:
        return [str(i) for i in range(GSM8K_TOTAL_TASKS)]

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        self._ensure_dataset()
        idx = int(index.task_id)
        if idx < 0 or idx >= len(self._dataset):
            raise IndexError(f"Task id {index.task_id} out of range for GSM8k.")
        instance = {"task_id": idx, **self._dataset[idx]}
        return {
            "settings": get_settings(),
            "include_calculator_tool": self._include_calculator_tool,
            "instance": instance,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        run_logger = _get_run_logger()
        scores: list[float] = []
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
        avg = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResults(
            benchmark_name="gsm8k",
            total_tasks=len(sessions),
            score=avg,
            metrics={},
        )


# ── Benchmark config ─────────────────────────────────────────────────


class GSM8kBenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "GSM8k"
    slug_name: ClassVar[str] = "gsm8k"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_evaluator_class(cls):
        return GSM8kEvaluator

    @classmethod
    def get_session_class(cls):
        return GSM8kSession

    subset: Literal["main"] = "main"
    include_calculator_tool: bool = True
    runner: RunnerName | None = "direct"  # Code is threadsafe

    def get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "include_calculator_tool": self.include_calculator_tool,
        }
