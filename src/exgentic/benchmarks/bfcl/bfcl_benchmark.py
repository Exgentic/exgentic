# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import logging
import re
import string
import urllib.request
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...adapters.executors.executer import make_executer
from ...core.actions import ActionsHandler
from ...core.benchmark import Benchmark
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
)
from ...observers.logging import get_logger
from ...utils.paths import get_run_paths
from ...utils.settings import ExecuterName, ExgenticSettings, get_settings

_run_logger: logging.Logger | None = None

BFCLSubset = Literal[
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]

_BFCL_BASE_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

_SUBSET_TASK_COUNTS: dict[str, int] = {
    "simple_python": 399,
    "simple_java": 99,
    "simple_javascript": 49,
    "multiple": 199,
    "parallel": 199,
    "parallel_multiple": 199,
    "irrelevance": 239,
}


def _get_run_logger() -> logging.Logger:
    global _run_logger
    if _run_logger is None:
        log_path = get_run_paths().tracker
        _run_logger = get_logger(__name__, str(log_path))
    return _run_logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "exgentic" / "bfcl"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _load_bfcl_data(subset: str) -> list[dict[str, Any]]:
    filename = f"BFCL_v4_{subset}.json"
    url = f"{_BFCL_BASE_URL}/{filename}"
    dest = _cache_dir() / filename
    _download_if_missing(url, dest)
    return _load_jsonl(dest)


def _load_bfcl_answers(subset: str) -> dict[str, list[dict[str, Any]]]:
    filename = f"BFCL_v4_{subset}.json"
    url = f"{_BFCL_BASE_URL}/possible_answer/{filename}"
    dest = _cache_dir() / "possible_answer" / filename
    _download_if_missing(url, dest)
    rows = _load_jsonl(dest)
    return {row["id"]: row["ground_truth"] for row in rows}


# ---------------------------------------------------------------------------
# Schema conversion: BFCL function → Pydantic model + ActionType
# ---------------------------------------------------------------------------

def _normalize_bfcl_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Convert BFCL parameter schema to standard JSON Schema."""
    schema: dict[str, Any] = {}
    param_type = params.get("type", "object")
    if param_type == "dict":
        param_type = "object"
    schema["type"] = param_type

    if "properties" in params:
        props: dict[str, Any] = {}
        for pname, pdef in params["properties"].items():
            p = dict(pdef)
            if p.get("type") == "dict":
                p["type"] = "object"
            if p.get("type") == "float":
                p["type"] = "number"
            if p.get("type") == "list":
                p["type"] = "array"
            if p.get("type") == "tuple":
                p["type"] = "array"
            # Remove non-standard keys
            p.pop("optional", None)
            props[pname] = p
        schema["properties"] = props

    if "required" in params:
        schema["required"] = params["required"]

    return schema


def _sanitize_tool_name(name: str) -> str:
    """Sanitize function name to match OpenAI tool name pattern ^[a-zA-Z0-9_-]+$."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _make_action_type_from_bfcl_func(func_def: dict[str, Any]) -> tuple[ActionType, str]:
    """Create an ActionType from a BFCL function definition.

    Returns (action_type, original_name) so the caller can map sanitized names
    back to original BFCL names for scoring.
    """
    from ...adapters.schemas.json_schema import make_args_model_from_json_schema

    original_name = func_def["name"]
    safe_name = _sanitize_tool_name(original_name)
    description = func_def.get("description", "")
    params = func_def.get("parameters", {"type": "object", "properties": {}})

    normalized = _normalize_bfcl_schema(params)

    try:
        args_model = make_args_model_from_json_schema(safe_name, normalized)
    except Exception:
        # Fallback: create a model that accepts any kwargs
        args_model = type(
            f"{safe_name}_Args",
            (BaseModel,),
            {"model_config": ConfigDict(extra="allow")},
        )

    # Build SingleAction subclass using the sanitized name
    action_cls = type(
        f"{safe_name}_Action",
        (SingleAction,),
        {
            "name": safe_name,
            "__annotations__": {
                "name": Literal[safe_name],  # type: ignore[valid-type]
                "arguments": args_model,
            },
        },
    )

    action_type = ActionType(
        name=safe_name,
        description=description,
        cls=action_cls,
    )
    return action_type, original_name


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _standardize_string(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _coerce_numeric(val: Any) -> float | None:
    """Try to coerce a value to float for numeric comparison."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _value_matches(predicted: Any, acceptable: list[Any]) -> bool:
    """Check if a predicted value matches any of the acceptable values."""
    for acc in acceptable:
        if acc == "" or acc is None:
            continue
        # Try numeric comparison (handles string↔number coercion)
        pred_num = _coerce_numeric(predicted)
        acc_num = _coerce_numeric(acc)
        if pred_num is not None and acc_num is not None:
            if abs(pred_num - acc_num) < 1e-6:
                return True
            continue
        if isinstance(acc, str) and isinstance(predicted, str):
            if _standardize_string(predicted) == _standardize_string(acc):
                return True
        elif isinstance(acc, list) and isinstance(predicted, list):
            if len(acc) == len(predicted):
                if all(_value_matches(p, [a]) for p, a in zip(predicted, acc)):
                    return True
        elif isinstance(acc, dict) and isinstance(predicted, dict):
            if acc == predicted:
                return True
        elif isinstance(acc, bool) and isinstance(predicted, bool):
            if acc == predicted:
                return True
        elif predicted == acc:
            return True
    return False


def _check_single_call(
    predicted_name: str,
    predicted_args: dict[str, Any],
    ground_truth: dict[str, dict[str, list[Any]]],
) -> bool:
    """Check if a single predicted function call matches the ground truth entry.

    ground_truth is {func_name: {param_name: [acceptable_values], ...}}
    """
    if predicted_name not in ground_truth:
        return False

    expected_params = ground_truth[predicted_name]

    for param_name, acceptable_values in expected_params.items():
        # Check if all acceptable values are empty/optional markers
        all_optional = all(v == "" or v is None for v in acceptable_values)
        if all_optional:
            continue

        # An empty string "" in the acceptable list means the parameter is optional
        has_optional_marker = any(v == "" or v is None for v in acceptable_values)

        # Treat missing or None-valued parameters the same way
        param_missing = param_name not in predicted_args or predicted_args[param_name] is None
        if param_missing:
            if has_optional_marker:
                continue
            return False

        if not _value_matches(predicted_args[param_name], acceptable_values):
            return False

    return True


def _score_function_calls(
    recorded_calls: list[dict[str, Any]],
    ground_truth: list[dict[str, dict[str, list[Any]]]],
    is_irrelevance: bool = False,
) -> float:
    """Score recorded function calls against BFCL ground truth.

    Returns 1.0 for a correct answer, 0.0 otherwise.
    """
    if is_irrelevance:
        # For irrelevance, the agent should NOT call any function
        return 1.0 if len(recorded_calls) == 0 else 0.0

    if len(recorded_calls) != len(ground_truth):
        return 0.0

    # Try to match each ground truth entry to a recorded call
    used = [False] * len(recorded_calls)
    for gt_entry in ground_truth:
        matched = False
        for i, call in enumerate(recorded_calls):
            if used[i]:
                continue
            if _check_single_call(call["name"], call["arguments"], gt_entry):
                used[i] = True
                matched = True
                break
        if not matched:
            return 0.0

    return 1.0


# ---------------------------------------------------------------------------
# Finish action
# ---------------------------------------------------------------------------

class BFCLFinishArgs(BaseModel):
    pass


class BFCLFinishAction(FinishAction):
    name: Literal["submit"] = "submit"
    arguments: BFCLFinishArgs = Field(default_factory=BFCLFinishArgs)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class BFCLSession(Session):
    """BFCL session: present function schemas, record calls, score against ground truth."""

    _done: bool
    _recorded_calls: list[dict[str, Any]]

    def __init__(
        self,
        settings: ExgenticSettings,
        instance: dict[str, Any],
        ground_truth: list[dict[str, Any]],
        is_irrelevance: bool,
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id
        self._instance = instance
        self._ground_truth = ground_truth
        self._is_irrelevance = is_irrelevance
        self._task_id_str = instance["id"]
        self._done = False
        self._recorded_calls = []

        # Extract question text from BFCL format: [[{"role": "user", "content": "..."}]]
        question_turns = instance["question"]
        messages = question_turns[0] if question_turns else []
        self._question = " ".join(
            msg["content"] for msg in messages if msg.get("role") == "user"
        )

        self._registry = ActionsHandler(logger=self.logger)
        # Map sanitized action names back to original BFCL names for scoring
        self._name_map: dict[str, str] = {}

        # Register each BFCL function as an action
        for func_def in instance.get("function", []):
            try:
                action_type, original_name = _make_action_type_from_bfcl_func(func_def)
                self._name_map[action_type.name] = original_name
                self._registry.add_action_type(action_type, self._handle_function_call)
            except Exception:
                self.logger.warning(f"Failed to create action for function: {func_def.get('name', '?')}")

        # Register submit/finish action
        self._registry.add_action(
            name="submit",
            description=(
                "Submit your answer. Call this after making all necessary function calls, "
                "or immediately if none of the available functions are relevant to the question."
            ),
            action_cls=BFCLFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )
        super().__init__()

    @property
    def task(self) -> str:
        task_text = (
            "You are given a question and a set of available functions.\n"
            "Your task is to call the correct function(s) with the right arguments to answer the question.\n"
        )
        if self._is_irrelevance:
            task_text += (
                "If none of the provided functions are relevant to the question, "
                "call 'submit' immediately without calling any function.\n"
            )
        else:
            task_text += (
                "Call the relevant function(s) with the correct arguments, "
                "then call 'submit' to complete the task.\n"
            )
        task_text += f"\nQuestion: {self._question}"
        return task_text

    @property
    def context(self) -> dict[str, Any]:
        return {}

    @property
    def task_id(self) -> str:
        return self._task_id_str

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    def start(self) -> Optional[Observation]:
        return EmptyObservation()

    def step(self, action: Action) -> Optional[Observation]:
        if action is None:
            self._done = True

        if self._done:
            return None

        observation = self._registry.execute(action)
        return observation

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        score = _score_function_calls(
            self._recorded_calls,
            self._ground_truth,
            is_irrelevance=self._is_irrelevance,
        )
        self.logger.info(
            f"Task: {self._task_id_str} "
            f"Calls: {self._recorded_calls} "
            f"Ground truth: {self._ground_truth} "
            f"Score: {score}"
        )
        finished = self._done
        success = score >= 1.0 - 1e-6
        return SessionScore(score=score, success=success, is_finished=finished)

    def close(self):
        super().close()
        sc = self.score()
        self.save_standard_results(sc)

    # -- Action handlers -------------------------------------------------------

    def _handle_function_call(self, action: SingleAction) -> Any:
        sanitized_name = action.name
        # Map back to original BFCL function name for scoring
        original_name = self._name_map.get(sanitized_name, sanitized_name)
        if isinstance(action.arguments, BaseModel):
            args = action.arguments.model_dump()
        elif isinstance(action.arguments, dict):
            args = action.arguments
        else:
            args = {}
        self.logger.info(f"Function call recorded: {original_name}({args})")
        self._recorded_calls.append({"name": original_name, "arguments": args})
        return f"Function '{original_name}' called successfully."

    def _handle_finish(self, action: SingleAction) -> None:
        self.logger.info(f"Submit called. Recorded {len(self._recorded_calls)} function call(s).")
        self._done = True


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class BFCLBenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "BFCL"
    slug_name: ClassVar[str] = "bfcl"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    subset: BFCLSubset = "simple_python"
    executer: Optional[ExecuterName] = "inprocess"

    # Internals
    _data: list[dict[str, Any]] | None = None
    _answers: dict[str, list[dict[str, Any]]] | None = None

    def _ensure_data(self) -> None:
        if self._data is None:
            self._data = _load_bfcl_data(self.subset)
            self._answers = _load_bfcl_answers(self.subset)

    def list_tasks(self) -> list[str]:
        return [str(i) for i in range(_SUBSET_TASK_COUNTS.get(self.subset, 0))]

    def create_session(self, index: SessionIndex) -> BFCLSession:
        self._ensure_data()
        assert self._data is not None
        assert self._answers is not None

        idx = int(index.task_id)
        if idx < 0 or idx >= len(self._data):
            raise IndexError(f"Task id {index.task_id} out of range for BFCL {self.subset}.")

        instance = self._data[idx]
        task_id = instance["id"]
        ground_truth = self._answers.get(task_id, [])
        is_irrelevance = self.subset == "irrelevance"

        executer = make_executer(
            self.executer,
            BFCLSession,
            get_settings(),
            instance,
            ground_truth,
            is_irrelevance,
            session_id=index.session_id,
        )
        proxy = executer.get_proxy()
        return proxy  # type: ignore[return-value]

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
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing benchmark result for session '{paths.session_id}' at {fp}"
                ) from None
            except Exception:
                run_logger.exception(
                    "Failed to load benchmark result for session %s at %s",
                    paths.session_id,
                    fp,
                )
                raise
        avg = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResults(
            benchmark_name="bfcl",
            total_tasks=len(sessions),
            score=avg,
            metrics={"subset": self.subset},
        )
