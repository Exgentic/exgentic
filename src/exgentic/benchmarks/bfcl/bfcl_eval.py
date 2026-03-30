# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""BFCL evaluator and session classes.

These classes import bfcl_eval (via bfcl_shim) at runtime.  They are only
ever instantiated inside the isolated runner subprocess, so the heavy
``bfcl_eval`` dependency is never required in the host process.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ...adapters.schemas.openai import openai_tools_to_action_types
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    EmptyObservation,
    MultiObservation,
    SessionIndex,
    SessionScore,
    SingleAction,
    SingleObservation,
)
from .bfcl_benchmark import BFCLFinishAction
from .bfcl_shim import load_bfcl_symbols

BFCLSubset = Literal[
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]


def _is_relevance_subset(subset: str) -> bool:
    return subset in {"irrelevance", "live_irrelevance", "live_relevance"}


def _is_multi_turn_subset(subset: str) -> bool:
    return subset.startswith("multi_turn_")


def _language_for_subset(subset: str, symbols: dict[str, Any]) -> Any:
    language = symbols["Language"]
    if subset == "simple_java":
        return language.JAVA
    if subset == "simple_javascript":
        return language.JAVASCRIPT
    return language.PYTHON


def _merge_observations(
    *items: SingleObservation | MultiObservation | None,
) -> SingleObservation | MultiObservation | None:
    observations: list[SingleObservation] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, MultiObservation):
            observations.extend(item.observations)
            continue
        observations.append(item)

    if not observations:
        return None
    if len(observations) == 1:
        return observations[0]
    return MultiObservation(observations=observations)


def _action_arguments_dict(action: SingleAction) -> dict[str, Any]:
    arguments = action.arguments
    if isinstance(arguments, BaseModel):
        return arguments.model_dump()
    if isinstance(arguments, dict):
        return dict(arguments)
    return {"value": arguments}


def _render_turn_text(turn_messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in turn_messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        role = str(message.get("role", ""))
        if role == "system":
            parts.append(f"System: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


class BFCLSession(Session):
    """Exgentic-native BFCL session with Gorilla-backed scoring."""

    def __init__(
        self,
        subset: BFCLSubset,
        prompt_entry: dict[str, Any],
        possible_answer_entry: dict[str, Any] | None,
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id

        self.subset = subset
        self.prompt_entry = prompt_entry
        self.possible_answer_entry = possible_answer_entry
        self._task_id = str(prompt_entry["id"])
        self._conversation_turns = [list(turn) for turn in prompt_entry.get("question", [])]
        self._turn_texts = [_render_turn_text(turn) for turn in self._conversation_turns]
        self._task_text = self._turn_texts[0] if self._turn_texts else ""
        self._action_types = self._build_action_types(prompt_entry)

        self._current_turn_index = 0
        self._completed = False
        self._result_payload: dict[str, Any] | None = None

        turn_count = max(1, len(self._conversation_turns))
        self._turn_step_calls: list[list[list[str]]] = [[] for _ in range(turn_count)]
        self._turn_step_actions: list[list[list[dict[str, Any]]]] = [[] for _ in range(turn_count)]

        super().__init__()

    @property
    def task(self) -> str:
        return self._task_text

    @property
    def context(self) -> dict[str, Any]:
        return {"policy": self._build_context_text()}

    @property
    def actions(self) -> list[ActionType]:
        return self._action_types

    @property
    def task_id(self) -> str:
        return self._task_id

    def start(self) -> EmptyObservation:
        return EmptyObservation()

    def step(self, action: Action) -> SingleObservation | MultiObservation | None:
        if self._completed:
            return None
        if action is None:
            raise ValueError("BFCL requires an action or the dedicated finish action.")

        flat_actions = action.to_action_list()
        finish_requested = False
        tool_actions: list[SingleAction] = []
        for item in flat_actions:
            if item.name == "finish":
                finish_requested = True
                continue
            tool_actions.append(item)

        step_call_strings = [self._action_to_function_call_string(item) for item in tool_actions]
        if step_call_strings:
            self._turn_step_calls[self._current_turn_index].append(step_call_strings)
            self._turn_step_actions[self._current_turn_index].append(
                [self._serialize_action(item) for item in tool_actions]
            )

        outputs = self._build_step_output_observation(tool_actions, step_call_strings)
        if finish_requested:
            return self._finish_turn(outputs)

        return outputs or EmptyObservation()

    def done(self) -> bool:
        return self._completed

    def score(self) -> SessionScore:
        if self._result_payload is None:
            self._result_payload = self._compute_result_payload()

        return SessionScore(
            score=float(self._result_payload["score"]),
            success=bool(self._result_payload["success"]),
            is_finished=self._result_payload.get("is_finished"),
            session_metrics=self._result_payload.get("session_metrics", {}),
            session_metadata=self._result_payload.get("session_metadata", {}),
        )

    def close(self) -> None:
        if self._result_payload is None:
            self._result_payload = self._compute_result_payload()
        _write_json(self.paths.benchmark_results, self._result_payload)

    def _build_action_types(self, prompt_entry: dict[str, Any]) -> list[ActionType]:
        symbols = load_bfcl_symbols()
        functions = self._collect_functions(prompt_entry)
        openai_tools = symbols["convert_to_tool"](
            functions,
            symbols["GORILLA_TO_OPENAPI"],
            symbols["ModelStyle"].OPENAI_COMPLETIONS,
        )
        action_types = openai_tools_to_action_types(openai_tools)
        action_types.append(
            ActionType(
                name="finish",
                description="End the current BFCL step.",
                cls=BFCLFinishAction,
                is_finish=True,
            )
        )
        return action_types

    def _collect_functions(self, prompt_entry: dict[str, Any]) -> list[dict[str, Any]]:
        functions = [deepcopy(item) for item in prompt_entry.get("function", [])]
        for items in prompt_entry.get("missed_function", {}).values():
            for item in items:
                functions.append(deepcopy(item))

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in functions:
            name = str(item.get("name", ""))
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(item)
        return deduped

    def _build_context_text(self) -> str:
        if len(self._conversation_turns) <= 1:
            return (
                "Complete the task using one or more actions, then call the dedicated "
                "finish action. Clarification questions or any type of interaction with "
                "the user is not permitted. In this task, actions are recorded rather "
                "than executed through a live environment. Calling finish ends the "
                "execution."
            )
        if _is_multi_turn_subset(self.subset):
            return (
                "Complete the current step using one or more actions, then call the "
                "dedicated finish action. Clarification questions or any type of "
                "interaction with the user is not permitted."
            )
        return (
            "Complete the current step using one or more actions, then call the "
            "dedicated finish action. Clarification questions or any type of "
            "interaction with the user is not permitted. In this task, actions are "
            "recorded rather than executed through a live environment."
        )

    def _build_turn_observation(self, turn_index: int) -> SingleObservation:
        if turn_index >= len(self._turn_texts):
            return EmptyObservation()
        text = self._turn_texts[turn_index]
        if not text:
            return EmptyObservation()
        return SingleObservation(
            result=text,
            invoking_actions=[],
        )

    def _finish_turn(
        self, outputs: SingleObservation | MultiObservation | None
    ) -> SingleObservation | MultiObservation | None:
        if self._current_turn_index >= len(self._conversation_turns) - 1:
            self._completed = True
            return outputs

        self._current_turn_index += 1
        next_turn = self._build_turn_observation(self._current_turn_index)
        return _merge_observations(outputs, next_turn) or EmptyObservation()

    def _build_step_output_observation(
        self,
        tool_actions: list[SingleAction],
        step_call_strings: list[str],
    ) -> SingleObservation | MultiObservation | None:
        if not tool_actions:
            return None

        if _is_multi_turn_subset(self.subset):
            raw_results = self._execute_multi_turn_step(step_call_strings)
        else:
            raw_results = ["Action recorded." for _ in tool_actions]

        observations = [
            SingleObservation(result=result, invoking_actions=[action])
            for action, result in zip(tool_actions, raw_results, strict=False)
        ]
        return _merge_observations(*observations)

    def _execute_multi_turn_step(self, step_call_strings: list[str]) -> list[str]:
        symbols = load_bfcl_symbols()
        execution_results, _ = symbols["execute_multi_turn_func_call"](
            func_call_list=step_call_strings,
            initial_config=self.prompt_entry["initial_config"],
            involved_classes=self.prompt_entry["involved_classes"],
            model_name=f"{symbols['proxy_model_name']}_{self.session_id}_runtime",
            test_entry_id=self._task_id,
            long_context=("long_context" in self.subset),
            is_evaL_run=False,
        )
        return execution_results

    def _action_to_function_call_string(self, action: SingleAction) -> str:
        arguments = _action_arguments_dict(action)
        if not arguments:
            return f"{action.name}()"
        rendered = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        return f"{action.name}({rendered})"

    def _serialize_action(self, action: SingleAction) -> dict[str, Any]:
        return {
            "id": action.id,
            "name": action.name,
            "arguments": _action_arguments_dict(action),
        }

    def _flatten_semantic_actions(self) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for turn in self._turn_step_actions:
            for step in turn:
                for action in step:
                    flattened.append({action["name"]: action["arguments"]})
        return flattened

    def _compute_result_payload(self) -> dict[str, Any]:
        trace_payload = {
            "task_id": self._task_id,
            "subset": self.subset,
            "turn_step_calls": self._turn_step_calls,
            "turn_step_actions": self._turn_step_actions,
        }
        trace_path = self.paths.benchmark_dir / "bfcl_trace.json"
        _write_json(trace_path, trace_payload)

        if not self._completed:
            payload = {
                "score": 0.0,
                "success": False,
                "is_finished": False,
                "session_metrics": {
                    "completed_turns": self._current_turn_index,
                },
                "session_metadata": {
                    "bfcl_task_id": self._task_id,
                    "trace_file": str(trace_path),
                },
            }
            _write_json(self.paths.benchmark_dir / "bfcl_score.json", payload)
            return payload

        try:
            if _is_multi_turn_subset(self.subset):
                payload = self._score_multi_turn(trace_path)
            elif _is_relevance_subset(self.subset):
                payload = self._score_relevance(trace_path)
            else:
                payload = self._score_ast(trace_path)
        except Exception as exc:
            payload = {
                "score": 0.0,
                "success": False,
                "is_finished": False,
                "session_metadata": {
                    "bfcl_task_id": self._task_id,
                    "trace_file": str(trace_path),
                    "error": str(exc),
                    "error_source": "benchmark",
                },
            }

        _write_json(self.paths.benchmark_dir / "bfcl_score.json", payload)
        return payload

    def _score_relevance(self, trace_path: Path) -> dict[str, Any]:
        tool_call_count = len(self._flatten_semantic_actions())
        success = tool_call_count == 0 if "irrelevance" in self.subset else tool_call_count > 0
        return {
            "score": 1.0 if success else 0.0,
            "success": success,
            "is_finished": True,
            "session_metrics": {
                "tool_call_count": tool_call_count,
                "accuracy": 1.0 if success else 0.0,
            },
            "session_metadata": {
                "bfcl_task_id": self._task_id,
                "trace_file": str(trace_path),
            },
        }

    def _score_ast(self, trace_path: Path) -> dict[str, Any]:
        if self.possible_answer_entry is None:
            raise ValueError(f"Missing ground truth for subset '{self.subset}'.")

        symbols = load_bfcl_symbols()
        checker_result = symbols["ast_checker"](
            self.prompt_entry["function"],
            self._flatten_semantic_actions(),
            self.possible_answer_entry["ground_truth"],
            _language_for_subset(self.subset, symbols),
            self.subset,
            symbols["proxy_model_name"],
        )
        score = 1.0 if checker_result["valid"] else 0.0
        return {
            "score": score,
            "success": bool(checker_result["valid"]),
            "is_finished": True,
            "session_metrics": {
                "accuracy": score,
                "action_count": len(self._flatten_semantic_actions()),
            },
            "session_metadata": {
                "bfcl_task_id": self._task_id,
                "trace_file": str(trace_path),
                "checker_result": checker_result,
            },
        }

    def _score_multi_turn(self, trace_path: Path) -> dict[str, Any]:
        if self.possible_answer_entry is None:
            raise ValueError(f"Missing ground truth for subset '{self.subset}'.")

        model_turns = self._turn_step_calls
        ground_truth_turns = self.possible_answer_entry["ground_truth"]
        if len(model_turns) != len(ground_truth_turns):
            checker_result = {
                "valid": False,
                "error_message": (
                    "Model was force-terminated before completing all turns. "
                    f"Observed {len(model_turns)} turns for {len(ground_truth_turns)} ground-truth turns."
                ),
                "error_type": "multi_turn:force_terminated",
            }
        else:
            symbols = load_bfcl_symbols()
            checker_result = symbols["multi_turn_checker"](
                model_turns,
                ground_truth_turns,
                self.prompt_entry,
                self.subset,
                f"{symbols['proxy_model_name']}_{self.session_id}_score",
            )

        score = 1.0 if checker_result["valid"] else 0.0
        return {
            "score": score,
            "success": bool(checker_result["valid"]),
            "is_finished": True,
            "session_metrics": {
                "accuracy": score,
                "turn_count": len(model_turns),
            },
            "session_metadata": {
                "bfcl_task_id": self._task_id,
                "trace_file": str(trace_path),
                "checker_result": checker_result,
            },
        }


# ── Evaluator ────────────────────────────────────────────────────────


class BFCLEvaluator(Evaluator):
    """Evaluator for BFCL — task discovery, session kwargs, aggregation."""

    def __init__(self, subset: str = "simple_python") -> None:
        self._subset: BFCLSubset = subset  # type: ignore[assignment]
        self._entries: list[dict[str, Any]] | None = None
        self._answers_by_id: dict[str, dict[str, Any]] | None = None

    def _ensure_loaded(self) -> None:
        if self._entries is not None and self._answers_by_id is not None:
            return
        symbols = load_bfcl_symbols()
        entries = symbols["load_dataset_entry"](self._subset)
        answers = (
            {}
            if _is_relevance_subset(self._subset)
            else {str(entry["id"]): entry for entry in symbols["load_ground_truth_entry"](self._subset)}
        )
        self._entries = entries
        self._answers_by_id = answers

    def list_tasks(self) -> list[str]:
        self._ensure_loaded()
        assert self._entries is not None
        return [str(entry["id"]) for entry in self._entries]

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._entries is not None
        prompt_entry = next(
            (entry for entry in self._entries if str(entry["id"]) == str(index.task_id)),
            None,
        )
        if prompt_entry is None:
            raise KeyError(f"Unknown BFCL task id '{index.task_id}' for subset '{self._subset}'.")
        answer_entry = None if _is_relevance_subset(self._subset) else self._answers_by_id.get(str(index.task_id))
        return {
            "subset": self._subset,
            "prompt_entry": prompt_entry,
            "possible_answer_entry": answer_entry,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        payloads: list[dict[str, Any]] = []
        for paths in self.get_sessions_paths(sessions):
            result_path = paths.benchmark_results
            if not result_path.exists():
                raise FileNotFoundError(
                    f"Missing BFCL results for planned session '{paths.session_id}' at {result_path}"
                )
            with open(result_path, encoding="utf-8") as handle:
                payloads.append(json.load(handle))

        total_tasks = len(payloads)
        total_score = sum(float(payload.get("score", 0.0)) for payload in payloads)
        successes = sum(1 for payload in payloads if payload.get("success"))
        return BenchmarkResults(
            benchmark_name=f"bfcl-{self._subset}",
            total_tasks=total_tasks,
            score=(total_score / total_tasks) if total_tasks else 0.0,
            metrics={
                "subset": self._subset,
                "success_count": successes,
                "accuracy": (total_score / total_tasks) if total_tasks else 0.0,
            },
        )
