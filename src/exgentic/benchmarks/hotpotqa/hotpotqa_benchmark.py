# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import asyncio
import json
import re
import stat
import string
import textwrap
import threading
from collections import Counter
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from ...adapters.schemas.openai import (
    mcp_tools_to_openai_tools,
    openai_tools_to_action_types,
)

# Copied scoring functions from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
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
from ...utils.settings import RunnerName

HOTPOTQA_TOTAL_TASKS = 7405


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0, 0, 0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class HotpotFinishArgs(BaseModel):
    answer: str


class HotpotFinishAction(FinishAction):
    name: Literal["finish"] = "finish"
    arguments: HotpotFinishArgs


class HotpotQASession(Session):
    """Session for HotpotQA benchmark evaluation."""

    _question: str
    _done: bool

    def __init__(
        self,
        with_search_tools: bool,
        instance: dict[str, Any],
        session_id: str | None = None,
        **_kwargs: Any,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id
        self._question = instance["question"]
        self.logger.info(f"question: {self._question}")
        self._gold_answer = instance["answer"]
        self._task_id = instance["task_id"]
        self._done = False
        self._final_answer = None
        self._with_search_tools = with_search_tools
        self._registry = ActionsHandler(logger=self.logger)
        self._mcp_ready = threading.Event()
        self._mcp_error: BaseException | None = None

        self.mcp_thread: threading.Thread | None = None
        if self._with_search_tools:
            self.mcp_thread = threading.Thread(target=self.run_wikipedia_server, daemon=True)
            self.mcp_thread.start()
            ready = self._mcp_ready.wait(timeout=60.0)
            if not ready or self._mcp_error is not None:
                err = self._mcp_error
                raise RuntimeError(f"MCP initialization failed or timed out: {err}") from err
        else:
            # No search tools requested; skip MCP startup.
            self._mcp_ready.set()
        # Only 'finish' is provided as the completion action
        self._registry.add_action(
            name="finish",
            description="Submit the final answer and complete the task.",
            action_cls=HotpotFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )
        super().__init__()

    def run_wikipedia_server(self):
        asyncio.run(self.run_wikipedia_server_async())

    def _mcp_client_config(self) -> dict[str, Any]:
        """Return a FastMCP client config that logs wikipedia-mcp stderr to the session benchmark dir."""
        log_path = self.paths.benchmark_dir / "wikipedia_mcp.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        from ...core.context import context_env

        ctx_env = context_env()
        ctx_env_json = json.dumps(ctx_env)

        # Generate a tiny Python wrapper to keep stdout for JSONRPC and send stderr to a file.
        wrapper_path = self.paths.benchmark_dir / "wikipedia_mcp_wrapper.py"
        wrapper_code = (
            textwrap.dedent(
                f"""
            #!/usr/bin/env python3
            import subprocess, sys, os, json

            log = open({str(log_path)!r}, "ab", buffering=0)
            os.environ.update(json.loads({ctx_env_json!r}))
            proc = subprocess.Popen(
                ["wikipedia-mcp", "--transport", "stdio"],
                stderr=log,
            )
            proc.wait()
            sys.exit(proc.returncode)
            """
            ).strip()
            + "\n"
        )
        wrapper_path.write_text(wrapper_code, encoding="utf-8")
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        command = str(wrapper_path)
        return {"mcpServers": {"wiki": {"command": command, "args": []}}}

    async def run_wikipedia_server_async(self):
        if not self._with_search_tools:
            self._mcp_ready.set()
            return
        try:
            from fastmcp import Client

            config = self._mcp_client_config()
            mcp_client = Client(config)

            async with mcp_client:
                tools = await mcp_client.list_tools()

            openai_tools = mcp_tools_to_openai_tools(tools)
            if self._with_search_tools:
                self._registry.add_actions(
                    openai_tools_to_action_types(openai_tools),
                    self._handle_mcp_action,
                )
        except Exception as e:
            self._mcp_error = e
            self.logger.exception(f"Failed to initialize wikipedia-mcp: {e}")
            raise
        finally:
            self._mcp_ready.set()

    @property
    def task(self) -> str:
        return (
            "Answer the user question. Submit the final answer as a short phrase by calling 'finish'. "
            "If the question is yes/no, answer 'yes' or 'no'. Do not add explanations.\n\n"
            f"Question: {self._question}"
        )

    @property
    def context(self) -> dict[str, Any]:
        return {}

    @property
    def task_id(self) -> str:
        return str(self._task_id)

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    def _to_observation(self, raw: Any, invoking: list[SingleAction] | None = None) -> Observation:
        return SingleObservation(invoking_actions=invoking or [], result=raw)

    def start(self) -> Observation | None:
        # Empty initial observation; question is carried in the task string.
        return EmptyObservation()

    def run_mcp_command(self, name, arguments) -> Any:
        return asyncio.run(self.run_mcp_command_async(name, arguments))

    async def run_mcp_command_async(self, name, arguments) -> Any:
        from fastmcp import Client

        config = self._mcp_client_config()
        mcp_client = Client(config)

        async with mcp_client:
            response = await mcp_client.call_tool(name=name, arguments=arguments.model_dump())
            # print(response.structured_content)
            return response.structured_content

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
        # Minimal: compute F1 and always mark success
        try:
            f1, precision, recall = f1_score(self._gold_answer, self._final_answer)
            score = float(f1)
        except Exception:
            score = 0.0
        self.logger.info(f"Gold: {self._gold_answer} Prediction: {self._final_answer} Score: {score}")
        # Finished only when the benchmark finish action stores a final answer.
        finished = self._final_answer is not None
        success = score >= 1.0 - 1e-6
        return SessionScore(score=score, success=success, is_finished=finished)

    def close(self):
        super().close()
        # Persist minimal results for aggregation
        sc = self.score()
        self.save_standard_results(sc)
        self.logger.debug("Closing MCP server..")
        if self.mcp_thread and self.mcp_thread.is_alive():
            self.logger.debug("Waiting for MCP server to shut down.")
            self.mcp_thread.join(timeout=60.0)
            if self.mcp_thread.is_alive():
                self.logger.warning("MCP server thread did shutdown cleanly, continuing anyway.")
            else:
                self.logger.debug("MCP server shutdown cleanly.")

    # Action handlers ------------------------------------------------------------
    def _handle_finish(self, action: SingleAction) -> Any:
        self.logger.info(f"Received final answer: {action}")
        answer = extract_argument(action.arguments, "answer", None)
        self._final_answer = answer
        self._done = True
        return None

    def _handle_mcp_action(self, action: SingleAction) -> Any:
        result = self.run_mcp_command(action.name, action.arguments)
        return result


# ── Evaluator ────────────────────────────────────────────────────────


class HotpotQAEvaluator(Evaluator):
    """Evaluator for HotpotQA — task discovery, session kwargs, aggregation."""

    def __init__(self, subset: str = "distractor", with_search_tools: bool = True) -> None:
        self._subset = subset
        self._with_search_tools = with_search_tools
        self._dataset = None

    def _ensure_dataset(self) -> None:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")["validation"]

    def list_tasks(self) -> list[str]:
        return [str(i) for i in range(HOTPOTQA_TOTAL_TASKS)]

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        self._ensure_dataset()
        idx = int(index.task_id)
        if idx < 0 or idx >= len(self._dataset):
            raise IndexError(f"Task id {index.task_id} out of range for HotpotQA.")
        instance = {"task_id": idx, **self._dataset[idx]}
        return {
            "with_search_tools": self._with_search_tools,
            "instance": instance,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores: list[float] = []
        for paths in self.get_sessions_paths(sessions):
            with open(paths.benchmark_results, encoding="utf-8-sig") as f:
                payload = json.load(f)
            s = float(payload["score"])
            scores.append(s)
        avg = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResults(
            benchmark_name="hotpotqa",
            total_tasks=len(sessions),
            score=avg,
            metrics={},
        )


# ── Benchmark config ─────────────────────────────────────────────────


class HotpotQABenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "HotpotQA"
    slug_name: ClassVar[str] = "hotpotqa"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_evaluator_class(cls):
        return HotpotQAEvaluator

    @classmethod
    def get_session_class(cls):
        return HotpotQASession

    subset: Literal["distractor"] = "distractor"
    with_search_tools: bool = True
    runner: RunnerName | None = "direct"  # Code is threadsafe

    def get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "with_search_tools": self.with_search_tools,
        }
