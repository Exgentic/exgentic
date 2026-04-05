# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""TAU2 benchmark adapter — light benchmark class only.

Evaluator, session, and proxy-agent classes live in ``tau2_eval.py``
and are loaded inside the runner subprocess via ``_get_evaluator_class()``
and ``_get_session_class()``.  This file must remain importable without
the ``tau2`` package installed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from ...core import Benchmark


class TAU2Benchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "Tau Bench 2"
    slug_name: ClassVar[str] = "tau2"
    available_subsets: ClassVar[list[str]] = ["mock", "retail", "airline", "telecom"]
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def _get_evaluator_class(cls):
        return "exgentic.benchmarks.tau2.tau2_eval:TAU2Evaluator"

    @classmethod
    def _get_session_class(cls):
        return "exgentic.benchmarks.tau2.tau2_eval:TAU2Session"

    subset: Literal["mock", "retail", "airline", "telecom"] = "retail"
    user_simulator_model: str = "openai/Azure/gpt-4.1"
    llm_temperature_user: float = 0.0
    llm_user_input_cost_per_token: float | None = None
    llm_user_output_cost_per_token: float | None = None
    max_steps: int = 200
    max_errors: int = 10
    num_trials: int = 1
    score_path: str | None = None

    def list_subsets(self) -> list[str]:  # type: ignore[override]
        return list(self.available_subsets)

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "score_path": self.score_path,
        }

    def _get_session_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "user_simulator_model": self.user_simulator_model,
            "llm_temperature_user": self.llm_temperature_user,
            "llm_user_input_cost_per_token": self.llm_user_input_cost_per_token,
            "llm_user_output_cost_per_token": self.llm_user_output_cost_per_token,
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "num_trials": self.num_trials,
            "seed": self.seed,
            "use_cache": self.use_cache,
        }
