# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""BFCL benchmark adapter — light benchmark class only.

Evaluator, session, and helper classes live in ``bfcl_eval.py``
and are loaded inside the runner subprocess via ``get_evaluator_class()``
and ``get_session_class()``.  This file must remain importable without
the ``bfcl_eval`` package installed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from ...core import Benchmark
from ...core.types import FinishAction

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


class BFCLFinishArgs(BaseModel):
    content: str = ""


class BFCLFinishAction(FinishAction):
    name: Literal["finish"] = "finish"
    arguments: BFCLFinishArgs


class BFCLBenchmark(Benchmark, BaseModel):
    """BFCL benchmark using Gorilla assets with an Exgentic-native runtime."""

    display_name: ClassVar[str] = "BFCL"
    slug_name: ClassVar[str] = "bfcl"
    available_subsets: ClassVar[list[str]] = [
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
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    @classmethod
    def get_evaluator_class(cls):
        return "exgentic.benchmarks.bfcl.bfcl_eval:BFCLEvaluator"

    @classmethod
    def get_session_class(cls):
        return "exgentic.benchmarks.bfcl.bfcl_eval:BFCLSession"

    subset: BFCLSubset = "simple_python"

    def list_subsets(self) -> list[str]:  # type: ignore[override]
        return list(self.available_subsets)

    def get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
        }
