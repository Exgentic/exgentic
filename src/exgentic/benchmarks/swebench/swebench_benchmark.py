# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""SWE-bench benchmark adapter -- light benchmark class only.

Evaluator, session, and helper classes live in ``swebench_eval.py``
and are loaded inside the runner subprocess via ``get_evaluator_class()``
and ``get_session_class()``.  This file must remain importable without
the ``swebench`` package installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ...core import Benchmark
from ...core.types import FinishAction, SingleAction
from ...core.types import SessionScore as BaseSessionScore

# =============================================================================
# Shared action / score types (importable without heavy deps)
# =============================================================================


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to execute")


class BashAction(SingleAction):
    name: str = "bash"
    arguments: BashArgs


class SubmitPatchArgs(BaseModel):
    summary: str = Field(description="Brief textual summary of the fix (no diff/patch)")


class SubmitPatchAction(FinishAction):
    name: str = "finish"
    arguments: SubmitPatchArgs


class SessionScore(BaseSessionScore):
    instance_id: str = ""
    agent: dict[str, Any] = Field(default_factory=dict)
    patch: dict[str, Any] = Field(default_factory=dict)
    container: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Configuration (light -- only stdlib + yaml lazy)
# =============================================================================

_CONFIG: dict[str, Any] = None


def get_config() -> dict[str, Any]:
    import yaml

    global _CONFIG
    if _CONFIG is None:
        path = Path(__file__).parent / "config.yaml"
        _CONFIG = yaml.safe_load(path.read_text())
    return _CONFIG


# =============================================================================
# Benchmark
# =============================================================================


class SWEBenchBenchmark(Benchmark):
    """Benchmark configuration for SWE-bench evaluation."""

    display_name: ClassVar[str] = "SWE-bench"
    slug_name: ClassVar[str] = "swebench"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_evaluator_class(cls):
        return "exgentic.benchmarks.swebench.swebench_eval:SWEBenchEvaluator"

    @classmethod
    def get_session_class(cls):
        return "exgentic.benchmarks.swebench.swebench_eval:SWEBenchSession"

    subset: str | None = None
    require_submit_for_patch_evaluation: bool = True
    docker_socket: bool = True  # SWE-bench sessions create sibling Docker containers

    def model_post_init(self, __context):
        cfg = get_config()
        benchmark_cfg = cfg["benchmark"]
        session_cfg = cfg["session"]
        if self.subset is None:
            self.subset = benchmark_cfg["subset"]
        if self.runner is None:
            self.runner = benchmark_cfg.get("runner")
        if "seed" in benchmark_cfg:
            self.seed = benchmark_cfg["seed"]
        if (
            "require_submit_for_patch_evaluation" in benchmark_cfg
            and "require_submit_for_patch_evaluation" not in self.model_fields_set
        ):
            self.require_submit_for_patch_evaluation = bool(benchmark_cfg["require_submit_for_patch_evaluation"])
        if self.max_interactions is None:
            self.max_interactions = session_cfg.get("max_interactions")

    def get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "require_submit_for_patch_evaluation": self.require_submit_for_patch_evaluation,
            "max_interactions": self.max_interactions,
        }
