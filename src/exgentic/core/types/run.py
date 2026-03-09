# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import random
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .evaluation import BaseEvaluationConfig
from .session import (
    SessionConfig,
    SessionExecutionStatus,
    SessionResults,
    SessionStatus,
)


class BenchmarkResults(BaseModel):
    """Minimal benchmark-level results returned by Benchmark.aggregate_sessions()."""

    benchmark_name: str
    total_tasks: int
    score: float
    metrics: dict[str, Any] = {}


class RunResults(BaseModel):
    """Aggregated run results produced by the tracker."""

    benchmark_name: str
    benchmark_slug_name: str | None = None
    agent_name: str
    agent_slug_name: str | None = None
    model_name: str | None = None
    model_names: list[str] | None = None
    subset_name: str | None = None
    total_sessions: int
    planned_sessions: Optional[int] = None
    planned_session_ids: Optional[list[str]] = None
    executed_session_ids: list[str] = Field(default_factory=list)
    max_workers: Optional[int] = None
    successful_sessions: int
    # Primary benchmark-level outcome (from benchmark.aggregate_sessions())
    benchmark_score: Optional[float] = None
    benchmark_results: Optional[dict[str, Any]] = None
    average_score: Optional[float] = None
    average_agent_cost: Optional[float] = None
    total_agent_cost: Optional[float] = None
    average_benchmark_cost: Optional[float] = None
    total_benchmark_cost: Optional[float] = None
    total_run_cost: Optional[float] = None
    accumulated_agent_report: Optional[Any] = None
    accumulated_benchmark_report: Optional[Any] = None
    session_results: list[SessionResults]
    average_steps: Optional[float] = None
    average_action_count: Optional[float] = None
    average_invalid_action_count: Optional[float] = None
    average_invalid_action_percent: Optional[float] = None
    percent_finished: Optional[float] = None
    percent_successful: Optional[float] = None
    percent_finished_successful: Optional[float] = None
    percent_finished_unsuccessful: Optional[float] = None
    percent_unfinished: Optional[float] = None
    percent_error: Optional[float] = None
    # Aggregation provenance
    aggregation_mode: Optional[str] = None
    completed_sessions: Optional[int] = None
    incomplete_sessions: Optional[int] = None
    missing_sessions: Optional[int] = None
    running_sessions: Optional[int] = None
    aggregated_session_ids: Optional[list[str]] = None
    skipped_session_ids: Optional[list[str]] = None
    skipped_session_reasons: Optional[dict[str, str]] = None
    missing_result_files: Optional[list[str]] = None


class Integration(BaseModel):
    """Metadata about a benchmark or agent integration."""

    name: str
    type: Literal["benchmark", "agent"]
    version: str
    bundled: bool
    installed: bool
    entry_point: str


class RunStatus(BaseModel):
    """Snapshot of the current run status and existing session artifacts."""

    run_id: str
    output_dir: str
    run_root: str
    run_dir: str
    sessions_root: str
    results_path: str
    results_exists: bool
    benchmark_results_path: str
    benchmark_results_exists: bool
    benchmark_name: str
    benchmark_slug_name: str
    agent_name: str
    agent_slug_name: str
    model_name: Optional[str] = None
    subset_name: Optional[str] = None
    task_ids: list[str]
    total_tasks: int
    session_statuses: list[SessionStatus] = Field(default_factory=list)
    completed_sessions: int = 0
    running_sessions: int = 0
    incomplete_sessions: int = 0
    missing_sessions: int = 0

    @classmethod
    def from_config(cls, run_config: RunConfig) -> RunStatus:
        session_configs = run_config.get_sessions()
        return cls.from_session_configs(run_config, session_configs)

    @classmethod
    def from_session_configs(
        cls,
        run_config: RunConfig,
        session_configs: list[SessionConfig],
    ) -> RunStatus:
        from ...utils.paths import get_run_paths

        context_config = run_config
        if session_configs:
            first = session_configs[0]
            updates = {}
            if run_config.run_id is None and first.run_id:
                updates["run_id"] = first.run_id
            if run_config.cache_dir is None and first.cache_dir:
                updates["cache_dir"] = first.cache_dir
            if run_config.output_dir is None and first.output_dir:
                updates["output_dir"] = first.output_dir
            if updates:
                context_config = run_config.model_copy(update=updates)
        with context_config.get_context():
            run_paths = get_run_paths()
            statuses = [
                SessionStatus.from_config(
                    session_config,
                    run_paths=run_paths,
                )
                for session_config in session_configs
            ]
            task_ids = [str(item.task_id) for item in session_configs]
            completed = sum(1 for item in statuses if item.status == SessionExecutionStatus.COMPLETED)
            running = sum(1 for item in statuses if item.status == SessionExecutionStatus.RUNNING)
            incomplete = sum(1 for item in statuses if item.status == SessionExecutionStatus.INCOMPLETE)
            missing = sum(1 for item in statuses if item.status == SessionExecutionStatus.MISSING)

            return cls(
                run_id=run_paths.run_id,
                output_dir=str(run_paths.root.parent),
                run_root=str(run_paths.root),
                run_dir=str(run_paths.run_dir),
                sessions_root=str(run_paths.sessions_root),
                results_path=str(run_paths.results),
                results_exists=run_paths.results.exists(),
                benchmark_results_path=str(run_paths.benchmark_results),
                benchmark_results_exists=run_paths.benchmark_results.exists(),
                benchmark_name=run_config.benchmark,
                benchmark_slug_name=run_config.benchmark,
                agent_name=run_config.agent,
                agent_slug_name=run_config.agent,
                model_name=(run_config.model or (run_config.agent_kwargs or {}).get("model")),
                subset_name=run_config.subset,
                task_ids=task_ids,
                total_tasks=len(task_ids),
                session_statuses=statuses,
                completed_sessions=completed,
                running_sessions=running,
                incomplete_sessions=incomplete,
                missing_sessions=missing,
            )


class RunPlan(BaseModel):
    """Planned execution derived from RunStatus."""

    run_config: RunConfig
    overwrite_sessions: bool
    to_run: list[SessionConfig] = Field(default_factory=list)
    reuse: list[SessionConfig] = Field(default_factory=list)
    running: list[SessionConfig] = Field(default_factory=list)
    missing: list[SessionConfig] = Field(default_factory=list)
    incomplete: list[SessionConfig] = Field(default_factory=list)

    @classmethod
    def from_config_and_status(
        cls,
        run_config: RunConfig,
        status: RunStatus,
    ) -> RunPlan:
        overwrite_sessions = run_config.overwrite_sessions
        to_run: list[SessionConfig] = []
        reuse: list[SessionConfig] = []
        running: list[SessionConfig] = []
        missing: list[SessionConfig] = []
        incomplete: list[SessionConfig] = []

        for session_status in status.session_statuses:
            session_config = run_config.to_session_config(session_status.task_id)
            match session_status.status:
                case SessionExecutionStatus.RUNNING:
                    running.append(session_config)
                    continue
                case SessionExecutionStatus.MISSING:
                    missing.append(session_config)
                case SessionExecutionStatus.INCOMPLETE:
                    incomplete.append(session_config)
                case SessionExecutionStatus.COMPLETED:
                    if overwrite_sessions:
                        to_run.append(session_config)
                    else:
                        reuse.append(session_config)
                    continue
            to_run.append(session_config)

        return cls(
            run_config=run_config,
            overwrite_sessions=overwrite_sessions,
            to_run=to_run,
            reuse=reuse,
            running=running,
            missing=missing,
            incomplete=incomplete,
        )


class RunConfig(BaseEvaluationConfig):
    """Configuration for a run of multiple sessions."""

    task_ids: Optional[list[str]] = None
    num_tasks: Optional[int] = None
    max_workers: Optional[int] = None
    max_steps: int = 100
    max_actions: int = 100
    overwrite_sessions: bool = False

    @field_validator("max_steps", "max_actions")
    @classmethod
    def _validate_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("limit must be > 0")
        return value

    def to_session_config(self, task_id: str) -> SessionConfig:
        """Derive a SessionConfig for a single task from this run config."""
        return SessionConfig.model_construct(
            benchmark=self.benchmark,
            agent=self.agent,
            task_id=str(task_id),
            subset=self.subset,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            run_id=self.run_id,
            model=self.model,
            overwrite_sessions=self.overwrite_sessions,
            benchmark_kwargs=dict(self.benchmark_kwargs or {}),
            agent_kwargs=dict(self.agent_kwargs or {}),
        )

    def get_session_configs(
        self,
        *,
        resolved_config: RunConfig | None = None,
    ) -> list[SessionConfig]:
        from ...interfaces.registry import load_benchmark

        resolved = resolved_config or self
        task_ids = resolved.task_ids

        if task_ids is None:
            bench_cls = load_benchmark(resolved.benchmark)
            benchmark = bench_cls(**(resolved.benchmark_kwargs or {}))
            try:
                selected = [str(t) for t in benchmark.list_tasks()]
                if resolved.num_tasks is not None:
                    seed = benchmark.seed
                    rng = random.Random(seed if seed is not None else 0)
                    rng.shuffle(selected)
                    selected = selected[: int(resolved.num_tasks)]
            finally:
                benchmark.close()
        else:
            selected = [str(t) for t in task_ids]
            if resolved.num_tasks is not None:
                selected = selected[: int(resolved.num_tasks)]

        return [resolved.to_session_config(task_id) for task_id in selected]

    def get_sessions(self) -> list[SessionConfig]:
        return self.get_session_configs()
