# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Optional

from filelock import FileLock, Timeout
from pydantic import BaseModel, Field, field_validator

from ...utils.cost import CostReport
from .evaluation import BaseEvaluationConfig


class SessionOutcomeStatus(StrEnum):
    SUCCESS = "success"
    UNSUCCESSFUL = "unsuccessful"
    UNFINISHED = "unfinished"
    LIMIT_REACHED = "limit_reached"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class SessionExecutionStatus(StrEnum):
    MISSING = "missing"
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"
    RUNNING = "running"


class SessionResults(BaseModel):
    """Results of a single session execution."""

    session_id: str
    success: bool
    score: Optional[float] = None
    is_finished: Optional[bool] = None
    status: SessionOutcomeStatus = SessionOutcomeStatus.UNKNOWN
    steps: int
    action_count: int = 0
    invalid_action_count: int = 0
    agent_cost: float
    benchmark_cost: float
    execution_time: float
    details: dict[str, Any] = {}
    cost_reports: dict[str, CostReport] = Field(default_factory=dict)
    task_id: Optional[str] = None

    @field_validator("cost_reports", mode="before")
    @classmethod
    def accept_instances(cls, v):
        if not isinstance(v, dict):
            raise TypeError(f"cost_reports must be a dict[str, CostReport | dict], got {type(v)}")

        for key, val in v.items():
            if isinstance(val, CostReport) or isinstance(val, dict):
                continue

            raise TypeError(f"Invalid type for cost_reports[{key}]: {type(val)}")

        return v


class SessionScore(BaseModel):
    """Minimal per-session score returned by Session.score()."""

    score: float
    success: bool
    is_finished: Optional[bool] = None
    session_metrics: dict[str, Any] = {}
    session_metadata: dict[str, Any] = {}


class SessionStatus(BaseModel):
    """Filesystem status for a single task session."""

    task_id: str
    session_id: str
    results_path: str
    session_dir: str
    status: SessionExecutionStatus
    result_status: Optional[SessionOutcomeStatus] = None

    @classmethod
    def _is_session_locked(cls, session_paths) -> bool:
        if not session_paths.lock.exists():
            return False
        lock = FileLock(str(session_paths.lock))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return True
        lock.release()
        return False

    @classmethod
    def _extract_result_status(cls, results_path) -> SessionOutcomeStatus:
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            return SessionOutcomeStatus.UNKNOWN
        status = payload.get("status")
        if status:
            try:
                return SessionOutcomeStatus(str(status))
            except ValueError:
                return SessionOutcomeStatus.UNKNOWN
        success = payload.get("success")
        is_finished = payload.get("is_finished")
        details = payload.get("details") or {}
        metadata = details.get("session_metadata") or {}
        error_source = metadata.get("error_source")
        if error_source == "cancelled":
            return SessionOutcomeStatus.CANCELLED
        if metadata.get("error") or error_source in ("agent", "benchmark"):
            return SessionOutcomeStatus.ERROR
        if is_finished is True:
            return SessionOutcomeStatus.SUCCESS if success else SessionOutcomeStatus.UNSUCCESSFUL
        if is_finished is False:
            return SessionOutcomeStatus.UNFINISHED
        return SessionOutcomeStatus.UNKNOWN

    @classmethod
    def from_config(cls, session_config, *, run_paths=None) -> SessionStatus:
        from ...core.context import get_context
        from ...utils.paths import RunPaths

        session_id = session_config.get_session_id()
        if run_paths is None:
            ctx = get_context()
            if session_config.run_id:
                run_paths = RunPaths(run_id=session_config.run_id, output_dir=ctx.output_dir)
            else:
                run_paths = RunPaths.from_context(ctx)
        sess_paths = run_paths.session(session_id)
        results_exists = sess_paths.results.exists()
        session_dir_exists = sess_paths.root.exists()
        is_locked = cls._is_session_locked(sess_paths)
        result_status: Optional[SessionOutcomeStatus] = None
        if is_locked:
            status = SessionExecutionStatus.RUNNING
        elif results_exists:
            result_status = cls._extract_result_status(sess_paths.results)
            if result_status in (
                SessionOutcomeStatus.ERROR,
                SessionOutcomeStatus.CANCELLED,
                SessionOutcomeStatus.UNKNOWN,
            ):
                status = SessionExecutionStatus.INCOMPLETE
            else:
                status = SessionExecutionStatus.COMPLETED
        elif session_dir_exists:
            status = SessionExecutionStatus.INCOMPLETE
        else:
            status = SessionExecutionStatus.MISSING
        return cls(
            task_id=str(session_config.task_id),
            session_id=session_id,
            results_path=str(sess_paths.results),
            session_dir=str(sess_paths.root),
            status=status,
            result_status=result_status,
        )


class SessionIndex(BaseModel):
    """Minimal mapping between a task id and a session id."""

    task_id: str
    session_id: str


class SessionConfig(BaseEvaluationConfig):
    """Configuration for running a single session."""

    task_id: str
    overwrite_sessions: bool = False

    def session_id_payload(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "benchmark_kwargs": dict(self.benchmark_kwargs or {}),
            "agent": self.agent,
            "agent_kwargs": dict(self.agent_kwargs or {}),
            "subset": self.subset,
            "task_id": str(self.task_id),
            "model": self.model,
        }

    def get_session_id(self) -> str:
        encoded = json.dumps(
            self.session_id_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:8]

    def to_index(self) -> SessionIndex:
        return SessionIndex(
            task_id=str(self.task_id),
            session_id=self.get_session_id(),
        )

    def get_context(self):
        from ..context import run_scope, session_scope

        @contextmanager
        def _ctx():
            with run_scope(
                output_dir=self.output_dir,
                cache_dir=self.cache_dir,
                run_id=self.run_id,
            ):
                with session_scope(self.get_session_id(), task_id=str(self.task_id)):
                    yield

        return _ctx()
