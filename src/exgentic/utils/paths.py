# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_WINDOWS_FORBIDDEN = set('<>:"/\\|?*')


def sanitize_path_component(value: str) -> str:
    cleaned = "".join("_" if ch in _WINDOWS_FORBIDDEN else ch for ch in value)
    cleaned = cleaned.rstrip(" .")
    return cleaned or "run"


@dataclass(frozen=True)
class SessionPaths:
    """All filesystem paths for a single session.

    Stores resolved values — no lazy lookups, no context dependency.
    """

    session_id: str
    run_id: str
    output_dir: Path

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    @classmethod
    def from_context(cls, ctx) -> SessionPaths:
        if ctx.session_id is None:
            raise ValueError("Context has no session_id")
        return cls(session_id=ctx.session_id, run_id=ctx.run_id, output_dir=ctx.output_dir)

    @property
    def root(self) -> Path:
        return self.output_dir / self.run_id / "sessions" / self.session_id

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def trajectory(self) -> Path:
        return self.root / "trajectory.jsonl"

    @property
    def benchmark_dir(self) -> Path:
        return self.root / "benchmark"

    @property
    def agent_dir(self) -> Path:
        return self.root / "agent"

    @property
    def benchmark_results(self) -> Path:
        return self.benchmark_dir / "results.json"

    @property
    def benchmark_config(self) -> Path:
        return self.benchmark_dir / "config.json"

    @property
    def benchmark_task(self) -> Path:
        return self.benchmark_dir / "task.json"

    @property
    def benchmark_context(self) -> Path:
        return self.benchmark_dir / "context.json"

    @property
    def session_manifest(self) -> Path:
        return self.root / "session.json"

    @property
    def session_config(self) -> Path:
        return self.root / "config.json"

    @property
    def session_log(self) -> Path:
        return self.benchmark_dir / "session.log"

    @property
    def agent_log(self) -> Path:
        return self.agent_dir / "agent.log"

    @property
    def error_log(self) -> Path:
        return self.root / "error.log"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def otel_log(self) -> Path:
        return self.root / "otel.log"

    @property
    def lock(self) -> Path:
        return self.root / "session.lock"


@dataclass(frozen=True)
class RunPaths:
    """All filesystem paths for a single run.

    Stores resolved values — no lazy lookups, no context dependency.
    """

    run_id: str
    output_dir: Path

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    @classmethod
    def from_context(cls, ctx) -> RunPaths:
        return cls(run_id=ctx.run_id, output_dir=ctx.output_dir)

    @property
    def root(self) -> Path:
        return self.output_dir / self.run_id

    @property
    def sessions_root(self) -> Path:
        return self.root / "sessions"

    @property
    def run_dir(self) -> Path:
        return self.root / "run"

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def benchmark_results(self) -> Path:
        return self.root / "benchmark_results.json"

    @property
    def tracker(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def warnings(self) -> Path:
        return self.run_dir / "warnings.log"

    @property
    def config(self) -> Path:
        return self.run_dir / "config.json"

    def session(self, session_id: str) -> SessionPaths:
        return SessionPaths(session_id=session_id, run_id=self.run_id, output_dir=self.output_dir)


# ---------------------------------------------------------------------------
# Convenience accessors — thin wrappers over get_context()
# ---------------------------------------------------------------------------


def get_run_id() -> str:
    """Return the current run ID from context."""
    from ..core.context import get_context

    return get_context().run_id


def get_run_paths() -> RunPaths:
    """Return RunPaths for the current context."""
    from ..core.context import get_context

    return RunPaths.from_context(get_context())


def get_session_paths(session_id: str | None = None) -> SessionPaths:
    """Return SessionPaths for the given (or current) session.

    If *session_id* is ``None``, uses the session ID from the current context.
    """
    from ..core.context import get_context

    ctx = get_context()
    sid = session_id or ctx.session_id
    if sid is None:
        raise ValueError("No session_id provided and none in context")
    return RunPaths.from_context(ctx).session(sid)
