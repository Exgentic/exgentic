# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import contextvars
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..utils.paths import sanitize_path_component
from ..utils.settings import get_settings

# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class Role(str, Enum):
    FRAMEWORK = "framework"
    AGENT = "agent"
    BENCHMARK = "benchmark"
    AGGREGATOR = "aggregator"


# ---------------------------------------------------------------------------
# OTEL Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OtelContext:
    """OpenTelemetry span context for distributed tracing."""

    trace_id: str
    span_id: str


# ---------------------------------------------------------------------------
# Core Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    run_id: str
    output_dir: str
    cache_dir: str
    session_id: str | None = None
    task_id: str | None = None
    role: Role = Role.FRAMEWORK
    otel_context: OtelContext | None = None

    def with_session(self, session_id: str, task_id: str | None = None) -> Context:
        return replace(self, session_id=session_id, task_id=task_id)

    def with_role(self, role: Role) -> Context:
        return replace(self, role=role)

    def with_otel_context(self, otel_context: OtelContext | None) -> Context:
        """Create a new Context with updated OTEL context."""
        return replace(self, otel_context=otel_context)


# ---------------------------------------------------------------------------
# RuntimeConfig — Pydantic model for runtime.json
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Schema for runtime.json -- the on-disk context + settings snapshot."""

    # Context
    run_id: str
    output_dir: str
    cache_dir: str
    session_id: str | None = None
    task_id: str | None = None
    role: str = "framework"

    # OTEL
    otel_trace_id: str | None = None
    otel_span_id: str | None = None

    # Settings (only non-default overrides)
    settings: dict[str, Any] = {}

    def to_context(self) -> Context:
        """Convert to an in-process Context."""
        otel_context = None
        if self.otel_trace_id and self.otel_span_id:
            otel_context = OtelContext(trace_id=self.otel_trace_id, span_id=self.otel_span_id)
        try:
            role = Role(self.role)
        except ValueError:
            role = Role.FRAMEWORK
        return Context(
            run_id=self.run_id,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            session_id=self.session_id,
            task_id=self.task_id,
            role=role,
            otel_context=otel_context,
        )

    @classmethod
    def from_current(cls, role: Role | None = None) -> RuntimeConfig:
        """Snapshot the current context + settings into a RuntimeConfig.

        If *role* is provided it overrides the role on the current context.
        This is the path runners take: they know the role of the service
        they're spawning regardless of what's on the in-process ContextVar.
        """
        ctx = get_context()
        settings = get_settings()
        effective_role = role if role is not None else ctx.role
        return cls(
            run_id=ctx.run_id,
            output_dir=ctx.output_dir,
            cache_dir=ctx.cache_dir,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            role=effective_role.value,
            otel_trace_id=ctx.otel_context.trace_id if ctx.otel_context else None,
            otel_span_id=ctx.otel_context.span_id if ctx.otel_context else None,
            settings=settings.get_overrides(),
        )

    def apply_settings(self) -> None:
        """Apply settings overrides to os.environ so get_settings() picks them up."""
        for name, value in self.settings.items():
            env_key = f"EXGENTIC_{name}".upper()
            if isinstance(value, bool):
                os.environ.setdefault(env_key, "true" if value else "false")
            else:
                os.environ.setdefault(env_key, str(value))


# ---------------------------------------------------------------------------
# Single ContextVar -- the single source of truth
# ---------------------------------------------------------------------------

_CONTEXT: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "exgentic_context",
    default=None,
)

# Fallback for threads that don't inherit ContextVar (uvicorn thread-pool
# workers, service runner threads). Set by init_context() and
# set_context_fallback().
_SUBPROCESS_CONTEXT: Context | None = None


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_context() -> Context:
    """Return the current Context. Raises RuntimeError if none is set."""
    ctx = _CONTEXT.get()
    if ctx is None:
        ctx = _SUBPROCESS_CONTEXT
    if ctx is None:
        raise RuntimeError("No context set. Use run_scope() or init_context().")
    return ctx


def try_get_context() -> Context | None:
    """Return the current Context, or None if none is set."""
    ctx = _CONTEXT.get()
    return ctx if ctx is not None else _SUBPROCESS_CONTEXT


def set_context(ctx: Context) -> None:
    """Imperatively set the current context."""
    _CONTEXT.set(ctx)


def set_context_fallback(ctx: Context | None) -> None:
    """Set a process-wide fallback for threads that don't inherit ContextVar."""
    global _SUBPROCESS_CONTEXT
    _SUBPROCESS_CONTEXT = ctx


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def run_scope(
    ctx: Context | None = None,
    *,
    run_id: str | None = None,
    output_dir: str | None = None,
    cache_dir: str | None = None,
    overwrite_run: bool = False,
) -> Iterator[Context]:
    """Enter a run context.

    Either pass an explicit *ctx*, or pass keyword args and the Context will
    be resolved from those args / settings defaults.
    """
    if ctx is None:
        ctx = _resolve_context(run_id, output_dir, cache_dir, overwrite_run)
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)


@contextmanager
def session_scope(session_id: str, task_id: str | None = None) -> Iterator[Context]:
    """Derive a session-scoped context from the current run context.

    runtime.json is not written here — each runner writes its own
    per-service file at spawn time via :func:`save_service_runtime`.
    """
    parent = get_context()
    ctx = parent.with_session(session_id, task_id)
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)


# ---------------------------------------------------------------------------
# Runtime file helpers
# ---------------------------------------------------------------------------


def _derive_runtime_path(ctx: Context | None, role: Role | None = None) -> Path | None:
    """Return the path to a service's ``runtime.json``.

    - With session_id: ``{output_dir}/{run_id}/sessions/{session_id}/{role}/runtime.json``
    - Without session_id (run-level services, e.g. aggregation):
      ``{output_dir}/{run_id}/{role}/runtime.json``

    Returns ``None`` when the context is missing output_dir or run_id.
    """
    if ctx is None or not ctx.output_dir or not ctx.run_id:
        return None
    effective_role = role if role is not None else ctx.role
    base = Path(ctx.output_dir) / ctx.run_id
    if ctx.session_id:
        base = base / "sessions" / ctx.session_id
    return base / effective_role.value / "runtime.json"


_RUNTIME_FILE_ENV = "EXGENTIC_RUNTIME_FILE"


def get_runtime_env() -> dict[str, str]:
    """Return env vars needed by a child process to find runtime.json.

    Callers should merge this into the subprocess env dict.
    Returns an empty dict when no runtime file is known.
    """
    path = os.environ.get(_RUNTIME_FILE_ENV)
    if path:
        return {_RUNTIME_FILE_ENV: path}
    return {}


def save_service_runtime(role: Role) -> Path:
    """Write a per-service ``runtime.json`` for *role* and return its path.

    The file is written to
    ``{output_dir}/{run_id}/sessions/{session_id}/{role}/runtime.json``
    (or the run-level fallback ``{output_dir}/{run_id}/{role}/runtime.json``
    when no session_id is on the context, e.g. aggregation).

    Called by runners before spawning a service subprocess.  Does NOT
    mutate the current process's env vars — the caller puts the returned
    path into the child's env via ``EXGENTIC_RUNTIME_FILE``.
    """
    path = _derive_runtime_path(get_context(), role=role)
    if path is None:
        raise RuntimeError("Cannot derive runtime.json path: current context must have " "output_dir and run_id set.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RuntimeConfig.from_current(role=role).model_dump_json(indent=2))
    return path


def _load_runtime(runtime_file: Path) -> Context | None:
    """Read ``runtime.json`` from *runtime_file* and return a Context, or None."""
    if not runtime_file.exists():
        return None
    config = RuntimeConfig.model_validate_json(runtime_file.read_text())
    config.apply_settings()
    return config.to_context()


def init_context() -> Context:
    """Bootstrap ContextVar from ``runtime.json`` on disk.

    The env var pointing to the runtime file is set automatically by
    :func:`get_runtime_env`.  Contains everything: context, settings, OTEL.

    Raises :class:`RuntimeError` if the env var is missing or the file
    cannot be loaded.
    """
    global _SUBPROCESS_CONTEXT

    runtime_file = os.environ.get(_RUNTIME_FILE_ENV)
    if not runtime_file:
        raise RuntimeError(
            f"No {_RUNTIME_FILE_ENV} set. Use run_scope() or ensure the orchestrator wrote runtime.json."
        )
    ctx = _load_runtime(Path(runtime_file))
    if ctx is None:
        raise RuntimeError(
            f"runtime.json not found at {runtime_file}. "
            "The spawning runner must write runtime.json before spawning children."
        )
    _CONTEXT.set(ctx)
    _SUBPROCESS_CONTEXT = ctx
    # Keep the env var set so worker threads spawned by this process
    # (uvicorn workers, tau2 thread pools) can bootstrap via try_init_context.
    os.environ[_RUNTIME_FILE_ENV] = runtime_file
    return ctx


def try_init_context() -> Context | None:
    """Like :func:`init_context` but returns ``None`` on failure."""
    runtime_file = os.environ.get(_RUNTIME_FILE_ENV)
    if not runtime_file:
        return None
    return init_context()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_context(
    run_id: str | None,
    output_dir: str | None,
    cache_dir: str | None,
    overwrite_run: bool,
) -> Context:
    settings = get_settings()
    resolved_run_id = run_id or datetime.now().isoformat().replace(":", "--")
    resolved_run_id = sanitize_path_component(resolved_run_id)
    resolved_output_dir = output_dir or settings.output_dir
    resolved_output_dir = str(Path(resolved_output_dir).expanduser().resolve())
    resolved_cache_dir = cache_dir or settings.cache_dir
    resolved_cache_dir = str(Path(resolved_cache_dir).expanduser().resolve())
    if overwrite_run:
        run_root = Path(resolved_output_dir) / resolved_run_id
        if run_root.exists():
            shutil.rmtree(run_root)
    return Context(
        run_id=resolved_run_id,
        output_dir=resolved_output_dir,
        cache_dir=resolved_cache_dir,
    )
