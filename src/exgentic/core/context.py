# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import contextvars
import os
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator

from ..utils.paths import sanitize_path_component
from ..utils.settings import get_settings

# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class Role(str, Enum):
    FRAMEWORK = "framework"
    AGENT = "agent"
    BENCHMARK = "benchmark"


# ---------------------------------------------------------------------------
# OTEL Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OtelContext:
    """OpenTelemetry span context for distributed tracing."""

    trace_id: str
    span_id: str


# ---------------------------------------------------------------------------
# Env-var keys used for subprocess transport
# ---------------------------------------------------------------------------

_ENV_RUN_ID = "EXGENTIC_CTX_RUN_ID"
_ENV_OUTPUT_DIR = "EXGENTIC_CTX_OUTPUT_DIR"
_ENV_CACHE_DIR = "EXGENTIC_CTX_CACHE_DIR"
_ENV_SESSION_ID = "EXGENTIC_CTX_SESSION_ID"
_ENV_TASK_ID = "EXGENTIC_CTX_TASK_ID"
_ENV_ROLE = "EXGENTIC_CTX_ROLE"
ENV_OTEL_TRACE_ID = "EXGENTIC_CTX_OTEL_TRACE_ID"
ENV_OTEL_SPAN_ID = "EXGENTIC_CTX_OTEL_SPAN_ID"
OTEL_ENABLED_ENV = "EXGENTIC_OTEL_ENABLED"


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
        return Context(
            run_id=self.run_id,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            session_id=session_id,
            task_id=task_id,
            role=self.role,
            otel_context=self.otel_context,
        )

    def with_role(self, role: Role) -> Context:
        return Context(
            run_id=self.run_id,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            session_id=self.session_id,
            task_id=self.task_id,
            role=role,
            otel_context=self.otel_context,
        )

    def with_otel_context(self, otel_context: OtelContext | None) -> Context:
        """Create a new Context with updated OTEL context."""
        return Context(
            run_id=self.run_id,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            session_id=self.session_id,
            task_id=self.task_id,
            role=self.role,
            otel_context=otel_context,
        )

    def to_env(self) -> dict[str, str]:
        env: dict[str, str] = {
            _ENV_RUN_ID: self.run_id,
            _ENV_OUTPUT_DIR: self.output_dir,
            _ENV_CACHE_DIR: self.cache_dir,
            _ENV_ROLE: self.role.value,
        }
        if self.session_id is not None:
            env[_ENV_SESSION_ID] = self.session_id
        if self.task_id is not None:
            env[_ENV_TASK_ID] = self.task_id
        if self.otel_context is not None:
            env[ENV_OTEL_TRACE_ID] = self.otel_context.trace_id
            env[ENV_OTEL_SPAN_ID] = self.otel_context.span_id
        return env

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Context:
        src = env if env is not None else os.environ
        run_id = src.get(_ENV_RUN_ID, "")
        if not run_id:
            raise RuntimeError(f"{_ENV_RUN_ID} not set in environment.")
        run_id = sanitize_path_component(run_id)
        output_dir = src.get(_ENV_OUTPUT_DIR) or get_settings().output_dir
        cache_dir = src.get(_ENV_CACHE_DIR) or get_settings().cache_dir
        session_id = src.get(_ENV_SESSION_ID) or None
        task_id = src.get(_ENV_TASK_ID) or None
        role_str = src.get(_ENV_ROLE)
        try:
            role = Role(role_str) if role_str else Role.FRAMEWORK
        except ValueError:
            role = Role.FRAMEWORK

        # Read OTEL context if present
        otel_context: OtelContext | None = None
        trace_id = src.get(ENV_OTEL_TRACE_ID)
        span_id = src.get(ENV_OTEL_SPAN_ID)
        if trace_id and span_id:
            otel_context = OtelContext(trace_id=trace_id, span_id=span_id)

        return cls(
            run_id=run_id,
            output_dir=output_dir,
            cache_dir=cache_dir,
            session_id=session_id,
            task_id=task_id,
            role=role,
            otel_context=otel_context,
        )


# ---------------------------------------------------------------------------
# Single ContextVar — the single source of truth
# ---------------------------------------------------------------------------

_CONTEXT: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "exgentic_context",
    default=None,
)

# Fallback for threads that don't inherit ContextVar (uvicorn thread-pool
# workers, service runner threads). Set by init_context_from_env() and
# set_context_fallback().
_SUBPROCESS_CONTEXT: Context | None = None

_ENV_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_context() -> Context:
    """Return the current Context. Raises RuntimeError if none is set."""
    ctx = _CONTEXT.get()
    if ctx is None:
        ctx = _SUBPROCESS_CONTEXT
    if ctx is None:
        raise RuntimeError("No context set. Use run_scope() or init_context_from_env().")
    return ctx


def try_get_context() -> Context | None:
    """Return the current Context, or None if none is set."""
    ctx = _CONTEXT.get()
    return ctx if ctx is not None else _SUBPROCESS_CONTEXT


def context_env() -> dict[str, str]:
    """Return context env vars for subprocess propagation, or empty dict."""
    ctx = try_get_context()
    if ctx is None:
        return {}
    return ctx.to_env()


@contextmanager
def context_env_scope() -> Iterator[None]:
    """Temporarily apply context env vars to os.environ (thread-safe)."""
    env = context_env()
    if not env:
        yield
        return
    with _ENV_LOCK:
        prev = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            yield
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


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
    be resolved from those args / env vars / settings defaults.
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
    """Derive a session-scoped context from the current run context."""
    parent = get_context()
    ctx = parent.with_session(session_id, task_id)
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)


@contextmanager
def agent_scope() -> Iterator[Context]:
    """Set role=AGENT for the duration of the block, restore on exit."""
    parent = get_context()
    ctx = parent.with_role(Role.AGENT)
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)


@contextmanager
def benchmark_scope() -> Iterator[Context]:
    """Set role=BENCHMARK for the duration of the block, restore on exit."""
    parent = get_context()
    ctx = parent.with_role(Role.BENCHMARK)
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)


def init_context_from_env() -> Context:
    """Bootstrap ContextVar from env vars (called once in subprocess / Docker)."""
    global _SUBPROCESS_CONTEXT
    ctx = Context.from_env()
    _CONTEXT.set(ctx)
    _SUBPROCESS_CONTEXT = ctx
    return ctx


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
    resolved_run_id = run_id or os.environ.get(_ENV_RUN_ID) or datetime.now().isoformat().replace(":", "--")
    resolved_run_id = sanitize_path_component(resolved_run_id)
    resolved_output_dir = output_dir or os.environ.get(_ENV_OUTPUT_DIR) or settings.output_dir
    resolved_output_dir = str(Path(resolved_output_dir).resolve())
    resolved_cache_dir = cache_dir or os.environ.get(_ENV_CACHE_DIR) or settings.cache_dir
    resolved_cache_dir = str(Path(resolved_cache_dir).resolve())
    if overwrite_run:
        run_root = Path(resolved_output_dir) / resolved_run_id
        if run_root.exists():
            shutil.rmtree(run_root)
    return Context(
        run_id=resolved_run_id,
        output_dir=resolved_output_dir,
        cache_dir=resolved_cache_dir,
    )
