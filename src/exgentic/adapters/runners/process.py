# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""PipeTransport — runs the object in a subprocess via multiprocessing + cloudpickle."""

from __future__ import annotations

import multiprocessing as mp
import weakref
from typing import Any

import cloudpickle as cp

from .transport import ObjectHost, Transport, deserialize_error, serialize_error

# ── worker process ───────────────────────────────────────────────────


def _worker(q_in: mp.Queue, q_out: mp.Queue) -> None:
    """Subprocess entry point: create the object and serve RPC requests."""
    # Late imports — these run in the child process.
    from ...core.context import init_context_from_env, set_context, try_get_context
    from ...observers.logging import configure_warnings_logging

    configure_warnings_logging(replace_existing_file_handlers=False)

    try:
        tag, target_cls, args, kwargs, ctx = cp.loads(q_in.get())
        assert tag == "init"

        # Restore context in child process.
        if ctx is not None:
            set_context(ctx)
        else:
            try:
                init_context_from_env()
            except RuntimeError:
                pass  # No context env vars — standalone worker

        obj = target_cls(*args, **kwargs)

        # If the object has a session_id, update context to include it.
        session_id = getattr(obj, "session_id", None)
        if session_id:
            current_ctx = try_get_context()
            if current_ctx is not None:
                set_context(current_ctx.with_session(str(session_id)))

        q_out.put(cp.dumps(("ready", None)))
    except Exception as exc:
        q_out.put(cp.dumps(("error", serialize_error(exc))))
        return

    host = ObjectHost(obj)
    while True:
        try:
            raw = q_in.get()
            if raw is None:  # shutdown sentinel
                break
            op, name, args, kwargs = cp.loads(raw)
            try:
                result = host.handle(op, name, *args, **kwargs)
                q_out.put(cp.dumps(("ok", result)))
            except Exception as exc:
                q_out.put(cp.dumps(("error", serialize_error(exc))))
        except (EOFError, BrokenPipeError):
            break


# ── transport ────────────────────────────────────────────────────────


class PipeTransport(Transport):
    """Runs the target in a child process with full memory isolation.

    Uses cloudpickle for serialization and multiprocessing queues
    for communication. Propagates the exgentic Context to the child.
    """

    def __init__(self, target_cls: type, *args: Any, **kwargs: Any) -> None:
        self._target_cls = target_cls
        self._args = args
        self._kwargs = kwargs
        self._ctx = mp.get_context("spawn")
        self._q_in: mp.Queue | None = None
        self._q_out: mp.Queue | None = None
        self._proc: mp.Process | None = None

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return

        from ...core.context import context_env_scope, try_get_context

        self._q_in = self._ctx.Queue()
        self._q_out = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker,
            args=(self._q_in, self._q_out),
            daemon=True,
        )
        # Ensure context env vars are in os.environ for the spawned process.
        with context_env_scope():
            self._proc.start()
        self._finalizer = weakref.finalize(self, _terminate, self._q_in, self._proc)

        # Send init payload with context.
        ctx = try_get_context()
        self._q_in.put(cp.dumps(("init", self._target_cls, self._args, self._kwargs, ctx)))
        status, payload = self._recv()
        if status == "error":
            self.close()
            raise deserialize_error(payload)

    # ── internal helpers ─────────────────────────────────────────────

    def _recv(self) -> tuple[str, Any]:
        assert self._q_out is not None
        if self._proc is not None and not self._proc.is_alive():
            raise RuntimeError(f"Worker process died (exit code: {self._proc.exitcode})")
        return cp.loads(self._q_out.get())

    def _rpc(self, op: str, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._proc is None or self._q_in is None or not self._proc.is_alive():
            raise RuntimeError("Worker process is not running")
        self._q_in.put(cp.dumps((op, name, args, kwargs)))
        status, payload = self._recv()
        if status == "error":
            raise deserialize_error(payload)
        return payload

    # ── Transport API ────────────────────────────────────────────────

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._rpc("call", method, *args, **kwargs)

    def get(self, name: str) -> Any:
        return self._rpc("get", name)

    def set(self, name: str, value: Any) -> None:
        self._rpc("set", name, value)

    def close(self) -> None:
        _terminate(self._q_in, self._proc)
        self._q_in = None
        self._q_out = None
        self._proc = None
        try:
            self._finalizer.detach()
        except Exception:
            pass

    def __repr__(self) -> str:
        pid = self._proc.pid if self._proc else None
        return f"PipeTransport({self._target_cls.__name__}, pid={pid})"


def _terminate(q_in: mp.Queue | None, proc: mp.Process | None) -> None:
    """Shut down the worker process (used by both close() and the weak finalizer)."""
    try:
        if q_in is not None:
            q_in.put(None)
    except Exception:
        pass
    try:
        if proc is not None:
            proc.join(timeout=2.0)
    except Exception:
        pass
    try:
        if proc is not None and proc.is_alive():
            proc.terminate()
    except Exception:
        pass
