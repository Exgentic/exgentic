# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Generic executer abstractions and a remote-process executer.

This module provides:
- BaseExecuter: minimal abstract interface for isolating a target object.
- RemoteProcessExecuter: implementation that runs the target in a child
  process and forwards attribute access and method calls via queues.

Notes:
- Serialization is done via cloudpickle to support
  dynamic classes and Pydantic models.
"""
from __future__ import annotations

import builtins
import multiprocessing as mp
import os
import traceback
import weakref
from abc import ABC, abstractmethod
from typing import Any, Optional

import cloudpickle as cp

from ...core.context import (
    context_env_scope,
    get_context,
    init_context_from_env,
    set_context,
    try_get_context,
)
from ...observers.logging import configure_warnings_logging, get_logger
from ...utils.settings import ExecuterName, get_settings


def _dumps(obj: Any) -> bytes:
    return cp.dumps(obj)


def _loads(data: Any) -> Any:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return cp.loads(data)
    return data


def _worker(queue_in: mp.Queue, queue_out: mp.Queue) -> None:
    """Worker loop that constructs the target object and executes commands."""
    configure_warnings_logging(replace_existing_file_handlers=False)
    try:
        raw = queue_in.get()
        tag, target_cls, args, kwargs, ctx = _loads(raw)
        assert tag == "init"
        if ctx is not None:
            set_context(ctx)
        else:
            # Bootstrap ContextVar from env vars inherited from parent process.
            try:
                init_context_from_env()
            except RuntimeError:
                pass  # No context env vars set — standalone worker
        obj = target_cls(*args, **kwargs)
        session_id = None
        try:
            session_id = getattr(obj, "session_id", None)
        except Exception:
            session_id = None
        if session_id:
            ctx = try_get_context()
            if ctx is not None:
                set_context(ctx.with_session(str(session_id)))
        queue_out.put(_dumps(("ready", None)))
    except Exception as e:
        queue_out.put(_dumps(("error", (type(e).__name__, str(e), traceback.format_exc()))))
        return

    while True:
        try:
            raw = queue_in.get()
            if raw is None:
                break
            cmd = _loads(raw)
            op, name, *params = cmd
            try:
                if op == "get":
                    result = getattr(obj, name)
                elif op == "set":
                    setattr(obj, name, params[0])
                    result = None
                elif op == "del":
                    delattr(obj, name)
                    result = None
                elif op == "call":
                    args, kwargs = params  # type: ignore[misc]
                    result = getattr(obj, name)(*args, **kwargs)
                elif op == "sig":
                    import inspect

                    target = getattr(obj, name)
                    if callable(target):
                        result = inspect.signature(target)
                    else:
                        result = None
                else:
                    raise ValueError(f"Unknown op: {op}")
                queue_out.put(_dumps(("ok", result)))
            except Exception as e:
                queue_out.put(_dumps(("error", (type(e).__name__, str(e), traceback.format_exc()))))
        except (EOFError, BrokenPipeError):
            break


class BaseExecuter(ABC):
    """Abstract executer interface for isolating and controlling a target."""

    @abstractmethod
    def start(self) -> None:
        """Start the isolated target (e.g., spawn a process)."""

    @abstractmethod
    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a method on the target object."""

    @abstractmethod
    def get(self, name: str) -> Any:
        """Get an attribute value from the target object."""

    @abstractmethod
    def set(self, name: str, value: Any) -> None:
        """Set an attribute value on the target object."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Delete an attribute from the target object."""

    @abstractmethod
    def shutdown(self) -> None:
        """Terminate the isolation mechanism (e.g., kill the process)."""

    def get_proxy(self) -> Any:
        """Optional: return a proxy object that forwards to this executer."""
        raise NotImplementedError


class RemoteProcessExecuter(BaseExecuter):
    """Executer that runs a target class instance in a child process.

    Usage:
        exe = RemoteProcessExecuter(MyClass, arg1, kw=2)
        exe.start()
        exe.call("do_work", 123)
        value = exe.get("status")
        exe.set("flag", True)
        exe.shutdown()
    """

    def __init__(self, target_cls: type, *args: Any, **kwargs: Any) -> None:
        self._target_cls = target_cls
        self._args = args
        self._kwargs = kwargs
        self._ctx = mp.get_context("spawn")
        self._qin: Optional[mp.Queue] = None
        self._qout: Optional[mp.Queue] = None
        self._proc: Optional[mp.Process] = None
        self._finalizer = weakref.finalize(self, RemoteProcessExecuter._cleanup_static, self)

        # Set up logging
        from ...utils.paths import RunPaths

        run_paths = RunPaths.from_context(get_context())
        run_paths.run_dir.mkdir(parents=True, exist_ok=True)
        logger_name = f"{__name__}.RemoteProcessExecuter.{run_paths.run_id}"
        self._logger = get_logger(logger_name, str(run_paths.executer_log))

    # --- lifecycle ---
    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._logger.info(f"Starting worker process for {self._target_cls.__name__}")
        qin = self._ctx.Queue()
        qout = self._ctx.Queue()
        proc = self._ctx.Process(target=_worker, args=(qin, qout), daemon=True)
        with context_env_scope():
            proc.start()
        self._qin, self._qout, self._proc = qin, qout, proc
        self._logger.info(f"Worker process started PID:{proc.pid}")
        # send init
        ctx = try_get_context()
        self._qin.put(_dumps(("init", self._target_cls, self._args, self._kwargs, ctx)))
        status, data = self._recv()
        if status == "error":
            self.shutdown()
            exc_type, exc_msg, _ = data
            self._logger.error(f"Worker init failed: {exc_type}: {exc_msg}")
            raise RuntimeError(f"Worker init failed: {exc_type}: {exc_msg}")
        self._logger.info("Worker process initialized successfully")

    def shutdown(self) -> None:
        self._logger.info("Shutting down worker process")
        try:
            if self._qin is not None:
                self._qin.put(None)
        except Exception:
            pass
        try:
            if self._proc is not None:
                self._logger.info(f"Joining worker process PID:{self._proc.pid}")
                self._proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._proc is not None and self._proc.is_alive():
                self._logger.warning(f"Terminating unresponsive worker process PID:{self._proc.pid}")
                self._proc.terminate()
        except Exception:
            pass
        self._qin = None
        self._qout = None
        self._proc = None
        try:
            self._finalizer.detach()
        except Exception:
            pass
        self._logger.info("Worker process shutdown complete")

    @staticmethod
    def _cleanup_static(self_ref: RemoteProcessExecuter) -> None:  # pragma: no cover
        try:
            if self_ref._qin is not None:
                self_ref._qin.put(None)
        except Exception:
            pass
        try:
            if self_ref._proc is not None:
                self_ref._proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self_ref._proc is not None and self_ref._proc.is_alive():
                self_ref._proc.terminate()
        except Exception:
            pass

    # --- RPC helpers ---
    def _send(self, cmd: tuple[Any, ...]) -> None:
        if self._proc is None or self._qin is None or self._qout is None or not self._proc.is_alive():
            raise RuntimeError("Worker process is not running")
        self._qin.put(_dumps(cmd))

    def _recv(self) -> tuple[str, Any]:
        assert self._qout is not None
        # Check if process is alive before waiting
        if not self._proc.is_alive():
            exitcode = self._proc.exitcode
            self._logger.error(f"Worker process died before response (exit code: {exitcode})")
            raise RuntimeError(f"Worker process died before response (exit code: {exitcode})")
        raw = self._qout.get()
        return _loads(raw)

    # --- BaseExecuter API ---
    def _recreate_original_exception(self, exc_type_name: str, exc_msg: str, exc_traceback: str) -> Exception:
        """Recreate the original exception type with the original message."""
        # Try to get the original exception class
        exc_class = getattr(builtins, exc_type_name, None)
        if exc_class is None:
            # Try common exception types
            import_map = {
                "FileNotFoundError": FileNotFoundError,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "AttributeError": AttributeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "RuntimeError": RuntimeError,
            }
            exc_class = import_map.get(exc_type_name, Exception)
        try:
            original_exc = exc_class(exc_msg)
        except Exception:
            original_exc = Exception(f"{exc_type_name}: {exc_msg}")
        original_exc.__remote_traceback__ = exc_traceback
        original_exc.__remote_worker__ = True
        return original_exc

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self._logger.debug(f"Calling {name}({args}, {kwargs})")
        self._send(("call", name, args, kwargs))
        status, data = self._recv()
        if status == "error":
            exc_type, exc_msg, exc_traceback = data
            self._logger.error(f"Worker method {name}() failed: {exc_type}: {exc_msg}")
            original_exc = self._recreate_original_exception(exc_type, exc_msg, exc_traceback)
            raise original_exc
        return data

    def get(self, name: str) -> Any:
        self._send(("get", name))
        status, data = self._recv()
        if status == "error":
            exc_type, exc_msg, exc_traceback = data
            self._logger.error(f"Worker get {name} failed: {exc_type}: {exc_msg}")
            original_exc = self._recreate_original_exception(exc_type, exc_msg, exc_traceback)
            raise original_exc
        return data

    def set(self, name: str, value: Any) -> None:
        self._send(("set", name, value))
        status, data = self._recv()
        if status == "error":
            exc_type, exc_msg, exc_traceback = data
            self._logger.error(f"Worker set {name} failed: {exc_type}: {exc_msg}")
            original_exc = self._recreate_original_exception(exc_type, exc_msg, exc_traceback)
            raise original_exc

    def delete(self, name: str) -> None:
        self._send(("del", name))
        status, data = self._recv()
        if status == "error":
            exc_type, exc_msg, exc_traceback = data
            self._logger.error(f"Worker delete {name} failed: {exc_type}: {exc_msg}")
            original_exc = self._recreate_original_exception(exc_type, exc_msg, exc_traceback)
            raise original_exc

    # --- Introspection helpers ---
    def signature(self, name: str):
        self._send(("sig", name))
        status, data = self._recv()
        if status == "error":
            exc_type, exc_msg, _ = data
            raise RuntimeError(f"{exc_type}: {exc_msg}")
        return data

    # --- Proxy convenience ---
    def get_proxy(self) -> Any:  # type: ignore[override]
        exe = self

        class _Proxy:
            def __init__(self) -> None:
                object.__setattr__(self, "_exe", exe)

            def __getattr__(self, name: str):
                # Try to fetch a real signature for methods
                try:
                    sig = self._exe.signature(name)
                except RuntimeError:
                    sig = None
                if sig is not None:

                    def _method(*args, **kwargs):
                        return self._exe.call(name, *args, **kwargs)

                    _method.__name__ = name  # type: ignore[attr-defined]
                    _method.__signature__ = sig  # type: ignore[attr-defined]
                    return _method
                # Not callable; return attribute value
                return self._exe.get(name)

            def __setattr__(self, name: str, value: Any) -> None:
                if name.startswith("_"):
                    object.__setattr__(self, name, value)
                else:
                    self._exe.set(name, value)

            def __delattr__(self, name: str) -> None:
                if name.startswith("_"):
                    object.__delattr__(self, name)
                else:
                    self._exe.delete(name)

            def shutdown(self) -> None:
                self._exe.shutdown()

        return _Proxy()


class InProcessExecuter(BaseExecuter):
    """Executer that instantiates the target locally (no isolation).

    Provides the same interface so callers can swap executors without
    changing call sites. `get_proxy()` returns the underlying object.
    """

    def __init__(self, target_cls: type, *args: Any, **kwargs: Any) -> None:
        self._target = target_cls(*args, **kwargs)

    def start(self) -> None:  # No-op for in-process
        return None

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._target, name)(*args, **kwargs)

    def get(self, name: str) -> Any:
        return getattr(self._target, name)

    def set(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def delete(self, name: str) -> None:
        delattr(self._target, name)

    def shutdown(self) -> None:  # No-op by default
        return None

    def get_proxy(self) -> Any:  # type: ignore[override]
        return self._target


def make_executer(kind: Optional[ExecuterName], target_cls: type, *args: Any, **kwargs: Any) -> BaseExecuter:
    if kind is None:
        kind = get_settings().default_executer
    if kind == "inprocess":
        return InProcessExecuter(target_cls, *args, **kwargs)
    if kind == "remote_process":
        # Ensure child process inherits context via env vars.
        ctx = get_context()
        os.environ.update(ctx.to_env())
        exe = RemoteProcessExecuter(target_cls, *args, **kwargs)
        exe.start()
        return exe
    raise ValueError(f"Unknown executer kind: {kind}")


if __name__ == "__main__":
    # Simple self-test for the executer and proxy
    class Calculator:
        def __init__(self, start: int = 0) -> None:
            self.value = start

        def add(self, a: int, b: int = 0) -> int:
            self.value += a + b
            return self.value

        def mul(self, k: int) -> int:
            self.value *= k
            return self.value

        def boom(self) -> None:
            raise ValueError("kaboom")

    exe = RemoteProcessExecuter(Calculator, 10)
    exe.start()
    calc = exe.get_proxy()

    print("Initial value:", calc.value)
    print("Signature of add:", exe.signature("add"))
    print("Add(2, b=3):", calc.add(2, b=3))
    print("Mul(4):", calc.mul(4))
    print("Current value:", calc.value)
    calc.value = 7
    print("Set value=7, now:", calc.value)
    try:
        calc.boom()
    except Exception as e:
        print("Caught error from remote:", type(e).__name__, e)
    finally:
        calc.shutdown()
