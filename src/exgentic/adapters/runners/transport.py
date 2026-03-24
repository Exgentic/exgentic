# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Core transport abstractions: Transport, ObjectHost, ObjectProxy, and error helpers."""

from __future__ import annotations

import builtins
import inspect
import traceback
from abc import ABC, abstractmethod
from typing import Any

import cloudpickle as cp

# Sentinel returned by ``get`` when the attribute is a bound method.
# The proxy checks for this to avoid serialising the entire instance.
CALLABLE_MARKER = {"__exgentic_callable__": True}

# ── Transport interface ──────────────────────────────────────────────


class Transport(ABC):
    """Communication channel between a proxy and a remote object.

    Every transport implements four operations so that ``ObjectProxy``
    can forward attribute access and method calls regardless of where
    the real object lives.
    """

    @abstractmethod
    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def get(self, name: str) -> Any:
        ...

    @abstractmethod
    def set(self, name: str, value: Any) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# ── ObjectHost — server side ─────────────────────────────────────────


class ObjectHost:
    """Executes operations on a real object (the "server side").

    Used identically whether the object lives in the same thread,
    a subprocess, or an HTTP server.
    """

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def handle(self, op: str, name: str, *args: Any, **kwargs: Any) -> Any:
        if op == "call":
            return getattr(self.obj, name)(*args, **kwargs)
        if op == "get":
            value = getattr(self.obj, name)
            # Bound methods cannot be reliably serialised (the instance may
            # contain locks, threads, etc.).  Return a lightweight marker so
            # the proxy knows to use ``call`` instead.
            if inspect.ismethod(value) or inspect.isbuiltin(value):
                return CALLABLE_MARKER
            return value
        if op == "set":
            setattr(self.obj, name, args[0])
            return None
        if op == "del":
            delattr(self.obj, name)
            return None
        raise ValueError(f"Unknown operation: {op!r}")


# ── Error serialization ──────────────────────────────────────────────


def serialize_error(exc: BaseException) -> dict:
    """Serialize an exception into a dict that can cross process/network boundaries.

    The dict always contains string fallbacks (``type``, ``msg``, ``tb``).
    When possible it also includes a ``pickled`` copy of the original
    exception so that custom exception types and attributes survive.
    """
    pickled = None
    try:
        pickled = cp.dumps(exc)
    except Exception:
        pass
    return {
        "type": type(exc).__qualname__,
        "msg": str(exc),
        "tb": traceback.format_exc(),
        "pickled": pickled,
    }


def deserialize_error(data: dict) -> BaseException:
    """Reconstruct an exception from a ``serialize_error`` dict.

    Strategy: try cloudpickle first (preserves custom types and state),
    then fall back to reconstructing a builtin type from its name.
    A ``__remote_traceback__`` attribute is always attached.
    """
    tb = data.get("tb", "")

    # Fast path: unpickle the original exception.
    pickled = data.get("pickled")
    if pickled is not None:
        try:
            exc = cp.loads(pickled)
            if isinstance(exc, BaseException):
                exc.__remote_traceback__ = tb  # type: ignore[attr-defined]
                return exc
        except Exception:
            pass

    # Fallback: reconstruct from type name (builtins only) + message.
    name = data.get("type", "RuntimeError")
    msg = data.get("msg", "")
    cls = getattr(builtins, name, None)
    if not (isinstance(cls, type) and issubclass(cls, BaseException)):
        cls = RuntimeError
    try:
        exc = cls(msg)
    except TypeError:
        exc = RuntimeError(f"{name}: {msg}")
    exc.__remote_traceback__ = tb  # type: ignore[attr-defined]
    return exc


# ── ObjectProxy — client side ────────────────────────────────────────


class ObjectProxy:
    """Transparent proxy that forwards attribute access over a Transport.

    Behaves like the real object: attribute reads, writes, and method
    calls are all forwarded through the transport.
    """

    def __init__(self, transport: Transport) -> None:
        object.__setattr__(self, "_transport", transport)

    def __getattr__(self, name: str) -> Any:
        transport: Transport = object.__getattribute__(self, "_transport")
        value = transport.get(name)
        if isinstance(value, dict) and value.get("__exgentic_callable__"):

            def method(*args: Any, **kwargs: Any) -> Any:
                return transport.call(name, *args, **kwargs)

            method.__name__ = name  # type: ignore[attr-defined]
            return method
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            transport: Transport = object.__getattribute__(self, "_transport")
            transport.set(name, value)

    def close(self) -> None:
        """Close the remote object, then tear down the transport."""
        transport: Transport = object.__getattribute__(self, "_transport")
        try:
            transport.call("close")
        except AttributeError:
            pass
        transport.close()

    def __del__(self) -> None:
        try:
            transport: Transport = object.__getattribute__(self, "_transport")
            transport.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        transport: Transport = object.__getattribute__(self, "_transport")
        return f"ObjectProxy({transport!r})"
