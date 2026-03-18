# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""ThreadTransport — runs the object in a dedicated thread, queue-based RPC."""

from __future__ import annotations

import contextvars
import queue
import threading
from typing import Any

from .transport import ObjectHost, Transport, deserialize_error, serialize_error

_SHUTDOWN = object()


class ThreadTransport(Transport):
    """Runs the target object in a dedicated daemon thread.

    Communication happens via two queues (request / response).
    The object's methods never block the caller's thread except
    while waiting for the result.
    """

    def __init__(self, target_cls: type, *args: Any, **kwargs: Any) -> None:
        self._target_cls = target_cls
        self._args = args
        self._kwargs = kwargs
        self._req: queue.Queue = queue.Queue()
        self._resp: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        ctx = contextvars.copy_context()
        self._thread = threading.Thread(target=ctx.run, args=(self._worker,), daemon=True)
        self._thread.start()
        status, payload = self._resp.get()
        if status == "error":
            raise deserialize_error(payload)

    # ── worker loop ──────────────────────────────────────────────────

    def _worker(self) -> None:
        try:
            obj = self._target_cls(*self._args, **self._kwargs)
        except Exception as exc:
            self._resp.put(("error", serialize_error(exc)))
            return

        host = ObjectHost(obj)
        self._resp.put(("ready", None))

        while True:
            msg = self._req.get()
            if msg is _SHUTDOWN:
                break
            op, name, args, kwargs = msg
            try:
                result = host.handle(op, name, *args, **kwargs)
                self._resp.put(("ok", result))
            except Exception as exc:
                self._resp.put(("error", serialize_error(exc)))

    # ── Transport API ────────────────────────────────────────────────

    def _rpc(self, op: str, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("Worker thread is not running")
        self._req.put((op, name, args, kwargs))
        status, payload = self._resp.get()
        if status == "error":
            raise deserialize_error(payload)
        return payload

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._rpc("call", method, *args, **kwargs)

    def get(self, name: str) -> Any:
        return self._rpc("get", name)

    def set(self, name: str, value: Any) -> None:
        self._rpc("set", name, value)

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._req.put(_SHUTDOWN)
            self._thread.join(timeout=5.0)
        self._thread = None

    def __repr__(self) -> str:
        return f"ThreadTransport({self._target_cls.__name__})"
