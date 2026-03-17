# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""DirectTransport — no isolation, calls the object in the same thread."""

from __future__ import annotations

from typing import Any

from .transport import ObjectHost, Transport


class DirectTransport(Transport):
    """Calls the object directly in the same thread and process.

    Useful as a baseline and as the default runner.
    """

    def __init__(self, obj: Any) -> None:
        self._host = ObjectHost(obj)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._host.handle("call", method, *args, **kwargs)

    def get(self, name: str) -> Any:
        return self._host.handle("get", name)

    def set(self, name: str, value: Any) -> None:
        self._host.handle("set", name, value)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"DirectTransport({self._host.obj!r})"
