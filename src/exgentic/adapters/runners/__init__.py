# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Runner & Transport abstractions for running objects in different isolation levels.

Runners wrap any object and control where it executes:

- ``direct``  — same thread, no isolation
- ``thread``  — separate thread, queue-based communication
- ``process`` — separate process, pipe-based communication with cloudpickle
- ``service`` — HTTP service in a background thread
- ``docker``  — HTTP service inside a Docker container
- ``venv``    — HTTP service in an isolated uv virtual environment

Usage::

    calc = with_runner(Calculator, runner="thread", value=10)
"""

from __future__ import annotations

from typing import Any, Literal

from .direct import DirectTransport
from .transport import ObjectHost, ObjectProxy, Transport

RunnerName = Literal["direct", "thread", "process", "service", "docker", "venv"]


def _resolve_cls(cls: type | str) -> type:
    """Resolve a ``"module:qualname"`` string to the actual class."""
    if isinstance(cls, type):
        return cls
    module_path, qualname = cls.rsplit(":", 1)
    import importlib

    mod = importlib.import_module(module_path)
    obj = mod
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj  # type: ignore[return-value]


def with_runner(cls: type | str, *args: Any, runner: RunnerName = "direct", **kwargs: Any) -> Any:
    """Create an instance of *cls* running in the specified isolation level.

    *cls* may be a class or a ``"module:qualname"`` string.  String
    references are resolved lazily — for ``venv`` and ``docker`` runners
    the string is forwarded directly so heavy imports never happen on the
    host.

    Returns an ``ObjectProxy`` that transparently forwards all
    attribute access and method calls to the real object.
    """
    if runner == "direct":
        cls = _resolve_cls(cls)
        return ObjectProxy(DirectTransport(cls(*args, **kwargs)))

    if runner == "thread":
        from .thread import ThreadTransport

        cls = _resolve_cls(cls)
        t = ThreadTransport(cls, *args, **kwargs)
        t.start()
        return ObjectProxy(t)

    if runner == "process":
        from .process import PipeTransport

        cls = _resolve_cls(cls)
        t = PipeTransport(cls, *args, **kwargs)
        t.start()
        return ObjectProxy(t)

    if runner == "service":
        from .service import ServiceRunner

        cls = _resolve_cls(cls)
        return ServiceRunner(cls, *args, **kwargs).start()

    if runner == "docker":
        from .docker import DockerRunner

        docker_kw = {}
        for key in (
            "env_name",
            "module_path",
            "image",
            "dockerfile",
            "port",
            "docker_args",
            "dependencies",
            "docker_socket",
            "volumes",
        ):
            if key in kwargs:
                docker_kw[key] = kwargs.pop(key)
        return DockerRunner(cls, *args, **docker_kw, **kwargs).start()

    if runner == "venv":
        from .venv import VenvRunner

        venv_kw = {}
        for key in (
            "env_name",
            "module_path",
            "port",
            "dependencies",
            "health_timeout",
        ):
            if key in kwargs:
                venv_kw[key] = kwargs.pop(key)
        return VenvRunner(cls, *args, **venv_kw, **kwargs).start()

    raise ValueError(f"Unknown runner: {runner!r}")


__all__ = [
    "DirectTransport",
    "ObjectHost",
    "ObjectProxy",
    "RunnerName",
    "Transport",
    "with_runner",
]
