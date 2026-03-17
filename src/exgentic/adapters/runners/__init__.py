# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Runner & Transport abstractions for running objects in different isolation levels.

Runners wrap any object and control where it executes:

- ``direct``  — same thread, no isolation
- ``thread``  — separate thread, queue-based communication
- ``process`` — separate process, pipe-based communication with cloudpickle
- ``service`` — HTTP service in a background thread
- ``docker``  — HTTP service inside a Docker container

Usage::

    calc = with_runner(Calculator, runner="thread", value=10)
"""

from __future__ import annotations

from typing import Any, Literal

from .direct import DirectTransport
from .transport import ObjectHost, ObjectProxy, Transport

RunnerName = Literal["direct", "thread", "process", "service", "docker"]


def with_runner(cls: type, *args: Any, runner: RunnerName = "direct", **kwargs: Any) -> Any:
    """Create an instance of *cls* running in the specified isolation level.

    Returns an ``ObjectProxy`` that transparently forwards all
    attribute access and method calls to the real object.
    """
    if runner == "direct":
        return ObjectProxy(DirectTransport(cls(*args, **kwargs)))

    if runner == "thread":
        from .thread import ThreadTransport

        t = ThreadTransport(cls, *args, **kwargs)
        t.start()
        return ObjectProxy(t)

    if runner == "process":
        from .process import PipeTransport

        t = PipeTransport(cls, *args, **kwargs)
        t.start()
        return ObjectProxy(t)

    if runner == "service":
        from .service import ServiceRunner

        return ServiceRunner(cls, *args, **kwargs).start()

    if runner == "docker":
        from .docker import DockerRunner

        docker_kw = {}
        for key in (
            "image",
            "dockerfile",
            "port",
            "docker_args",
            "dependencies",
            "setup_script",
            "docker_socket",
            "volumes",
        ):
            if key in kwargs:
                docker_kw[key] = kwargs.pop(key)
        return DockerRunner(cls, *args, **docker_kw, **kwargs).start()

    raise ValueError(f"Unknown runner: {runner!r}")


__all__ = [
    "RunnerName",
    "Transport",
    "ObjectHost",
    "ObjectProxy",
    "DirectTransport",
    "with_runner",
]
