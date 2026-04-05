# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared utilities for runner implementations."""

from __future__ import annotations

import base64
import json
import socket
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    """Return the project root directory.

    Walks up from the exgentic package looking for a ``pyproject.toml``.
    When none is found (e.g. ``uv tool install exgentic``), falls back
    to ``~/.exgentic/`` so that benchmark venvs and caches still have a
    stable home directory.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    fallback = Path.home() / ".exgentic"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def find_free_port() -> int:
    """Return an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def serialize_kwargs(kwargs: dict[str, Any]) -> tuple[str, str]:
    """Serialize kwargs for the ``exgentic serve`` CLI.

    Returns ``(flag, value)`` — either ``("--kwargs", json_str)``
    or ``("--kwargs-b64", pickled_b64)`` for non-JSON-serializable values.
    """
    try:
        return "--kwargs", json.dumps(kwargs)
    except TypeError:
        import cloudpickle as cp

        return "--kwargs-b64", base64.b64encode(cp.dumps(kwargs)).decode("ascii")


_SYSTEM_ENV_BLOCKLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "HOSTNAME",
        "LANG",
        "TERM",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "TMPDIR",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
    }
)
_PREFIX_BLOCKLIST = ("VSCODE_", "UV_", "PIP_")


def prepare_subprocess_env() -> dict[str, str]:
    """Build a filtered env dict for subprocess runners (venv, docker).

    Forwards API tokens and user config while excluding system-level
    vars, IDE noise, and Python-path-manager prefixes that could
    conflict with the isolated environment.
    """
    import os

    root = find_project_root()
    project_root = str(root) if (root / "pyproject.toml").exists() else ""

    env: dict[str, str] = {
        k: v
        for k, v in os.environ.items()
        if k not in _SYSTEM_ENV_BLOCKLIST
        and not any(k.startswith(p) for p in _PREFIX_BLOCKLIST)
        and not v.startswith(project_root + "/src/")
    }
    return env


def inject_exgentic_env(env: dict[str, str]) -> None:
    """Add exgentic context vars and resolved settings paths into *env*.

    Mutates *env* in-place.
    """
    from ...core.context import context_env
    from ...environment.instance import get_manager
    from ...utils.settings import get_settings

    for k, v in context_env().items():
        env[k] = v
    for key in ("EXGENTIC_CTX_OUTPUT_DIR", "EXGENTIC_CTX_CACHE_DIR"):
        if key in env:
            env[key] = str(Path(env[key]).resolve())

    settings = get_settings()
    # Use the EnvironmentManager's base_dir (~/.exgentic/) so that
    # EXGENTIC_CACHE_DIR points to the same location where benchmark
    # data is actually installed.  The old settings.cache_dir default
    # (".exgentic") resolved to a CWD-relative path that diverged from
    # the manager's absolute ~/.exgentic/ path, breaking Docker mounts.
    manager = get_manager()
    env.setdefault("EXGENTIC_CACHE_DIR", str(manager.base_dir))
    env.setdefault("EXGENTIC_OUTPUT_DIR", str(Path(settings.output_dir).resolve()))


def make_close(transport: Any, stop_fn: Any) -> Any:
    """Create a close function for an ObjectProxy.

    Attempts a graceful ``close()`` on the remote object, then shuts
    down the transport and calls *stop_fn* to tear down the underlying
    process/container.
    """

    def _close() -> None:
        try:
            transport.call("close")
        except Exception:
            pass
        transport.close()
        stop_fn()

    return _close
