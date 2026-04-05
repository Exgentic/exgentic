# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared utilities for runner implementations."""

from __future__ import annotations

import base64
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.context import Role


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


def inject_exgentic_env(env: dict[str, str], role: Role | None = None) -> None:
    """Point *env* at a per-service ``runtime.json`` and propagate settings.

    When *role* is provided, this writes a fresh ``runtime.json`` for that
    service's role at the per-service path
    (``sessions/{session_id}/{role}/runtime.json``) and points the child at
    it via ``EXGENTIC_RUNTIME_FILE``.  When *role* is ``None`` the child
    inherits whatever ``EXGENTIC_RUNTIME_FILE`` is set in the current
    process (used by sub-services like litellm proxies that share their
    parent's role).

    Mutates *env* in-place.
    """
    from ...core.context import get_runtime_env, save_service_runtime
    from ...environment.instance import get_manager
    from ...utils.settings import get_settings

    if role is not None:
        # Role transition — write a fresh per-service runtime.json.
        runtime_path = save_service_runtime(role)
        env["EXGENTIC_RUNTIME_FILE"] = str(runtime_path)
    else:
        # No role transition — inherit the parent's runtime file.
        for k, v in get_runtime_env().items():
            env[k] = v

    settings = get_settings()
    # Propagate all EXGENTIC_* settings (otel_enabled, log_level, etc.)
    # to the subprocess so it inherits the parent's configuration.
    for k, v in settings.get_env().items():
        env.setdefault(k, v)

    # Use the EnvironmentManager's base_dir (~/.exgentic/) so that
    # EXGENTIC_CACHE_DIR points to the same location where benchmark
    # data is actually installed.  The old settings.cache_dir default
    # (".exgentic") resolved to a CWD-relative path that diverged from
    # the manager's absolute ~/.exgentic/ path, breaking Docker mounts.
    manager = get_manager()
    env["EXGENTIC_CACHE_DIR"] = str(manager.base_dir)
    env["EXGENTIC_OUTPUT_DIR"] = str(Path(settings.output_dir).resolve())


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
