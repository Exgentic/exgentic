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


# Env vars forwarded to subprocess runners (venv, docker).  We use a
# pattern-based allowlist so we don't leak the host shell's entire
# environment (IDE vars, macOS/Claude-specific vars, SSH sockets, etc.)
# into the child, while still covering the long tail of LiteLLM
# providers without enumerating all ~100+ of them by prefix.
# Exgentic's own settings travel through runtime.json, not env vars.
#
# Suffix patterns catch virtually all provider credentials
# (``OPENAI_API_KEY``, ``FIREWORKS_API_KEY``, ``ANTHROPIC_BASE_URL``, …).
_FORWARD_SUFFIXES = (
    "_API_KEY",
    "_API_BASE",
    "_API_VERSION",
    "_API_TYPE",
    "_BASE_URL",
    "_ACCESS_KEY",
    "_SECRET_KEY",
    "_ACCESS_TOKEN",
    "_AUTH_TOKEN",
    "_SESSION_TOKEN",
    "_TOKEN",
)
# Prefix patterns catch multi-variable providers that use keys without
# the above suffixes (region names, credential file paths, model IDs).
_FORWARD_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GEMINI_",
    "OPENAI_",
    "ANTHROPIC_",
    "VERTEX_",
    "VERTEXAI_",
    "WATSONX_",
    "HF_",
    "HUGGINGFACE_",
    "LITELLM_",
    "OTEL_",
)


def prepare_subprocess_env() -> dict[str, str]:
    """Build a filtered env dict for subprocess runners (venv, docker).

    Forwards only model-provider credentials, base URLs, and OTEL
    configuration (see :data:`_FORWARD_SUFFIXES` and
    :data:`_FORWARD_PREFIXES`).  Exgentic context and settings travel
    via ``runtime.json`` (see :func:`inject_exgentic_env`), so no
    ``EXGENTIC_*`` vars need to be forwarded here.
    """
    import os

    return {k: v for k, v in os.environ.items() if k.endswith(_FORWARD_SUFFIXES) or k.startswith(_FORWARD_PREFIXES)}


def inject_exgentic_env(env: dict[str, str], role: Role | None = None) -> None:
    """Point *env* at a per-service ``runtime.json``.

    When *role* is provided, writes a fresh ``runtime.json`` for that
    service's role and points the child at it via
    ``EXGENTIC_RUNTIME_FILE``.  When *role* is ``None`` the child
    inherits whatever ``EXGENTIC_RUNTIME_FILE`` is set in the current
    process (used by sub-services like litellm proxies that share their
    parent's role).

    The child bootstraps context, settings, and OTEL state from the
    runtime file via :func:`init_context` — no individual ``EXGENTIC_*``
    env vars are needed.

    Mutates *env* in-place.
    """
    from ...core.context import get_runtime_env, save_service_runtime

    if role is not None:
        runtime_path = save_service_runtime(role)
        env["EXGENTIC_RUNTIME_FILE"] = str(runtime_path)
    else:
        for k, v in get_runtime_env().items():
            env[k] = v


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
