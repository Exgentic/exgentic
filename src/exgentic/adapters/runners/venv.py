# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""VenvRunner — runs the HTTP service inside a uv virtual environment.

Uses the same HTTPTransport as ServiceRunner and DockerRunner, but the
uvicorn server runs in a subprocess with its own isolated venv instead
of the host Python or a Docker container.

The venv is created and managed by the EnvironmentManager — VenvRunner
only starts the subprocess and optionally installs extra runtime
dependencies.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.context import Role

from ._utils import (
    find_free_port,
    inject_exgentic_env,
    make_close,
    prepare_subprocess_env,
    serialize_kwargs,
)
from .service import HTTPTransport, _wait_for_health
from .transport import ObjectProxy

_HEALTH_TIMEOUT = 120.0
_TRANSPORT_TIMEOUT = 600.0


def _uv(*args: str, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("uv CLI not found on PATH")
    result = subprocess.run([uv_bin, *args], check=False, **kwargs)
    if check and result.returncode != 0:
        stderr = getattr(result, "stderr", "") or ""
        stdout = getattr(result, "stdout", "") or ""
        raise RuntimeError(f"uv {' '.join(args[:3])} failed (exit {result.returncode}):\n{stderr}\n{stdout}")
    return result


class VenvRunner:
    """Start an HTTP service in an isolated uv venv and return an ObjectProxy.

    Parameters
    ----------
    target_cls:       Class to instantiate inside the venv subprocess.
    env_name:         Environment name for EnvironmentManager (e.g. "benchmarks/bfcl").
    module_path:      Dotted module path for locating package resources.
    port:             Host port to bind (auto-selected if None).
    dependencies:     Extra pip packages to install in the venv at runtime.
    health_timeout:   Seconds to wait for the health endpoint.
    """

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        env_name: str = "",
        module_path: str = "",
        port: int | None = None,
        dependencies: list[str] | None = None,
        health_timeout: float | None = None,
        role: Role | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "VenvRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._env_name = env_name
        self._module_path = module_path
        self._port = port or find_free_port()
        self._dependencies = dependencies or []
        self._health_timeout = health_timeout or _HEALTH_TIMEOUT
        self._role = role
        self._process: subprocess.Popen | None = None

    # ── venv handling ─────────────────────────────────────────────────

    def _get_venv_dir(self) -> Path:
        """Return the venv directory managed by EnvironmentManager."""
        from ...environment.instance import get_manager

        return get_manager().env_path(self._env_name) / "venv"

    def _venv_python(self) -> Path:
        """Return the path to the Python binary inside the venv."""
        venv = self._get_venv_dir()
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    def _ensure_venv(self) -> Path:
        """Ensure the venv exists via EnvironmentManager."""
        from ...environment import EnvType
        from ...environment.helpers import get_exgentic_install_target
        from ...environment.instance import get_manager

        mgr = get_manager()
        project_root, packages = get_exgentic_install_target()
        mgr.install(
            self._env_name,
            env_type=EnvType.VENV,
            module_path=self._module_path,
            project_root=project_root,
            packages=packages,
        )
        return self._get_venv_dir()

    def _install_deps(self) -> None:
        """Install extra runtime dependencies into the venv."""
        if not self._dependencies:
            return
        python = self._venv_python()
        _uv(
            "pip",
            "install",
            "--python",
            str(python),
            "--no-cache",
            *self._dependencies,
            capture_output=True,
            text=True,
        )

    # ── subprocess lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        venv = self._ensure_venv()
        self._install_deps()

        if isinstance(self._target_cls, str):
            cls_ref = self._target_cls
        else:
            cls_ref = f"{self._target_cls.__module__}:{self._target_cls.__qualname__}"
        kwargs_flag, kwargs_value = serialize_kwargs(self._kwargs)

        # Build a filtered environment (same filtering as DockerRunner).
        env = prepare_subprocess_env()
        env["VIRTUAL_ENV"] = str(venv)
        venv_bin = str(venv / "bin")
        # Prepend venv bin to the *system* PATH so external tools (docker,
        # podman, git, …) remain reachable from within the venv subprocess.
        system_path = os.environ.get("PATH", "")
        env["PATH"] = venv_bin + os.pathsep + system_path
        inject_exgentic_env(env, role=self._role)

        exgentic_bin = self._get_venv_dir() / "bin" / "exgentic"
        cmd = [
            str(exgentic_bin),
            "serve",
            "--cls",
            cls_ref,
            kwargs_flag,
            kwargs_value,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        atexit.register(self._stop_process)

        url = f"http://127.0.0.1:{self._port}"
        try:
            _wait_for_health(url, timeout=self._health_timeout)
        except TimeoutError:
            proc = self._process
            if proc is not None:
                proc.terminate()
                stdout, stderr = proc.communicate(timeout=5)
            else:
                stdout, stderr = b"", b""
            self._stop_process()
            raise TimeoutError(
                f"Venv service did not become healthy within {self._health_timeout}s.\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            ) from None

        transport = HTTPTransport(url, timeout=_TRANSPORT_TIMEOUT)
        proxy = ObjectProxy(transport)
        object.__setattr__(proxy, "close", make_close(transport, self._stop_process))
        return proxy

    def _stop_process(self) -> None:
        if self._process is None:
            return
        proc = self._process
        self._process = None
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
