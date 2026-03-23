# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""VenvRunner — runs the HTTP service inside a uv virtual environment.

Uses the same HTTPTransport as ServiceRunner and DockerRunner, but the
uvicorn server runs in a subprocess with its own isolated venv instead
of the host Python or a Docker container.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ._utils import (
    find_free_port,
    find_project_root,
    inject_exgentic_env,
    make_close,
    prepare_subprocess_env,
    serialize_kwargs,
)
from .service import HTTPTransport, _wait_for_health
from .transport import ObjectProxy

_HEALTH_TIMEOUT = 30.0
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
    venv_dir:         Directory to create the virtual environment in.
                      If None, uses ``{cache_dir}/venv/``.
    port:             Host port to bind (auto-selected if None).
    dependencies:     Extra pip packages to install in the venv.
    setup_script:     Path to a shell script to run after venv creation.
    requirements_txt: Path to a requirements.txt to install in the venv.
    """

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        venv_dir: str | None = None,
        port: int | None = None,
        dependencies: list[str] | None = None,
        setup_script: str | None = None,
        requirements_txt: str | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "VenvRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._venv_dir = Path(venv_dir) if venv_dir else None
        self._port = port or find_free_port()
        self._dependencies = dependencies or []
        self._setup_script = setup_script
        self._requirements_txt = requirements_txt
        self._process: subprocess.Popen | None = None

    # ── venv handling ─────────────────────────────────────────────────

    def _get_venv_dir(self) -> Path:
        """Return the venv directory, defaulting to cache_dir/venv/.

        Always returns an absolute path so subprocess commands work
        regardless of the current working directory.
        """
        if self._venv_dir is not None:
            return Path(self._venv_dir).resolve()
        from ...utils.settings import get_settings

        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        return Path(get_settings().cache_dir).expanduser().resolve() / f"venv-py{pyver}"

    def _venv_python(self) -> Path:
        """Return the path to the Python binary inside the venv."""
        venv = self._get_venv_dir()
        if sys.platform == "win32":
            return venv / "Scripts" / "python.exe"
        return venv / "bin" / "python"

    def _ensure_venv(self) -> Path:
        """Create the venv if it doesn't exist, install exgentic into it."""
        venv = self._get_venv_dir()
        python = self._venv_python()

        if python.exists():
            return venv

        venv.mkdir(parents=True, exist_ok=True)

        _uv(
            "venv",
            str(venv),
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            capture_output=True,
            text=True,
        )

        root = find_project_root()
        _uv("pip", "install", "--python", str(python), "--no-cache", str(root), capture_output=True, text=True)

        return venv

    def _install_deps(self) -> None:
        """Install requirements.txt and extra dependencies into the venv."""
        python = self._venv_python()

        if self._requirements_txt:
            req_path = Path(self._requirements_txt)
            if req_path.exists():
                env = os.environ.copy()
                env["GIT_LFS_SKIP_SMUDGE"] = "1"
                _uv(
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--no-cache",
                    "-r",
                    str(req_path),
                    capture_output=True,
                    text=True,
                    env=env,
                )

        if self._dependencies:
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

    def _run_setup_script(self) -> None:
        """Run setup.sh with the venv activated."""
        if not self._setup_script:
            return
        script_path = Path(self._setup_script)
        if not script_path.exists():
            raise FileNotFoundError(f"Setup script not found: {self._setup_script}")

        venv = self._get_venv_dir()
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(venv)
        venv_bin = str(venv / "bin")
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        from ...utils.settings import get_settings

        env["EXGENTIC_CACHE_DIR"] = str(Path(get_settings().cache_dir).resolve())

        subprocess.run(["bash", str(script_path)], check=True, env=env)

    # ── subprocess lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        venv = self._ensure_venv()
        self._install_deps()
        self._run_setup_script()

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
        inject_exgentic_env(env)

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
            _wait_for_health(url, timeout=_HEALTH_TIMEOUT)
        except TimeoutError:
            proc = self._process
            if proc is not None:
                proc.terminate()
                stdout, stderr = proc.communicate(timeout=5)
            else:
                stdout, stderr = b"", b""
            self._stop_process()
            raise TimeoutError(
                f"Venv service did not become healthy within {_HEALTH_TIMEOUT}s.\n"
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
