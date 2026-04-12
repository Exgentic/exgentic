# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Lightweight helper to launch a single-model LiteLLM proxy for the lifetime of an object.

Usage (expects OPENAI_API_BASE/OPENAI_API_KEY already set for your backend):
    from . import LitellmProxy

    with LitellmProxy(model="openai/gpt-4o-mini") as proxy:
        os.environ["OPENAI_API_BASE"] = proxy.base_url
        # run your Codex/Exgentic flow that expects an OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from ...core.context import try_get_context
from ...core.types.model_settings import ModelSettings
from ...utils.paths import get_session_paths
from ...utils.settings import get_settings
from .trace_logger import (
    DEFAULT_FILE as DEFAULT_USAGE_FILE,
)

TRACE_CALLBACK = "exgentic.integrations.litellm.trace_logger.trace_logger"
# TODO: see config.py — single callback is sufficient on litellm>=1.82.
# ASYNC_TRACE_CALLBACK = "exgentic.integrations.litellm.trace_logger.async_trace_logger"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


class LitellmProxy:
    """Launch a LiteLLM proxy serving a single model.

    The proxy process is started on creation (or __enter__) and torn down on close/__exit__.
    """

    def __init__(
        self,
        model: str,
        *,
        port: int | None = None,
        model_alias_map: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        log_path: str | None = None,
        usage_log_path: str | None = None,
        startup_timeout: float = 15.0,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.model = model
        self.port = port or _get_free_port()
        self.model_alias_map = model_alias_map or {}
        self._env_overrides = env or {}
        self._log_path = log_path
        self.usage_log_path = usage_log_path
        self.startup_timeout = startup_timeout
        self.model_settings = model_settings or ModelSettings()
        self._log_file = None
        self._proc: subprocess.Popen[str] | None = None
        self._config_file: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        env = self._build_env()
        self._set_trace_log_env(env, self._resolve_usage_log_path(), self._resolve_session_root())
        self._config_file = self._write_config_file()
        config_dir = str(self._config_file.parent)
        env["PYTHONPATH"] = os.pathsep.join([config_dir, env.get("PYTHONPATH", "")])
        cmd = self._build_command(self._config_file)
        stdout, stderr = self._build_stdio()

        self._proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env,
        )

        # Wait briefly for the proxy to bind; fail fast if it exits.
        self._wait_for_startup(stdout=stdout)

    def _build_command(self, config_path: Path) -> list[str]:
        cmd = ["litellm", "--port", str(self.port)]
        cmd.extend(["--config", str(config_path)])
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._env_overrides)

        repo_root = Path(__file__).resolve().parents[4]
        src_path = repo_root / "src"
        extra_paths: list[str] = [str(src_path), str(repo_root)]

        from ...core.context import get_runtime_env

        settings = get_settings()
        for k, v in get_runtime_env().items():
            env.setdefault(k, v)

        existing = {key.lower() for key in env}
        for key, value in settings.get_env().items():
            if key.lower() in existing:
                continue
            env[key] = value

        # Keep proxy cache bootstrap behavior enabled by default for subprocess startup.
        env["EXGENTIC_PROXY_CACHE_INIT"] = "true"

        env["PYTHONPATH"] = os.pathsep.join([*extra_paths, env.get("PYTHONPATH", "")])
        return env

    def _resolve_usage_log_path(self) -> Path:
        if self.usage_log_path:
            return Path(self.usage_log_path)
        if self._log_path:
            return Path(self._log_path).with_name("trace.jsonl")
        return Path(DEFAULT_USAGE_FILE)

    def _resolve_session_root(self) -> Path | None:
        ctx = try_get_context()
        if ctx is not None:
            session_id = ctx.session_id
            if session_id is not None:
                return get_session_paths(session_id).root
        return None

    @staticmethod
    def _set_trace_log_env(env: dict[str, str], usage_path: Path, session_root: Path | None) -> None:
        from .trace_logger import FILE_ENV

        env.setdefault(FILE_ENV, str(usage_path))

    def _build_litellm_params(self) -> dict[str, object]:
        litellm_params: dict[str, object] = {"model": self.model}
        if self.model_settings.temperature is not None:
            litellm_params["temperature"] = self.model_settings.temperature
        if self.model_settings.max_tokens is not None:
            litellm_params["max_tokens"] = self.model_settings.max_tokens
        if self.model_settings.top_p is not None:
            litellm_params["top_p"] = self.model_settings.top_p
        return litellm_params

    def _build_config_data(self) -> dict[str, object]:
        trace_cb = TRACE_CALLBACK
        config_data: dict[str, object] = {
            "model_list": [
                {
                    "model_name": self.model,
                    "litellm_params": self._build_litellm_params(),
                }
            ],
            "litellm_settings": {
                "success_callback": [trace_cb],
                "failure_callback": [trace_cb],
                # Route /v1/messages through chat/completions so
                # non-Anthropic backends can serve them.
                "use_chat_completions_url_for_anthropic_messages": True,
            },
        }
        router_settings: dict[str, object] = {}
        if self.model_alias_map:
            router_settings["model_group_alias"] = self.model_alias_map
        if self.model_settings.num_retries is not None:
            router_settings["num_retries"] = self.model_settings.num_retries
        router_settings["retry_after"] = self.model_settings.retry_after
        if router_settings:
            config_data["router_settings"] = router_settings
        return config_data

    def _write_config_file(self) -> Path:
        if self._log_path:
            cfg_path = Path(self._log_path).with_name("litellm_config.json")
        else:
            cfg_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".json").name)
        config_data = self._build_config_data()
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)
        return cfg_path

    def _build_stdio(self):
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
        if self._log_path:
            self._log_file = open(self._log_path, "w", encoding="utf-8")
            stdout = self._log_file
            stderr = subprocess.STDOUT
        return stdout, stderr

    def _wait_for_startup(self, *, stdout) -> None:
        deadline = time.time() + self.startup_timeout
        last_err: str | None = None
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                # Process exited; capture any output for debugging.
                out, err = ("", "")
                if stdout is subprocess.PIPE:
                    out, err = self._proc.communicate(timeout=0.5)
                if self._log_path and os.path.exists(self._log_path):
                    with open(
                        self._log_path,
                        encoding="utf-8-sig",
                        errors="replace",
                        newline="",
                    ) as lf:
                        last_err = lf.read()
                raise RuntimeError(f"LiteLLM proxy exited early: {err or out or last_err or 'no output'}")
            if _is_port_open("127.0.0.1", self.port):
                if _is_proxy_ready("127.0.0.1", self.port):
                    return
            time.sleep(0.05)

        raise RuntimeError(f"LiteLLM proxy did not open port {self.port} within timeout")

    def close(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:
                # fall through to kill below
                pass
            if proc.poll() is None:
                proc.kill()
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._config_file = None

    def __enter__(self) -> LitellmProxy:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except Exception:
        return False


def _probe_http_status(host: str, port: int, path: str) -> int | None:
    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            return getattr(resp, "status", 200)
    except urllib.error.HTTPError as err:
        return err.code
    except Exception:
        return None


def _is_proxy_ready(host: str, port: int) -> bool:
    status = _probe_http_status(host, port, "/health/liveliness")
    if status is not None and 200 <= status < 300:
        return True
    status = _probe_http_status(host, port, "/v1/models")
    if status is None:
        return False
    if 200 <= status < 300:
        return True
    if status in (401, 403):
        # Auth-required, but proxy is up.
        return True
    return False
