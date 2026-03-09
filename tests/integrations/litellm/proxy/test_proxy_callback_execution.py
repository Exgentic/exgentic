# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from exgentic.integrations.litellm import LitellmProxy


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_proxy(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=1.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("LiteLLM proxy did not become ready")


@pytest.mark.skipif(
    not (Path(sys.executable).parent / "litellm").exists(),
    reason="litellm CLI not installed in active venv",
)
def test_exgentic_proxy_executes_default_trace_callback_and_writes_trace(tmp_path, fake_openai_server) -> None:
    backend_port = fake_openai_server.server_address[1]
    backend_base = f"http://127.0.0.1:{backend_port}/v1"
    trace_path = tmp_path / "trace.jsonl"

    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_BASE": backend_base,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
            "EXGENTIC_LLM_LOG_FILE": str(trace_path),
        }
    )

    with LitellmProxy(
        model="openai/gpt-4o-mini",
        env=env,
        startup_timeout=10.0,
    ) as proxy:
        response = requests.post(
            f"{proxy.base_url}/v1/chat/completions",
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            timeout=10,
        )
        response.raise_for_status()

    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert rows and rows[0]["status"] == "success"


@pytest.mark.skipif(
    not (Path(sys.executable).parent / "litellm").exists(),
    reason="litellm CLI not installed in active venv",
)
def test_raw_litellm_proxy_executes_litellm_settings_callbacks_and_writes_trace(tmp_path, fake_openai_server) -> None:
    backend_port = fake_openai_server.server_address[1]
    backend_base = f"http://127.0.0.1:{backend_port}/v1"
    proxy_port = _free_port()
    trace_path = tmp_path / "trace.jsonl"
    config_path = tmp_path / "litellm_config.json"

    callback_path = "exgentic.integrations.litellm.trace_logger.trace_logger"
    async_callback_path = "exgentic.integrations.litellm.trace_logger.async_trace_logger"
    config_path.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "openai/gpt-4o-mini",
                        "litellm_params": {"model": "openai/gpt-4o-mini"},
                    }
                ],
                "litellm_settings": {
                    "success_callback": [callback_path, async_callback_path],
                    "failure_callback": [callback_path, async_callback_path],
                },
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_BASE": backend_base,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
            "EXGENTIC_LLM_LOG_FILE": str(trace_path),
        }
    )
    repo_root = Path(__file__).resolve().parents[4]
    src_path = repo_root / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(src_path),
            str(repo_root),
            env.get("PYTHONPATH", ""),
        ]
    )

    proc = subprocess.Popen(
        [
            str(Path(sys.executable).parent / "litellm"),
            "--config",
            str(config_path),
            "--port",
            str(proxy_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        _wait_for_proxy(f"http://127.0.0.1:{proxy_port}/v1/models")
        response = requests.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            timeout=10,
        )
        response.raise_for_status()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert rows and rows[0]["status"] == "success"
