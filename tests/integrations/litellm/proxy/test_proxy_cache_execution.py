# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import requests
from exgentic.integrations.litellm import LitellmProxy


@pytest.mark.skipif(
    not (Path(sys.executable).parent / "litellm").exists(),
    reason="litellm CLI not installed in active venv",
)
def test_proxy_reuses_cached_completion_response(tmp_path, fake_openai_server) -> None:
    fake_openai_server.RequestHandlerClass.request_count = 0
    backend_port = fake_openai_server.server_address[1]
    backend_base = f"http://127.0.0.1:{backend_port}/v1"

    venv_bin = Path(sys.executable).parent
    env = {
        "OPENAI_API_BASE": backend_base,
        "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
        "EXGENTIC_CACHE_DIR": str(tmp_path / "cache"),
        "EXGENTIC_litellm_cache_dir": ".litellm_cache",
        "EXGENTIC_litellm_caching": "true",
        "EXGENTIC_LOG_LEVEL": "DEBUG",
        "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.0,
    }

    with LitellmProxy(
        model="openai/gpt-4o-mini",
        env=env,
        startup_timeout=10.0,
    ) as proxy:
        url = f"{proxy.base_url}/v1/chat/completions"
        first = requests.post(url, json=payload, timeout=10)
        first.raise_for_status()
        second = requests.post(url, json=payload, timeout=10)
        second.raise_for_status()

    assert fake_openai_server.RequestHandlerClass.request_count == 1
