# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Integration test for proxy subprocess callback registration.

This test verifies that trace callbacks work end-to-end when the litellm proxy
runs as an actual subprocess, catching issues like those in PR #61 where
callback registration silently broke in the subprocess.

Background:
- PR #61 broke trace logging in subprocess while passing all in-process tests
- Existing tests in test_proxy_callback_execution.py call configure_litellm() directly
- None verify the full subprocess startup -> callback registration -> trace write path
- This test fills that gap by running the proxy as a real subprocess
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import requests
from exgentic.integrations.litellm import LitellmProxy


@pytest.mark.skipif(
    not (Path(sys.executable).parent / "litellm").exists(),
    reason="litellm CLI not installed in active venv",
)
def test_proxy_subprocess_writes_trace_end_to_end(tmp_path, fake_openai_server) -> None:
    """Integration test: verify trace callbacks work in actual proxy subprocess.

    This test catches regressions where callback registration breaks in the subprocess
    (like PR #61) while passing in-process tests.

    Test flow:
    1. LitellmProxy writes config with trace callbacks to disk
    2. Proxy subprocess starts via Popen and loads the config
    3. Callbacks are registered in the subprocess (not in-process)
    4. Request flows through the subprocess proxy
    5. Trace callback executes in subprocess and writes to trace.jsonl
    6. Verify trace file exists and contains expected data

    This is the ONLY test that verifies the full subprocess callback path.
    """
    backend_port = fake_openai_server.server_address[1]
    backend_base = f"http://127.0.0.1:{backend_port}/v1"
    trace_path = tmp_path / "trace.jsonl"
    log_path = tmp_path / "proxy.log"

    import os

    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_BASE": backend_base,
            "OPENAI_API_KEY": "test-key",  # pragma: allowlist secret
            "EXGENTIC_LLM_LOG_FILE": str(trace_path),
        }
    )

    # Start proxy as actual subprocess (not mocked)
    with LitellmProxy(
        model="openai/gpt-4o-mini",
        env=env,
        log_path=str(log_path),
        startup_timeout=15.0,
    ) as proxy:
        # Verify config was written with callbacks
        config_path = log_path.with_name("litellm_config.json")
        assert config_path.exists(), "Config file should be written"

        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "litellm_settings" in config_data
        assert "success_callback" in config_data["litellm_settings"]

        expected_callbacks = {
            "exgentic.integrations.litellm.trace_logger.trace_logger",
            "exgentic.integrations.litellm.trace_logger.async_trace_logger",
        }
        actual_callbacks = set(config_data["litellm_settings"]["success_callback"])
        assert actual_callbacks == expected_callbacks

        # Verify proxy is running as subprocess (not in-process)
        assert proxy._proc is not None
        assert proxy._proc.poll() is None

        # Send request through subprocess proxy
        response = requests.post(
            f"{proxy.base_url}/v1/chat/completions",
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Test message"}],
            },
            timeout=10,
        )
        response.raise_for_status()
        response_data = response.json()

        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        assert response_data["choices"][0]["message"]["content"] == "ok"

        # Give subprocess time to write trace (callbacks are async)
        time.sleep(0.5)

    # Verify trace was written by subprocess
    assert trace_path.exists(), f"Trace file should exist at {trace_path}"

    trace_lines = trace_path.read_text().strip().splitlines()
    assert len(trace_lines) > 0, "Trace file should have at least one entry"

    trace_entry = json.loads(trace_lines[0])
    assert trace_entry["status"] == "success"
    assert trace_entry["model"] == "openai/gpt-4o-mini"
    assert trace_entry["prompt_tokens"] == 1
    assert trace_entry["completion_tokens"] == 1
    assert trace_entry["total_tokens"] == 2
    assert "request" in trace_entry
    assert "response" in trace_entry
    assert "timestamp" in trace_entry
