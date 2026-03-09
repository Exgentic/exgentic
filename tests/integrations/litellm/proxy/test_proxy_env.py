# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from exgentic.core.context import Context, set_context
from exgentic.integrations.litellm import LitellmProxy


def test_proxy_passes_context_env(monkeypatch):
    captured = {}

    class _DummyProc:
        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return 0

        def kill(self) -> None:
            return None

    def _fake_popen(*args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _DummyProc()

    monkeypatch.setattr("exgentic.integrations.litellm.proxy.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("exgentic.integrations.litellm.proxy._is_port_open", lambda *_a, **_k: True)
    monkeypatch.setattr("exgentic.integrations.litellm.proxy._is_proxy_ready", lambda *_a, **_k: True)

    ctx = Context(run_id="run-proxy", output_dir="/tmp/out", cache_dir="/tmp/cache")
    set_context(ctx)

    with LitellmProxy(model="openai/gpt-4o-mini", port=49998, startup_timeout=1.0):
        env = captured["env"]
        assert env["EXGENTIC_CTX_RUN_ID"] == "run-proxy"
