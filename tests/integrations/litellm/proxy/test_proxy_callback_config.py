# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

from exgentic.integrations.litellm import LitellmProxy


def test_proxy_writes_exgentic_trace_callback_to_litellm_settings_config(tmp_path, monkeypatch) -> None:
    import exgentic.integrations.litellm.proxy as proxy_mod

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
        return _DummyProc()

    monkeypatch.setattr(proxy_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(proxy_mod, "_is_port_open", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(proxy_mod, "_is_proxy_ready", lambda *_args, **_kwargs: True)

    log_path = tmp_path / "litellm.log"

    with LitellmProxy(
        model="openai/gpt-4o-mini",
        port=49999,
        log_path=str(log_path),
        startup_timeout=1.0,
    ):
        config_path = log_path.with_name("litellm_config.json")
        assert config_path.exists()
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        callback = "exgentic.integrations.litellm.trace_logger.trace_logger"
        async_callback = "exgentic.integrations.litellm.trace_logger.async_trace_logger"
        assert set(config_data["litellm_settings"]["success_callback"]) == {
            callback,
            async_callback,
        }
        assert set(config_data["litellm_settings"]["failure_callback"]) == {
            callback,
            async_callback,
        }
