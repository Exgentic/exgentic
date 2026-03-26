# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

from exgentic.integrations.litellm import LitellmProxy


def test_proxy_writes_exgentic_trace_callback_to_litellm_settings_config(tmp_path, monkeypatch) -> None:
    """Verify that proxy config is created without explicit callbacks.

    Callbacks are now auto-registered via configure_litellm() during settings
    initialization to prevent duplicates. This test verifies the config file
    doesn't contain explicit callback registrations.
    """
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
        # Verify callbacks are NOT in the config (they're registered via configure_litellm)
        assert "success_callback" not in config_data.get("litellm_settings", {})
        assert "failure_callback" not in config_data.get("litellm_settings", {})
