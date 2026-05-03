# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json

from exgentic.integrations.litellm import LitellmProxy


class _DummyProc:
    def poll(self):
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        return None


def _patch_subprocess(monkeypatch) -> None:
    import exgentic.integrations.litellm.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod.subprocess, "Popen", lambda *_a, **_k: _DummyProc())
    monkeypatch.setattr(proxy_mod, "_is_port_open", lambda *_a, **_k: True)
    monkeypatch.setattr(proxy_mod, "_is_proxy_ready", lambda *_a, **_k: True)


def test_litellm_params_extra_merged_into_proxy_config(tmp_path, monkeypatch) -> None:
    _patch_subprocess(monkeypatch)
    log_path = tmp_path / "litellm.log"

    extras = {
        "api_base": "https://example.invalid/v1",
        "api_key": "secret",  # pragma: allowlist secret
        "extra_headers": {"X-Backend-Auth": "secret"},  # pragma: allowlist secret
    }

    with LitellmProxy(
        model="openai/gpt-4o-mini",
        port=49997,
        log_path=str(log_path),
        startup_timeout=1.0,
        litellm_params_extra=extras,
    ):
        config_path = log_path.with_name("litellm_config.json")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        params = config_data["model_list"][0]["litellm_params"]

        assert params["api_base"] == extras["api_base"]
        assert params["api_key"] == extras["api_key"]
        assert params["extra_headers"] == extras["extra_headers"]
        # default ``model`` is preserved when extras don't override it
        assert params["model"] == "openai/gpt-4o-mini"


def test_litellm_params_extra_can_override_underlying_model(tmp_path, monkeypatch) -> None:
    _patch_subprocess(monkeypatch)
    log_path = tmp_path / "litellm.log"

    with LitellmProxy(
        model="my-alias",
        port=49996,
        log_path=str(log_path),
        startup_timeout=1.0,
        litellm_params_extra={"model": "hosted_vllm/some-model"},
    ):
        config_path = log_path.with_name("litellm_config.json")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        entry = config_data["model_list"][0]

        # client-facing alias is unchanged …
        assert entry["model_name"] == "my-alias"
        # … but the underlying litellm_params.model is overridden
        assert entry["litellm_params"]["model"] == "hosted_vllm/some-model"


def test_litellm_params_extra_default_does_not_change_existing_config(tmp_path, monkeypatch) -> None:
    _patch_subprocess(monkeypatch)
    log_path = tmp_path / "litellm.log"

    with LitellmProxy(
        model="openai/gpt-4o-mini",
        port=49995,
        log_path=str(log_path),
        startup_timeout=1.0,
    ):
        config_path = log_path.with_name("litellm_config.json")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        params = config_data["model_list"][0]["litellm_params"]

        assert params["model"] == "openai/gpt-4o-mini"
        assert "api_base" not in params
        assert "api_key" not in params
        assert "extra_headers" not in params
