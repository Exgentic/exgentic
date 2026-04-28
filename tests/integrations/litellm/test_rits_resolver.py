# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError

import pytest
from exgentic.integrations.litellm.rits_resolver import (
    RITS_API_KEY_ENV,
    RITS_API_URL_ENV,
    build_rits_overrides,
    clear_rits_model_cache,
    get_rits_model_list,
    resolve_rits_model,
)


class _FakeResponse(BytesIO):
    def __init__(self, payload: object, status: int = 200):
        super().__init__(json.dumps(payload).encode("utf-8"))
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_rits_model_cache()
    yield
    clear_rits_model_cache()


def _patch_urlopen(monkeypatch, payload: object, calls: list[object] | None = None):
    def fake_urlopen(request, timeout):
        if calls is not None:
            calls.append((request, timeout))
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_get_rits_model_list_requires_api_key(monkeypatch):
    monkeypatch.delenv(RITS_API_KEY_ENV, raising=False)

    with pytest.raises(EnvironmentError, match=RITS_API_KEY_ENV):
        get_rits_model_list()


def test_resolve_rits_model_requires_api_url(monkeypatch):
    monkeypatch.setenv(RITS_API_KEY_ENV, "secret")  # pragma: allowlist secret
    monkeypatch.delenv(RITS_API_URL_ENV, raising=False)

    with pytest.raises(EnvironmentError, match=RITS_API_URL_ENV):
        resolve_rits_model("granite")


def test_resolve_rits_model_fetches_and_builds_api_base(monkeypatch):
    monkeypatch.setenv(RITS_API_KEY_ENV, "secret")  # pragma: allowlist secret
    monkeypatch.setenv(RITS_API_URL_ENV, "https://rits.example/base/")
    calls: list[object] = []
    _patch_urlopen(
        monkeypatch,
        [{"model_name": "granite", "endpoint": "https://rits.example/models/granite-url"}],
        calls,
    )

    model, api_base = resolve_rits_model("granite")

    assert model == "hosted_vllm/granite"
    assert api_base == "https://rits.example/base/granite-url/v1"
    request, timeout = calls[0]
    expected_api_key = "secret"  # pragma: allowlist secret
    assert any(
        key.lower().replace("_", "-") == RITS_API_KEY_ENV.lower().replace("_", "-") and value == expected_api_key
        for key, value in request.headers.items()
    )
    assert timeout > 0


def test_build_rits_overrides_strips_provider_hint_after_full_lookup_misses(monkeypatch):
    monkeypatch.setenv(RITS_API_KEY_ENV, "secret")  # pragma: allowlist secret
    monkeypatch.setenv(RITS_API_URL_ENV, "https://rits.example")
    _patch_urlopen(
        monkeypatch,
        [{"model_name": "gpt-oss-120b", "endpoint": "/serving/gpt-oss"}],
    )

    overrides = build_rits_overrides("rits/openai/gpt-oss-120b")

    assert overrides == {
        "model": "hosted_vllm/gpt-oss-120b",
        "api_base": "https://rits.example/gpt-oss/v1",
        "api_key": "secret",  # pragma: allowlist secret
        "headers": {RITS_API_KEY_ENV: "secret"},  # pragma: allowlist secret
    }


def test_get_rits_model_list_raises_clear_error_for_http_error(monkeypatch):
    monkeypatch.setenv(RITS_API_KEY_ENV, "secret")  # pragma: allowlist secret

    def fake_urlopen(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=BytesIO(b"backend down"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="503"):
        get_rits_model_list()


def test_get_rits_model_list_rejects_malformed_payload(monkeypatch):
    monkeypatch.setenv(RITS_API_KEY_ENV, "secret")  # pragma: allowlist secret
    _patch_urlopen(monkeypatch, {"model_name": "not-a-list"})

    with pytest.raises(RuntimeError, match="list"):
        get_rits_model_list()


def test_clear_rits_model_cache_refetches(monkeypatch):
    monkeypatch.setenv(RITS_API_KEY_ENV, "secret")  # pragma: allowlist secret
    calls: list[object] = []
    _patch_urlopen(
        monkeypatch,
        [{"model_name": "granite", "endpoint": "/serving/granite"}],
        calls,
    )

    assert get_rits_model_list() == {"granite": "granite"}
    assert get_rits_model_list() == {"granite": "granite"}
    assert len(calls) == 1

    clear_rits_model_cache()
    assert get_rits_model_list() == {"granite": "granite"}
    assert len(calls) == 2


def test_get_rits_model_list_cache_not_keyed_by_api_key(monkeypatch):
    calls: list[object] = []
    _patch_urlopen(
        monkeypatch,
        [{"model_name": "granite", "endpoint": "/serving/granite"}],
        calls,
    )

    monkeypatch.setenv(RITS_API_KEY_ENV, "first-key")  # pragma: allowlist secret
    assert get_rits_model_list() == {"granite": "granite"}

    monkeypatch.setenv(RITS_API_KEY_ENV, "second-key")  # pragma: allowlist secret
    assert get_rits_model_list() == {"granite": "granite"}

    # Discovery cache should be keyed by endpoint only, not secret values.
    assert len(calls) == 1
