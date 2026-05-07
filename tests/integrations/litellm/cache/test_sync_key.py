# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

from typing import Any

from framework.integrations.litellm.cache import CustomCache


def _build_response() -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }


def test_sync_cache_key_reuse_for_call_id() -> None:
    cache = CustomCache(type="local")
    base_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "call-1",
    }
    assert cache.get_cache(**base_kwargs) is None

    mutated_kwargs = dict(base_kwargs)
    mutated_kwargs["messages"] = [{"role": "user", "content": "different"}]
    cache.add_cache(_build_response(), **mutated_kwargs)

    followup_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "call-2",
    }
    assert cache.get_cache(**followup_kwargs) is not None


def test_sync_cache_without_call_id_uses_computed_key() -> None:
    cache = CustomCache(type="local")
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert cache.get_cache(**kwargs) is None
    cache.add_cache(_build_response(), **kwargs)
    assert cache.get_cache(**kwargs) is not None


def test_sync_cache_key_reuse_scoped_to_call_id() -> None:
    cache = CustomCache(type="local")
    base_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "call-1",
    }
    assert cache.get_cache(**base_kwargs) is None

    mutated_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "different"}],
        "litellm_call_id": "call-2",
    }
    cache.add_cache(_build_response(), **mutated_kwargs)

    followup_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "call-3",
    }
    assert cache.get_cache(**followup_kwargs) is None
