# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any

import pytest
from exgentic.integrations.litellm.cache import CustomCache


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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_async_cache_key_reuse_for_call_id() -> None:
    cache = CustomCache(type="local")
    base_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "call-1",
    }
    assert await cache.async_get_cache(**base_kwargs) is None

    mutated_kwargs = dict(base_kwargs)
    mutated_kwargs["messages"] = [{"role": "user", "content": "different"}]
    response = _build_response()
    await cache.async_add_cache(response, **mutated_kwargs)

    followup_kwargs = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "call-2",
    }
    assert await cache.async_get_cache(**followup_kwargs) is not None
