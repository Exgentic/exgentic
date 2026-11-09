# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from diskcache_community import Cache, Disk
from exgentic.integrations.litellm.cache import build_litellm_cache
from exgentic.utils.settings import ExgenticSettings

planted_calls = []


def planted():
    planted_calls.append(True)


class Planted:
    def __reduce__(self):
        return (planted, ())


def test_litellm_cache_skips_planted_pickle(tmp_path) -> None:
    cache = build_litellm_cache(ExgenticSettings(cache_dir=str(tmp_path), litellm_cache_dir="litellm"))
    kwargs = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
    with Cache(cache.cache.disk_cache.directory, disk=Disk) as other_writer:
        other_writer[cache.get_cache_key(**kwargs)] = Planted()

    assert cache.get_cache(**kwargs) is None
    assert planted_calls == []
