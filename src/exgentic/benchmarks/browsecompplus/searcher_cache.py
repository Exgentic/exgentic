# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import hashlib
from typing import Any, Dict

from ...utils.disk_cache import DiskCacheSessionMixin


class SearchDiskCacheSession(DiskCacheSessionMixin):
    CACHE_DIR = "./exgentic_session_cache/browsecomp/searcher"  # single DB for all runs

    def __init__(
        self,
        query: str,
        n: int,
        k: int,
        search_type: str,
        search_model: str,
        normalize: bool = False,
        use_cache: bool = True,
    ):
        self.query = query
        self.n = n
        self.k = k
        self.search_type = search_type
        self.search_model = search_model
        self.normalize = normalize
        self._init_cache_mixin(use_cache=use_cache)

    def build_cache_key_payload(self) -> dict:
        return {
            "search_type": self.search_type,
            "search_model": self.search_model,
            "normalize": self.normalize,
            "n": self.n,
            "k": self.k,
            "q": hashlib.sha256(self.query.encode("utf-8")).hexdigest(),
        }

    def build_additional_cache_metadata(self) -> dict:
        return {
            "search_type": self.search_type,
            "search_model": self.search_model,
            "normalize": self.normalize,
        }

    def on_cache_hit(self, payload: Dict[str, Any]) -> bool:
        results = payload.get("results")
        if not isinstance(results, dict):
            return False
        result = results.get("raw")
        if result is None or not isinstance(result, str):
            return False
        return True

    def cache_results(self, result: str):
        self.set_results_payload({"raw": result})
        self.cache_score({})

    def handle_start_fetch_results(self):
        if self.handle_cache_start():
            cache_results = self.get_results_payload()
            if cache_results:
                try:
                    return cache_results["raw"]
                except Exception:
                    return None
        return None
