# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, ClassVar, Optional

from diskcache import Cache, JSONDisk


class DiskCacheSessionMixin:
    """Mixin that provides disk-backed caching for benchmark sessions.

    Subclasses should set `CACHE_DIR` to control the cache location and
    implement the abstract hooks:
      - build_cache_key_payload() -> Dict[str, Any]
      - on_cache_hit(payload) -> bool
      - prepare_cache_payload(score) -> Dict[str, Any]
    """

    CACHE_DIR: ClassVar[str] = "./exgentic_session_cache"
    _cache_local: ClassVar[threading.local] = threading.local()
    logger: Any | None = None
    use_cache: bool = True
    _cache_hit: bool = False
    _cache_payload: Optional[dict[str, Any]] = None
    _cached_score: Optional[dict[str, Any]] = None
    _results_payload: Optional[dict[str, Any]] = None

    def _init_cache_mixin(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self._cache_hit = False
        self._cache_payload: Optional[dict[str, Any]] = None
        self._cached_score: Optional[dict[str, Any]] = None
        self._results_payload: Optional[dict[str, Any]] = None

    # --- Logging helpers -------------------------------------------------
    def _log_debug(self, message: str) -> None:
        if self.logger:
            self.logger.debug(message)

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)

    def _log_warning(self, message: str) -> None:
        if self.logger:
            self.logger.warning(message)

    # --- Cache internals -------------------------------------------------
    @classmethod
    def _cache(cls) -> Cache:
        # Use thread-local storage so each thread gets its own Cache instance.
        # diskcache supports concurrent access from separate Cache objects but
        # a single Cache (wrapping one SQLite connection) cannot be shared
        # across threads.
        attr = f"_cache_{cls.__name__}"
        cache = getattr(cls._cache_local, attr, None)
        if cache is None:
            Path(cls.CACHE_DIR).mkdir(parents=True, exist_ok=True)
            cache = Cache(cls.CACHE_DIR, disk=JSONDisk)
            setattr(cls._cache_local, attr, cache)
        return cache

    def build_cache_key_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def build_cache_key(self) -> Optional[str]:
        try:
            payload = self.build_cache_key_payload()
            return json.dumps(payload, sort_keys=True, default=str)
        except TypeError as exc:
            self._log_warning(f"Failed to serialize cache key payload: {exc}")
            return None

    def on_cache_hit(self, payload: dict[str, Any]) -> bool:
        """Hook invoked when cache payload is loaded. Return False to ignore."""
        return True

    def prepare_cache_payload(self, score: dict[str, Any]) -> dict[str, Any]:
        """Hook used to create payload to store in cache."""
        payload: dict[str, Any] = {"score": score}
        results_payload = self.get_results_payload()
        if results_payload is not None:
            payload["results"] = results_payload
        metadata = self.build_additional_cache_metadata()
        if metadata:
            payload["metadata"] = metadata
        return payload

    def build_additional_cache_metadata(self) -> dict[str, Any]:
        """Hook for subclasses to include extra metadata in the cache entry."""
        return {}

    # --- Public helpers --------------------------------------------------
    @property
    def cache_hit(self) -> bool:
        return self._cache_hit

    @property
    def cache_payload(self) -> Optional[dict[str, Any]]:
        return self._cache_payload

    @property
    def cached_score(self) -> Optional[dict[str, Any]]:
        return self._cached_score

    def set_results_payload(self, payload: Optional[dict[str, Any]]) -> None:
        self._results_payload = payload

    def get_results_payload(self) -> Optional[dict[str, Any]]:
        return self._results_payload

    def handle_cache_start(self) -> bool:
        return self.maybe_load_from_cache()

    def maybe_load_from_cache(self) -> bool:
        if not self.use_cache:
            return False
        cache_key = self.build_cache_key()
        if not cache_key:
            return False
        payload = self._cache().get(cache_key)
        if payload is None:
            self._log_debug("No cached entry for session.")
            return False
        if not self.on_cache_hit(payload):
            self._log_warning("Cache payload rejected by session hook.")
            return False
        self._set_cached_score(payload.get("score"))
        self.set_results_payload(payload.get("results"))
        self._cache_hit = True
        self._cache_payload = payload
        self._log_info(f"Reusing session results found in disk cache at {self.CACHE_DIR}")
        return True

    def write_cache_entry(self, payload: dict[str, Any]) -> None:
        if not self.use_cache:
            return
        cache_key = self.build_cache_key()
        if not cache_key:
            return
        try:
            self._cache()[cache_key] = payload
            self._cache_payload = payload
            self._log_info(f"Saving session results to cache at {self.CACHE_DIR}")
        except Exception as exc:  # pragma: no cover - defensive
            self._log_warning(f"Failed to write cache entry: {exc}")

    def cache_score(self, score: dict[str, Any]) -> None:
        self._set_cached_score(score)
        payload = self.prepare_cache_payload(score)
        self.write_cache_entry(payload)

    def restore_cache_payload(self) -> None:
        payload = self.cache_payload
        if payload:
            self.on_cache_restore(payload)

    def on_cache_restore(self, payload: dict[str, Any]) -> None:
        """Optional hook to restore artifacts from cache."""
        return

    def _set_cached_score(self, score: Optional[dict[str, Any]]) -> None:
        self._cached_score = score
