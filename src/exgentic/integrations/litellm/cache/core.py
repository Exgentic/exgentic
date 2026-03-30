# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Litellm cache implementation using deterministic keying."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from litellm.caching.caching import Cache
from litellm.main import ModelResponse

from ....utils.settings import resolve_cache_path
from .key import (
    _NOT_HANDLED,
    CacheKeyBuilder,
    MessageNormalizer,
    _hash_text,
    _resolve_custom_param,
)
from .log import CacheLogger

# ---------------------------------------------------------------------------
# Cache key store (sync/async boundary bookkeeping)
# ---------------------------------------------------------------------------


class CacheKeyStore:
    """Remembers cache keys so add_cache can reuse the key from get_cache.

    Litellm's callback pipeline may lose/regenerate keys between the get and
    add phases, especially in async streaming.
    """

    def __init__(self) -> None:
        self._sync: dict[str, str] = {}
        self._async: dict[str, str] = {}

    @staticmethod
    def _valid(call_id: Any) -> bool:
        return isinstance(call_id, str)

    # -- sync --

    def remember_sync(self, call_id: Any, key: Optional[str]) -> None:
        if self._valid(call_id) and key:
            self._sync[call_id] = key

    def inject_sync_key(self, kwargs: dict[str, Any]) -> None:
        call_id = kwargs.get("litellm_call_id")
        if not self._valid(call_id) or "cache_key" in kwargs:
            return
        stored = self._sync.pop(call_id, None)
        if stored is not None:
            kwargs["cache_key"] = stored

    # -- async --

    def remember_async(self, call_id: Any, key: Optional[str]) -> None:
        if self._valid(call_id) and key:
            self._async[call_id] = key

    def pop_async(self, call_id: Any) -> Optional[str]:
        return self._async.pop(call_id, None) if self._valid(call_id) else None

    def clear_async(self, call_id: Any) -> None:
        if self._valid(call_id):
            self._async.pop(call_id, None)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _is_empty_response(result: Any) -> bool:
    """True if result is None or a ModelResponse with no content/tool_calls."""
    if result is None:
        return True
    if not isinstance(result, ModelResponse):
        return False
    msg = result.choices[0].message
    return msg.content is None and not msg.tool_calls and not msg.function_call


def _extract_max_age(kwargs: dict[str, Any]) -> float:
    cc = kwargs.get("cache", {})
    return cc.get("s-maxage") or cc.get("s-max-age") or float("inf")


def _classify_miss(cached_result: Any, max_age: float) -> tuple[str, Optional[int]]:
    if isinstance(cached_result, dict) and "timestamp" in cached_result and max_age is not None:
        age = time.time() - cached_result["timestamp"]
        if age > max_age:
            return "expired", int(age)
    return "not_found", None


def _dump_raw_key(raw_key: Optional[str]) -> Optional[str]:
    if raw_key is None:
        return None
    return json.dumps(raw_key, ensure_ascii=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class LLMCache(Cache):
    """Disk-backed LLM response cache with deterministic keys and diagnostics."""

    def __init__(self, *, delete_time_from_messages: bool = False, **kwargs: Any) -> None:
        self._normalizer = MessageNormalizer(strip_time=delete_time_from_messages)
        self._key_store = CacheKeyStore()
        self._log = CacheLogger(
            disk_cache_dir=kwargs.get("disk_cache_dir"),
            strip_time=delete_time_from_messages,
        )
        super().__init__(**kwargs)
        # Must be created after super().__init__ so self is a valid Cache instance.
        self._key_builder = CacheKeyBuilder(self._normalizer, self)

    @property
    def logger(self) -> Any:
        return self._log.raw

    # -- litellm Cache interface --

    def get_cache_key(self, **kwargs: Any) -> str:  # type: ignore[override]
        key, _, _, _ = self._key_builder.build(kwargs)
        return key or ""

    def get_cache(self, dynamic_cache_object: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        if self.should_use_cache(**kwargs) is not True:
            self._log.detail("skip", reason="disabled", model=kwargs.get("model"))
            return None

        key, raw_key, source, _ = self._build_key_logged(kwargs)
        max_age = _extract_max_age(kwargs)
        self._key_store.remember_sync(kwargs.get("litellm_call_id"), key)

        if key is None:
            self._log_miss(reason="cache_key_none", source=source, kwargs=kwargs)
            return None

        backend = dynamic_cache_object if dynamic_cache_object is not None else self.cache
        cached_result = backend.get_cache(key, **kwargs)
        return self._finalize_lookup(cached_result, key, source, raw_key, kwargs, max_age)

    async def async_get_cache(self, dynamic_cache_object: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        if self.should_use_cache(**kwargs) is not True:
            self._log.detail("skip", reason="disabled", model=kwargs.get("model"))
            return None

        key, raw_key, source, _ = self._build_key_logged(kwargs)
        max_age = _extract_max_age(kwargs)
        self._key_store.remember_async(kwargs.get("litellm_call_id"), key)

        if key is None:
            self._log_miss(reason="cache_key_none", source=source, kwargs=kwargs)
            return None

        if dynamic_cache_object is not None:
            cached_result = await dynamic_cache_object.async_get_cache(key, **kwargs)
        else:
            cached_result = await self.cache.async_get_cache(key, **kwargs)
        return self._finalize_lookup(cached_result, key, source, raw_key, kwargs, max_age)

    def add_cache(self, result: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log.detail(
            "add_cache_called",
            result_type=type(result).__name__,
            model=kwargs.get("model"),
        )
        if _is_empty_response(result):
            self._log.detail("skip", reason="empty_assistant_content", model=kwargs.get("model"))
            return
        self._key_store.inject_sync_key(kwargs)
        super().add_cache(result, **kwargs)
        self._log.detail(
            "add_cache_written",
            result_type=type(result).__name__,
            model=kwargs.get("model"),
        )

    async def async_add_cache(self, result: Any, dynamic_cache_object: Any = None, **kwargs: Any) -> None:  # type: ignore[override]
        call_id = kwargs.get("litellm_call_id")
        self._log.detail(
            "async_add_cache_called",
            result_type=type(result).__name__,
            model=kwargs.get("model"),
            call_id=call_id,
        )

        if _is_empty_response(result):
            self._key_store.clear_async(call_id)
            self._log.detail("skip", reason="empty_assistant_content", model=kwargs.get("model"))
            return

        stored_key = self._key_store.pop_async(call_id)
        if stored_key:
            self._verify_async_key_if_debug(stored_key, kwargs)
            cached_data = {"timestamp": time.time(), "response": result}
            if dynamic_cache_object is not None:
                await dynamic_cache_object.async_set_cache(stored_key, cached_data, **kwargs)
            else:
                await self.cache.async_set_cache(stored_key, cached_data, **kwargs)
            self._log.detail(
                "async_add_cache_written",
                key=stored_key,
                result_type=type(result).__name__,
                model=kwargs.get("model"),
            )
            return

        await super().async_add_cache(result, dynamic_cache_object=dynamic_cache_object, **kwargs)
        self._log.detail(
            "async_add_cache_written",
            result_type=type(result).__name__,
            model=kwargs.get("model"),
        )

    # Override so litellm's parent class uses our normalizer.
    def _get_param_value(self, param: str, kwargs: dict) -> Optional[str]:  # type: ignore[override]
        result = _resolve_custom_param(param, kwargs, self._normalizer)
        if result is not _NOT_HANDLED:
            return result
        return super()._get_param_value(param, kwargs)

    # -- shared lookup helpers --

    def _finalize_lookup(
        self,
        cached_result: Any,
        key: str,
        source: str,
        raw_key: Optional[str],
        kwargs: dict[str, Any],
        max_age: float,
    ) -> Any:
        """Apply max-age logic and log the outcome. Shared by sync and async get."""
        result = self._get_cache_logic(cached_result=cached_result, max_age=max_age)
        if result is not None:
            self._log.hit()
            self._log.detail("hit", key=key, key_source=source, model=kwargs.get("model"))
        else:
            reason, age_seconds = _classify_miss(cached_result, max_age)
            self._log.miss()
            self._log.detail(
                "miss",
                reason=reason,
                key=key,
                key_source=source,
                raw_key=_dump_raw_key(raw_key),
                model=kwargs.get("model"),
                age=age_seconds,
                max_age=None if max_age == float("inf") else max_age,
            )
        return result

    def _log_miss(self, *, reason: str, source: str, kwargs: dict[str, Any]) -> None:
        self._log.miss()
        self._log.detail("miss", reason=reason, model=kwargs.get("model"), source=source)

    def _build_key_logged(self, kwargs: dict[str, Any]) -> tuple[Optional[str], Optional[str], str, list[str]]:
        if self._log.is_debug:
            from .key import _EXCLUDED_KEY_FIELDS

            self._log.detail(
                "cache_key_input",
                keys=sorted(kwargs.keys()),
                excluded=sorted(_EXCLUDED_KEY_FIELDS),
            )

        key, raw_key, source, fields = self._key_builder.build(kwargs)

        if self._log.is_debug:
            if source in ("explicit", "preset"):
                self._log.detail(f"cache_key_{source}", key=key)
            else:
                raw_hash = _hash_text(raw_key) if raw_key else None
                self._log.detail(
                    "cache_key_generated",
                    key=key,
                    raw_key=_dump_raw_key(raw_key),
                    raw_key_hash=raw_hash,
                    raw_key_len=len(raw_key) if raw_key else 0,
                )
                self._log.material(
                    {
                        "cache_key": key,
                        "raw_key_hash": raw_hash,
                        "raw_key_len": len(raw_key) if raw_key else 0,
                        "included_fields": fields,
                    }
                )

        return key, raw_key, source, fields

    def _verify_async_key_if_debug(self, stored_key: str, kwargs: dict[str, Any]) -> None:
        if not self._log.is_debug:
            return
        current_key, _, _, _ = self._key_builder.build(dict(kwargs))
        if current_key and current_key != stored_key:
            self._log.detail("cache_key_mismatch", stored_key=stored_key, current_key=current_key)
        else:
            self._log.detail("cache_key_async_reuse", key=stored_key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_litellm_cache(settings: Any) -> Cache:
    cache_dir = getattr(settings, "cache_dir", ".exgentic")
    litellm_cache_dir = getattr(settings, "litellm_cache_dir", "~/.cache/exgentic/litellm")
    return LLMCache(
        type="disk",
        disk_cache_dir=resolve_cache_path(cache_dir, litellm_cache_dir),
        delete_time_from_messages=settings.litellm_delete_time_from_cache_key,
    )
