# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Cache key construction and message normalization."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from litellm.caching.caching import Cache, ModelParamHelper, litellm
from litellm.types.utils import all_litellm_params

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _to_stable_json(value: Any) -> str:
    """Deterministic JSON string; falls back to str() for non-serializable types."""
    if not isinstance(value, (dict, list)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        return str(value)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. Message normalization
# ---------------------------------------------------------------------------

_MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

_TODAYS_DATE_REGEX = re.compile(r"(?i)\bToday'?s date(?:\s+is|:)\s*[^\n]*")

_DATE_TIME_REGEXES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?" r"(?:Z|[+-]\d{2}:\d{2})?\b"),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    re.compile(
        rf"\b{_MONTH_PATTERN}\s+\d{{1,2}}(?:st|n(?:d)|rd|th)?(?:,\s*\d{{2}}|\s+\d{{4}})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b\d{{1,2}}(?:st|n(?:d)|rd|th)?\s+{_MONTH_PATTERN}(?:,\s*\d{{2}}|\s+\d{{4}})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?\b"),
    re.compile(r"\b\d{1,2}\s?(?:[AaPp][Mm])\b"),
]


class MessageNormalizer:
    """Produces stable, cache-friendly representations of LLM message lists.

    Strips "Today's date" lines and date/time literals (when requested),
    removes non-deterministic IDs, drops None values and empty assistant messages.
    """

    def __init__(self, strip_time: bool) -> None:
        self._strip_time = strip_time

    def normalize(self, messages: Any) -> Any:
        normalized = self._normalize_value(messages)
        if isinstance(normalized, list):
            normalized = [m for m in normalized if not self._is_empty_assistant(m)]
        return normalized

    def to_json(self, messages: Any) -> Optional[str]:
        """Normalize then serialize. Returns None if messages is None."""
        if messages is None:
            return None
        normalized = self.normalize(messages)
        try:
            return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except TypeError:
            return str(normalized)

    # -- recursive value cleaning --

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clean_string(value)
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, dict):
            return self._clean_dict(value)
        for attr in ("model_dump", "dict"):
            fn = getattr(value, attr, None)
            if callable(fn):
                return self._normalize_value(fn())
        return value

    def _clean_string(self, text: str) -> str:
        cleaned = _TODAYS_DATE_REGEX.sub("", text)
        if self._strip_time:
            cleaned = _strip_date_time_literals(cleaned)
        return cleaned

    def _clean_dict(self, d: dict) -> dict:
        is_tool_msg = d.get("role") == "tool"
        result: dict[Any, Any] = {}
        for key, val in d.items():
            if val is None or key == "tool_call_id":
                continue
            if key == "id" and is_tool_msg:
                continue
            if key == "tool_calls" and isinstance(val, list):
                result[key] = [
                    self._normalize_value(
                        {k: v for k, v in call.items() if k != "id"}
                        if isinstance(call, dict) and "id" in call
                        else call
                    )
                    for call in val
                ]
                continue
            result[key] = self._normalize_value(val)
        return result

    @staticmethod
    def _is_empty_assistant(msg: Any) -> bool:
        return (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
            and msg.get("content", "").strip() == "(no content)"
        )


def _strip_date_time_literals(text: str) -> str:
    if not text:
        return text
    for regex in _DATE_TIME_REGEXES:
        text = regex.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# Backward-compatible module-level aliases.
strip_date_time_from_text = _strip_date_time_literals


def sanitize_messages_for_cache(messages: Any, *, strip_time: bool) -> Any:
    return MessageNormalizer(strip_time).normalize(messages)


# ---------------------------------------------------------------------------
# 2. Cache key building
# ---------------------------------------------------------------------------

_EXCLUDED_KEY_FIELDS = frozenset(
    {
        "litellm_call_id",
        "litellm_trace_id",
        "litellm_logging_obj",
        "litellm_metadata",
        "proxy_server_request",
        "parent_otel_span",
        "secret_fields",
        "shared_session",
        "use_in_pass_through",
        "use_litellm_proxy",
        "model_info",
        "provider_specific_header",
        "user",
        "dynamic_cache_object",
    }
)


def _sort_tools(tools: Any) -> Any:
    """Return tools in a deterministic order for stable cache keys."""
    if not isinstance(tools, list):
        return tools

    def sort_key(item: Any) -> tuple[str, str, str]:
        if not isinstance(item, dict):
            return ("", "", str(item))
        tool_type = str(item.get("type", ""))
        func = item.get("function")
        func_name = str(func.get("name", "")) if isinstance(func, dict) else ""
        try:
            stable = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except TypeError:
            stable = str(item)
        return (tool_type, func_name, stable)

    return sorted(tools, key=sort_key)


_NOT_HANDLED = object()


def _resolve_custom_param(param: str, kwargs: dict[str, Any], normalizer: MessageNormalizer) -> Any:
    """Resolve params that need custom serialization.

    Returns the string value for params we handle (messages, tools, sentinels),
    or _NOT_HANDLED for params that should fall through to litellm's default.

    Shared by CacheKeyBuilder and LLMCache._get_param_value so the same
    serialization logic is never duplicated.
    """
    if param in ("litellm_logging_obj", "litellm_call_id"):
        return param  # sentinel — present but content-irrelevant
    if param == "tools":
        return json.dumps(_sort_tools(kwargs.get("tools", [])), sort_keys=True)
    if param == "messages":
        return normalizer.to_json(kwargs.get("messages"))
    return _NOT_HANDLED


class CacheKeyBuilder:
    """Builds deterministic cache keys from LLM call kwargs.

    Pure logic — no logging, no side effects.
    """

    def __init__(self, normalizer: MessageNormalizer, cache: Cache) -> None:
        self._normalizer = normalizer
        self._cache = cache

    def build(self, kwargs: dict[str, Any]) -> tuple[Optional[str], Optional[str], str, list[str]]:
        """Return (cache_key, raw_key, source, included_fields)."""
        if "cache_key" in kwargs:
            return kwargs["cache_key"], None, "explicit", []

        preset = self._cache._get_preset_cache_key_from_kwargs(**kwargs)
        if preset is not None:
            return preset, None, "preset", []

        raw_key, included_fields = self._build_raw_key(kwargs)

        hashed = Cache._get_hashed_cache_key(raw_key)
        hashed = self._cache._add_namespace_to_cache_key(hashed, **kwargs)
        self._cache._set_preset_cache_key_in_kwargs(preset_cache_key=hashed, **kwargs)

        return hashed, raw_key, "generated", included_fields

    def _build_raw_key(self, kwargs: dict[str, Any]) -> tuple[str, list[str]]:
        filtered = {k: v for k, v in kwargs.items() if k not in _EXCLUDED_KEY_FIELDS}
        combined_params = ModelParamHelper._get_all_llm_api_params()
        litellm_params = all_litellm_params

        raw_key = ""
        included: list[str] = []

        for param in sorted(filtered):
            value = self._resolve(param, filtered, combined_params, litellm_params)
            if value is not None:
                raw_key += f"{param}: {value}"
                included.append(param)

        return raw_key, included

    def _resolve(
        self,
        param: str,
        kwargs: dict[str, Any],
        combined: set[str],
        litellm_only: set[str],
    ) -> Optional[str]:
        if param in combined:
            result = _resolve_custom_param(param, kwargs, self._normalizer)
            if result is not _NOT_HANDLED:
                return result
            # Fallback to litellm's default handling (e.g. model name normalization).
            return self._cache._get_param_value(param, kwargs)

        if param not in litellm_only and litellm.enable_caching_on_provider_specific_optional_params:
            value = kwargs[param]
            return _to_stable_json(value) if value is not None else None

        return None
