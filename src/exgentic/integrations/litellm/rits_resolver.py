# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""RITS model discovery and LiteLLM override construction."""

from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

RITS_API_KEY_ENV = "RITS_API_KEY"  # pragma: allowlist secret
RITS_API_URL_ENV = "RITS_API_URL"
RITS_INFERENCE_INFO_URL = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
RITS_REQUEST_TIMEOUT_SECONDS = 10.0


def get_rits_model_list() -> dict[str, str]:
    """Fetch available RITS models as ``model_name -> endpoint fragment``."""
    return _get_rits_model_list_cached(RITS_INFERENCE_INFO_URL)


@lru_cache(maxsize=8)
def _get_rits_model_list_cached(inference_info_url: str) -> dict[str, str]:
    api_key = os.getenv(RITS_API_KEY_ENV)
    if not api_key:
        raise OSError(f"Missing API key; please set '{RITS_API_KEY_ENV}'")
    request = Request(inference_info_url, headers={RITS_API_KEY_ENV: api_key})
    try:
        with urllib.request.urlopen(request, timeout=RITS_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            status = getattr(response, "status", 200)
    except HTTPError as exc:
        body = _read_error_body(exc)
        raise RuntimeError(f"Failed to fetch RITS model list: HTTP {exc.code} {exc.reason}. {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch RITS model list: {exc.reason}") from exc

    if status != 200:
        raise RuntimeError(f"Failed to fetch RITS model list: HTTP {status}. {body}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse RITS model list JSON") from exc

    return _parse_rits_model_list(payload)


def _read_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")
    except Exception:
        return ""


def _parse_rits_model_list(payload: Any) -> dict[str, str]:
    if not isinstance(payload, list):
        raise RuntimeError("RITS model list response must be a list")

    models: dict[str, str] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"RITS model list item {index} must be an object")
        model_name = item.get("model_name")
        endpoint = item.get("endpoint")
        if not isinstance(model_name, str) or not model_name:
            raise RuntimeError(f"RITS model list item {index} has invalid model_name")
        if not isinstance(endpoint, str) or not endpoint:
            raise RuntimeError(f"RITS model list item {index} has invalid endpoint")
        models[model_name] = endpoint.rstrip("/").split("/")[-1]
    return models


def resolve_rits_model(model_name: str) -> tuple[str, str]:
    """Resolve a RITS model name to ``(litellm_model, api_base)``."""
    api_url = os.getenv(RITS_API_URL_ENV)
    if not api_url:
        raise OSError(f"Missing API URL; please set '{RITS_API_URL_ENV}'")

    model_list = get_rits_model_list()
    model_url = model_list.get(model_name)
    if not model_url:
        available = ", ".join(sorted(model_list))
        raise ValueError(f"Model '{model_name}' not found in RITS. Available models: {available}")

    return f"hosted_vllm/{model_name}", f"{api_url.rstrip('/')}/{model_url}/v1"


def clear_rits_model_cache() -> None:
    """Clear the cached RITS model discovery response."""
    _get_rits_model_list_cached.cache_clear()


def build_rits_overrides(model: str) -> dict[str, Any]:
    """Build LiteLLM kwargs for a ``rits/...`` model alias."""
    remainder = model.removeprefix("rits/")

    try:
        litellm_model, api_base = resolve_rits_model(remainder)
    except ValueError:
        if "/" not in remainder:
            raise
        model_name = remainder.split("/", 1)[1]
        litellm_model, api_base = resolve_rits_model(model_name)

    api_key = os.getenv(RITS_API_KEY_ENV)
    if not api_key:
        raise OSError(f"{RITS_API_KEY_ENV} environment variable must be set to use RITS models")

    return {
        "model": litellm_model,
        "api_base": api_base,
        "api_key": api_key,
        "headers": {RITS_API_KEY_ENV: api_key},
    }
