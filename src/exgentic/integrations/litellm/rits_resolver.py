"""RITS model discovery and URL resolution."""

from __future__ import annotations

import os
import requests
from functools import lru_cache
from typing import Dict

RITS_API_KEY_ENV = "RITS_API_KEY"
RITS_API_URL_ENV = "RITS_API_URL"
RITS_INFERENCE_INFO_URL = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
# Note: The actual RITS_API_URL may vary. Common values:
# - https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com
# - https://rits.fmaas.res.ibm.com/ritsapi


@lru_cache(maxsize=1)
def get_rits_model_list() -> Dict[str, str]:
    """Fetch available RITS models and their endpoint URLs.
    Returns:
        Dict mapping model_name to model_url fragment
    Raises:
        EnvironmentError: If RITS_API_KEY is not set
        RuntimeError: If API request fails
    """
    api_key = os.getenv(RITS_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key; please set '{RITS_API_KEY_ENV}' environment variable"
        )

    response = requests.get(
        RITS_INFERENCE_INFO_URL, headers={RITS_API_KEY_ENV: api_key}
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch RITS model list: {response.status_code}\n{response.text}"
        )

    return {
        model["model_name"]: model["endpoint"].split("/")[-1]
        for model in response.json()[::-1]
    }


def resolve_rits_model(model_name: str) -> tuple[str, str]:
    """Resolve RITS model name to LiteLLM format and API base.
    Args:
        model_name: Model name (e.g., "granite-3.1-8b-instruct")
    Returns:
        Tuple of (litellm_model, api_base)
        - litellm_model: "hosted_vllm/{model_name}"
        - api_base: "{RITS_API_URL}/{model_url}/v1"
    Raises:
        ValueError: If model not found in RITS
        EnvironmentError: If required env vars not set
    """
    api_url = os.getenv(RITS_API_URL_ENV)
    if not api_url:
        raise EnvironmentError(
            f"Missing API URL; please set '{RITS_API_URL_ENV}' environment variable"
        )

    model_list = get_rits_model_list()
    model_url = model_list.get(model_name)

    if not model_url:
        available = ", ".join(sorted(model_list.keys()))
        raise ValueError(
            f"Model '{model_name}' not found in RITS. "
            f"Available models: {available}"
        )

    litellm_model = f"hosted_vllm/{model_name}"
    api_base = f"{api_url.rstrip('/')}/{model_url}/v1"

    return litellm_model, api_base


def clear_rits_model_cache() -> None:
    """Clear the cached RITS model list.
    Useful when models are added/removed from RITS.
    """
    get_rits_model_list.cache_clear()


def build_rits_overrides(model: str) -> dict[str, str]:
    """Resolve a ``rits/`` prefixed model to concrete litellm parameters.

    Accepts model strings in these forms:
    - ``rits/<model_name>``               e.g. ``rits/gpt-oss-120b``
    - ``rits/<provider>/<model_name>``     e.g. ``rits/openai/gpt-oss-120b``

    The optional ``<provider>/`` segment (e.g. ``openai/``) is a litellm
    provider hint and is **not** part of the RITS model name, so it is
    stripped before lookup.

    Returns a dict with keys ``model``, ``api_base``, ``api_key`` and
    ``headers`` ready to be merged into completion call kwargs.
    """
    remainder = model.removeprefix("rits/")

    try:
        litellm_model, api_base = resolve_rits_model(remainder)
    except ValueError:
        if "/" in remainder:
            model_name = remainder.split("/", 1)[1]
            litellm_model, api_base = resolve_rits_model(model_name)
        else:
            raise

    api_key = os.getenv(RITS_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"{RITS_API_KEY_ENV} environment variable must be set to use RITS models"
        )

    return {
        "model": litellm_model,
        "api_base": api_base,
        "api_key": api_key,
        "headers": {RITS_API_KEY_ENV: api_key},
    }
