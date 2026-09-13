# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""LiteLLM model accessibility checks.

The check is layered, cheapest first:

1. **Environment validation** — :func:`validate_model_environment` asks litellm
   which credentials a model needs and which are missing. Free, offline, and
   gives an actionable message before any network traffic.
2. **Endpoint reachability** — :func:`check_models_endpoint` issues a
   ``GET /v1/models`` against the OpenAI-compatible surface. Unbilled, needs no
   token budget, and (unlike a completion probe) reports whether the model is
   actually served rather than inferring liveness from a generation succeeding.

Note that ``/v1/models`` validates the *serving* layer: a gateway can list a
model whose backend is broken. Verifying inference itself requires a real
completion call, which these checks deliberately avoid.

:func:`litellm.ahealth_check` is not used — it reaches into ``litellm.proxy``
internals that require the optional ``backoff`` package (declared only under
``litellm[proxy]``) and raises ``ImportError`` at runtime without it. Its
``chat`` mode is also just ``acompletion``, i.e. a billed call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.types.model_settings import ModelSettings


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Maps litellm exception type names to their category.  We match by name
# (not isinstance) so we don't need to import litellm at module level.
_TRANSIENT_TYPES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "BadGatewayError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    }
)

_REACHABLE_TYPES = frozenset(
    {
        "APIResponseValidationError",
        "BadRequestError",
        "BlockedPiiEntityError",
        "ContentPolicyViolationError",
        "ContextWindowExceededError",
        "GuardrailInterventionNormalStringError",
        "GuardrailRaisedException",
        "ImageFetchError",
        "InvalidRequestError",
        "JSONSchemaValidationError",
        "RejectedRequestError",
        "UnprocessableEntityError",
        "UnsupportedParamsError",
    }
)

_PERMANENT_TYPES = frozenset(
    {
        "AuthenticationError",
        "BudgetExceededError",
        "LiteLLMUnknownProvider",
        "NotFoundError",
        "PermissionDeniedError",
    }
)

_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503})
_REACHABLE_STATUS_CODES = frozenset({400, 422})
_PERMANENT_STATUS_CODES = frozenset({401, 403, 404})


class ErrorCategory(Enum):
    """Classification of a litellm error for retry/health-check decisions."""

    TRANSIENT = "transient"
    """May resolve on its own — retry with backoff."""

    REACHABLE = "reachable"
    """Model endpoint IS alive — it just rejected this request."""

    PERMANENT = "permanent"
    """Will not resolve across retries — fail immediately."""

    UNKNOWN = "unknown"
    """Unclassified — treat as non-retryable."""


def classify_error(exc: BaseException) -> ErrorCategory:
    """Classify a litellm / network error into a retry category.

    Classification priority:
    1. Python built-in types (TimeoutError, ConnectionError)
    2. HTTP status code (extracted from exc or exc.original_exception)
    3. litellm exception type name
    """
    # 1. Python built-ins
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorCategory.TRANSIENT

    # Check wrapped originals for timeouts.
    original = getattr(exc, "original_exception", None)
    if original is not None and isinstance(original, (TimeoutError, asyncio.TimeoutError)):
        return ErrorCategory.TRANSIENT

    # 2. HTTP status code (from exc or exc.original_exception)
    status = getattr(exc, "status_code", None)
    if status is None and original is not None:
        status = getattr(original, "status_code", None)
    if status is not None:
        status = int(status)
        if status in _PERMANENT_STATUS_CODES:
            return ErrorCategory.PERMANENT
        if status in _REACHABLE_STATUS_CODES:
            return ErrorCategory.REACHABLE
        if status in _TRANSIENT_STATUS_CODES:
            return ErrorCategory.TRANSIENT

    # 3. litellm exception type name
    name = type(exc).__name__
    if name in _PERMANENT_TYPES:
        return ErrorCategory.PERMANENT
    if name in _REACHABLE_TYPES:
        return ErrorCategory.REACHABLE
    if name in _TRANSIENT_TYPES:
        return ErrorCategory.TRANSIENT

    return ErrorCategory.UNKNOWN


# Convenience helpers for callers that just need a boolean.
def _is_transient_error(exc: BaseException) -> bool:
    return classify_error(exc) == ErrorCategory.TRANSIENT


def _is_model_reachable_error(exc: BaseException) -> bool:
    return classify_error(exc) == ErrorCategory.REACHABLE


def _is_permanent_error(exc: BaseException) -> bool:
    return classify_error(exc) == ErrorCategory.PERMANENT


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HealthCheckError(RuntimeError):
    """Raised when the model health check fails."""


# ---------------------------------------------------------------------------
# Layer 1: environment validation (free, offline)
# ---------------------------------------------------------------------------


def validate_model_environment(model: str) -> list[str]:
    """Return the credential env vars *model* needs but that are not set.

    Wraps :func:`litellm.validate_environment`, which knows the per-provider
    credential requirements. Costs nothing and touches no network, so it is the
    right first gate: a misconfigured run fails here with a precise message
    instead of surfacing as an opaque auth error mid-task.

    Args:
        model: Model identifier, optionally provider-prefixed (``azure/gpt-4.1``).

    Returns:
        Names of missing environment variables; empty when the environment is
        complete. Also empty when litellm has no requirements recorded for the
        model, so an empty list is not positive proof of a usable config.
    """
    import litellm

    try:
        result = litellm.validate_environment(model=model)
    except Exception:
        # Unknown/aliased models raise rather than report requirements. Absence
        # of information is not a failure — let the endpoint check decide.
        return []

    missing = result.get("missing_keys") or []
    return [str(key) for key in missing]


# ---------------------------------------------------------------------------
# Layer 2: endpoint reachability via GET /v1/models (unbilled)
# ---------------------------------------------------------------------------

# Env vars consulted for a provider's base URL, in precedence order. Checked
# per provider prefix (OPENAI_API_BASE, AZURE_API_BASE, ...) before falling
# back to the generic names.
_BASE_URL_SUFFIXES = ("_API_BASE", "_BASE_URL")

# Providers whose public endpoint is OpenAI-compatible and needs no explicit
# base URL to probe.
_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
}

_MODELS_PROBE_TIMEOUT = 10.0


def _resolve_api_base(provider: str) -> str | None:
    """Best-effort resolution of the OpenAI-compatible base URL for *provider*.

    Precedence: ``litellm.api_base`` → ``<PROVIDER>_API_BASE`` /
    ``<PROVIDER>_BASE_URL`` → generic ``OPENAI_API_BASE`` / ``OPENAI_BASE_URL``
    → a known public default. Returns *None* when nothing is discoverable, in
    which case the caller should skip the probe rather than guess.
    """
    import litellm

    configured = getattr(litellm, "api_base", None)
    if configured:
        return str(configured)

    candidates = [f"{provider.upper()}{suffix}" for suffix in _BASE_URL_SUFFIXES]
    candidates += [f"OPENAI{suffix}" for suffix in _BASE_URL_SUFFIXES]
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value

    return _DEFAULT_BASE_URLS.get(provider)


def _resolve_api_key(provider: str) -> str | None:
    """Return the API key for *provider* from the environment, if set."""
    for name in (f"{provider.upper()}_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _models_url(api_base: str) -> str:
    """Build the ``/v1/models`` URL for *api_base*, tolerating a trailing ``/v1``."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _fetch_models(url: str, api_key: str | None, timeout: float) -> tuple[int | None, object]:
    """GET *url*, returning ``(status_code, parsed_body_or_None)``.

    A status of *None* means the endpoint could not be reached at all
    (DNS/TCP/TLS failure or timeout) as opposed to responding with an error.
    """
    request = urllib.request.Request(url, method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            try:
                return status, json.loads(response.read().decode("utf-8"))
            except Exception:
                return status, None
    except urllib.error.HTTPError as err:
        return err.code, None
    except Exception:
        return None, None


def _served_model_ids(payload: object) -> set[str]:
    """Extract model ids from an OpenAI-style ``{"data": [{"id": ...}]}`` body."""
    if not isinstance(payload, dict):
        return set()
    data = payload.get("data")
    if not isinstance(data, list):
        return set()
    return {entry["id"] for entry in data if isinstance(entry, dict) and isinstance(entry.get("id"), str)}


def check_models_endpoint(
    model: str,
    logger: logging.Logger,
    timeout: float = _MODELS_PROBE_TIMEOUT,
) -> None:
    """Verify the model endpoint is reachable via ``GET /v1/models``.

    Unbilled and token-budget-free, so it behaves identically for reasoning and
    non-reasoning models. Distinguishes three outcomes:

    - **2xx** — endpoint is serving. When the body lists model ids and *model*
      is absent, that is logged as a warning rather than an error: gateways
      routinely alias or omit names from the listing.
    - **401/403** — endpoint is up but rejected our credentials. Treated as
      reachable (mirroring ``_is_proxy_ready``); a real auth failure surfaces on
      the first actual call with a far clearer message than a probe can give.
    - **unreachable / 5xx** — raised as :class:`HealthCheckError`.

    Skipped silently when no base URL is discoverable, since guessing one would
    produce false failures for self-hosted providers.

    Raises:
        HealthCheckError: If the endpoint cannot be reached or returns 5xx.
    """
    from litellm import get_llm_provider

    try:
        resolved_model, provider, _, _ = get_llm_provider(model=model)
    except Exception:
        resolved_model, provider = model, "openai"

    api_base = _resolve_api_base(provider)
    if not api_base:
        logger.debug("No API base discoverable for %s (provider=%s); skipping endpoint check", model, provider)
        return

    url = _models_url(api_base)
    status, payload = _fetch_models(url, _resolve_api_key(provider), timeout)

    if status is None:
        raise HealthCheckError(f"Model endpoint for {model} is unreachable at {url}")

    if status in (401, 403):
        logger.debug("Model endpoint %s is up but requires auth (HTTP %s)", url, status)
        return

    if status >= 500:
        raise HealthCheckError(f"Model endpoint for {model} returned HTTP {status} at {url}")

    if not 200 <= status < 300:
        # 4xx other than auth: the endpoint answered, so it is alive. The path
        # may not exist on this surface (some gateways omit /v1/models).
        logger.debug("Model endpoint %s returned HTTP %s; treating as reachable", url, status)
        return

    served = _served_model_ids(payload)
    if served and resolved_model not in served and model not in served:
        logger.warning(
            "Model %s not listed by %s (%d models served); it may be aliased or hidden",
            model,
            url,
            len(served),
        )
        return

    logger.debug("Model endpoint check passed for %s at %s", model, url)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


# Health checks run once at session startup with no retries to fail fast.
_HEALTH_MIN_RETRIES = 0
_HEALTH_MIN_RETRY_DELAY = 5.0
_HEALTH_MAX_RETRY_DELAY = 30.0


async def acheck_model_accessible(
    model: str,
    *,
    model_settings: ModelSettings | None = None,
) -> None:
    """Raise if LiteLLM cannot access the configured model.

    Uses a minimal ``acompletion`` call instead of ``ahealth_check`` because
    the latter pulls in ``litellm.proxy`` internals that require the optional
    ``backoff`` package (only declared under ``litellm[proxy]``).

    Error handling follows :func:`classify_error` categories:

    - **REACHABLE** — health check passes (endpoint responded).
    - **PERMANENT** — fails immediately (auth, not found, etc.).
    - **TRANSIENT** — retried with exponential backoff.
    - **UNKNOWN** — treated as non-retryable.

    Health checks enforce a minimum of :data:`_HEALTH_MIN_RETRIES`
    attempts with at least :data:`_HEALTH_MIN_RETRY_DELAY` seconds
    base delay (capped at :data:`_HEALTH_MAX_RETRY_DELAY`) so that
    flaky endpoints have enough time to recover.
    """
    import litellm

    from ...core.types.model_settings import ModelSettings as _ModelSettings
    from ...core.types.model_settings import RetryStrategy

    if model_settings is None:
        model_settings = _ModelSettings()

    num_retries = max(model_settings.num_retries or 0, _HEALTH_MIN_RETRIES)
    retry_after = max(model_settings.retry_after, _HEALTH_MIN_RETRY_DELAY)
    is_exponential = model_settings.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    last_exc: BaseException | None = None
    for attempt in range(1 + num_retries):
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                caching=False,
            )
            return
        except Exception as exc:
            last_exc = exc
            category = classify_error(exc)

            if category == ErrorCategory.REACHABLE:
                return
            if category == ErrorCategory.PERMANENT:
                raise
            if category == ErrorCategory.TRANSIENT and attempt < num_retries:
                delay = retry_after * (2**attempt) if is_exponential else retry_after
                delay = min(delay, _HEALTH_MAX_RETRY_DELAY)
                await asyncio.sleep(delay)
                continue
            raise

    if last_exc is not None:  # pragma: no cover
        raise last_exc


async def acheck_model_reachable(
    model: str,
    logger: logging.Logger,
    timeout: float = _MODELS_PROBE_TIMEOUT,
) -> None:
    """Async form of the layered check: environment validation + ``/v1/models``.

    Mirrors :func:`check_model_accessible_sync` minus the optional strict
    completion call. The probe itself is blocking ``urllib``, so it is run in a
    thread to avoid stalling the event loop.

    Raises:
        HealthCheckError: If the model is misconfigured or unreachable
    """
    missing = validate_model_environment(model)
    if missing:
        joined = ", ".join(missing)
        logger.error("Model health check failed for %s: missing %s", model, joined)
        raise HealthCheckError(f"Model {model} is not accessible: missing environment variables: {joined}")

    await asyncio.to_thread(check_models_endpoint, model, logger, timeout)


def check_model_accessible_sync(
    model: str,
    logger: logging.Logger,
    timeout: float = 30.0,
    model_settings: ModelSettings | None = None,
    *,
    strict: bool = False,
) -> None:
    """Check a model is configured and its endpoint is reachable.

    Runs the two cheap layers described in the module docstring: environment
    validation, then a ``GET /v1/models`` probe. Neither is billed and neither
    sends a token budget, so behaviour is identical for reasoning and
    non-reasoning models.

    This intentionally does *not* issue a completion call. A generation probe
    costs tokens and latency on every invocation and, because a malformed
    request classifies as :attr:`ErrorCategory.REACHABLE`, could report success
    without verifying anything. Pass ``strict=True`` to add that call back when
    inference itself must be verified.

    Args:
        model: The model identifier to check
        logger: Logger for info/error messages
        timeout: Timeout in seconds for the endpoint probe (default: 30s)
        model_settings: Optional retry settings, used only when *strict*.
        strict: Also issue a real completion call to verify inference works.

    Raises:
        HealthCheckError: If the model is misconfigured or unreachable
    """
    logger.info("Running LiteLLM model health check (model=%s)", model)

    missing = validate_model_environment(model)
    if missing:
        joined = ", ".join(missing)
        logger.error("Model health check failed for %s: missing %s", model, joined)
        raise HealthCheckError(f"Model {model} is not accessible: missing environment variables: {joined}")

    try:
        check_models_endpoint(model, logger, timeout=min(timeout, _MODELS_PROBE_TIMEOUT))
    except HealthCheckError as exc:
        logger.error("Model health check failed for %s: %s", model, exc)
        raise
    except Exception as exc:
        error_msg = getattr(exc, "message", "") or str(exc) or repr(exc)
        logger.error("Model health check failed for %s: %s", model, error_msg)
        raise HealthCheckError(f"Model {model} is not accessible: {error_msg}") from exc

    if strict:
        from ...utils.sync import run_sync

        try:
            run_sync(
                acheck_model_accessible(model, model_settings=model_settings),
                timeout=timeout,
            )
        except Exception as exc:
            error_msg = getattr(exc, "message", "") or str(exc) or repr(exc)
            logger.error("Model health check failed for %s: %s", model, error_msg)
            raise HealthCheckError(f"Model {model} is not accessible: {error_msg}") from exc

    logger.info("Model health check passed for %s", model)
