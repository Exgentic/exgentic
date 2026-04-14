# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Minimal LiteLLM model accessibility check."""

from __future__ import annotations

import asyncio
import logging
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
# Health check
# ---------------------------------------------------------------------------


class HealthCheckError(RuntimeError):
    """Raised when the model health check fails."""


# Health checks run once at session startup and are not
# latency-sensitive, so we retry more patiently than regular API calls.
_HEALTH_MIN_RETRIES = 7
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


def check_model_accessible_sync(
    model: str,
    logger: logging.Logger,
    timeout: float = 240.0,
    model_settings: ModelSettings | None = None,
) -> None:
    """Synchronous wrapper for model health check.

    Args:
        model: The model identifier to check
        logger: Logger for info/error messages
        timeout: Timeout in seconds for the health check
        model_settings: Optional retry settings; uses ``ModelSettings()`` defaults
            when *None*.

    Raises:
        HealthCheckError: If the model is not accessible
    """
    from ...utils.sync import run_sync

    logger.info("Running LiteLLM model health check (model=%s)", model)
    try:
        run_sync(
            acheck_model_accessible(model, model_settings=model_settings),
            timeout=timeout,
        )
        logger.info("Model health check passed for %s", model)
    except Exception as exc:
        error_msg = getattr(exc, "message", "") or str(exc) or repr(exc)
        logger.error("Model health check failed for %s: %s", model, error_msg)
        raise HealthCheckError(f"Model {model} is not accessible: {error_msg}") from exc
