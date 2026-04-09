# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Minimal LiteLLM model accessibility check."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.types.model_settings import ModelSettings


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient failure worth retrying.

    Covers rate-limit (429), timeouts, and connection errors.
    """
    # Timeouts — endpoint is slow/loaded, not broken.
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    # Connection errors
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    # HTTP 429 rate limit
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    # Some wrappers surface the code inside a nested ``original_exception``.
    original = getattr(exc, "original_exception", None)
    if original is not None:
        if isinstance(original, (TimeoutError, asyncio.TimeoutError)):
            return True
        if getattr(original, "status_code", None) == 429:
            return True
    return False


class HealthCheckError(RuntimeError):
    """Raised when the model health check fails."""


async def acheck_model_accessible(
    model: str,
    *,
    model_settings: ModelSettings | None = None,
) -> None:
    """Raise if LiteLLM cannot access the configured model.

    Uses a minimal ``acompletion`` call instead of ``ahealth_check`` because
    the latter pulls in ``litellm.proxy`` internals that require the optional
    ``backoff`` package (only declared under ``litellm[proxy]``).

    Transient errors (rate-limit 429, timeouts, connection errors) are
    retried according to the retry parameters in *model_settings*
    (defaults to ``ModelSettings()``). Other errors are raised immediately.
    """
    import litellm

    from ...core.types.model_settings import ModelSettings as _ModelSettings
    from ...core.types.model_settings import RetryStrategy

    if model_settings is None:
        model_settings = _ModelSettings()

    num_retries = model_settings.num_retries or 0
    retry_after = model_settings.retry_after
    is_exponential = model_settings.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    last_exc: BaseException | None = None
    for attempt in range(1 + num_retries):
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                caching=False,  # Must hit the real endpoint, not a cached response.
            )
            return
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc) or attempt >= num_retries:
                raise
            delay = retry_after * (2**attempt) if is_exponential else retry_after
            await asyncio.sleep(delay)

    # Should be unreachable, but satisfy type checkers.
    if last_exc is not None:  # pragma: no cover
        raise last_exc


def check_model_accessible_sync(
    model: str,
    logger: logging.Logger,
    timeout: float = 60.0,
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
        RuntimeError: If the model is not accessible
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
