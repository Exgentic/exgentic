# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Minimal LiteLLM model accessibility check."""

from __future__ import annotations

import logging


async def acheck_model_accessible(model: str, **kwargs: object) -> None:
    """Raise if LiteLLM cannot access the configured model.

    Uses a minimal ``acompletion`` call instead of ``ahealth_check`` because
    the latter pulls in ``litellm.proxy`` internals that require the optional
    ``backoff`` package (only declared under ``litellm[proxy]``).

    Extra *kwargs* (e.g. ``api_base``, ``api_key``, ``headers``) are forwarded
    to ``litellm.acompletion`` so that custom providers such as RITS can be
    health-checked correctly.
    """
    import litellm

    await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        **kwargs,
    )


def check_model_accessible_sync(
    model: str,
    logger: logging.Logger,
    timeout: float = 60.0,
    **kwargs: object,
) -> None:
    """Synchronous wrapper for model health check.

    Args:
        model: The model identifier to check
        logger: Logger for info/error messages
        timeout: Timeout in seconds for the health check
        **kwargs: Extra arguments forwarded to ``acheck_model_accessible``
            (e.g. ``api_base``, ``api_key``, ``headers`` for custom providers).

    Raises:
        RuntimeError: If the model is not accessible
    """
    from ...utils.sync import run_sync

    logger.info("Running LiteLLM model health check (model=%s)", model)
    try:
        run_sync(acheck_model_accessible(model, **kwargs), timeout=timeout)
        logger.info("Model health check passed for %s", model)
    except Exception as exc:
        error_msg = getattr(exc, "message", "") or str(exc) or repr(exc)
        logger.error("Model health check failed for %s: %s", model, error_msg)
        raise RuntimeError(f"Model {model} is not accessible: {error_msg}") from exc
