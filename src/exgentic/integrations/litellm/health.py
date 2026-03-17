# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Minimal LiteLLM model accessibility check."""

from __future__ import annotations

import logging


async def acheck_model_accessible(model: str) -> None:
    """Raise if LiteLLM cannot access the configured model."""
    import litellm

    result = await litellm.ahealth_check(
        {
            "model": model,
            "messages": [{"role": "user", "content": "test from litellm"}],
            "max_tokens": 5,
            "no-log": True,
        },
        mode="chat",
    )
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"LiteLLM health check failed: {result['error']}")


def check_model_accessible_sync(
    model: str,
    logger: logging.Logger,
    timeout: float = 15.0,
) -> None:
    """Synchronous wrapper for model health check.

    Args:
        model: The model identifier to check
        logger: Logger for info/error messages
        timeout: Timeout in seconds for the health check

    Raises:
        RuntimeError: If the model is not accessible
    """
    from ...utils.sync import run_sync

    logger.info("Running LiteLLM model health check (model=%s)", model)
    try:
        run_sync(acheck_model_accessible(model), timeout=timeout)
        logger.info("Model health check passed for %s", model)
    except Exception as exc:
        logger.error("Model health check failed for %s: %s", model, exc)
        raise RuntimeError(f"Model {model} is not accessible: {exc}") from exc
