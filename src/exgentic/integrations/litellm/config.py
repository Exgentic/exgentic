# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""LiteLLM configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LitellmSettings:
    """Global LiteLLM configuration."""

    litellm_caching: bool
    litellm_delete_time_from_cache_key: bool
    cache_dir: str
    litellm_cache_dir: str
    log_level: str
    drop_params: bool = True
    modify_params: bool = True
    timeout: int = 180


def configure_litellm(
    *,
    config: LitellmSettings,
    cache_only: bool = False,
) -> None:
    """Configure LiteLLM for Exgentic.

    Args:
        config: Explicit LiteLLM configuration.
        cache_only: When True, only refresh cache configuration.
    """
    _configure_cache(config)
    if cache_only:
        return
    _configure_logging(config)
    _configure_callbacks()
    _configure_inference(config)


def _configure_cache(config: LitellmSettings) -> None:
    if not config.litellm_caching:
        return
    try:
        import litellm
    except ImportError:
        return
    from .cache_utils import build_litellm_cache

    litellm.cache = build_litellm_cache(config)
    litellm.enable_cache()


def _configure_callbacks() -> None:
    try:
        import litellm
    except ImportError:
        return
    from .trace_logger import (
        AsyncTraceLogger,
        SyncTraceLogger,
        async_trace_logger,
        sync_trace_logger,
    )

    # CustomLogger subclasses must be in litellm.callbacks (not
    # success_callback / failure_callback) so that litellm invokes
    # log_success_event / log_failure_event on them.
    if not any(isinstance(cb, SyncTraceLogger) for cb in litellm.callbacks):
        litellm.callbacks.append(sync_trace_logger)
    if not any(isinstance(cb, AsyncTraceLogger) for cb in litellm.callbacks):
        litellm.callbacks.append(async_trace_logger)


def _configure_logging(config: LitellmSettings) -> None:
    try:
        import logging

        import litellm
    except ImportError:
        return

    level = logging.DEBUG if str(config.log_level).upper() == "DEBUG" else logging.WARNING
    litellm.log_level = "DEBUG" if level == logging.DEBUG else "WARNING"
    litellm.suppress_debug_info = level != logging.DEBUG
    litellm.set_verbose = level == logging.DEBUG

    for name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router", "litellm"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False


def _configure_inference(config: LitellmSettings) -> None:
    try:
        import litellm
    except ImportError:
        return

    litellm.drop_params = config.drop_params
    litellm.modify_params = config.modify_params
    litellm.timeout = config.timeout
