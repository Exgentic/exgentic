# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tau2 import shim that centralizes logging configuration.

This module suppresses Tau2's default console logging at import time and
provides re-exports for Tau2 symbols used by Exgentic. It also disables
propagation on the stdlib logger for Tau2 so library logs don't bubble to
the application's console. Session code can add file sinks as needed.
"""
from __future__ import annotations

import logging

# 1) Quiet stdlib logging for the 'tau2' namespace
_tau2_logger = logging.getLogger("tau2")
if not _tau2_logger.handlers:
    _tau2_logger.addHandler(logging.NullHandler())
_tau2_logger.propagate = False

# 2) Quiet Loguru's default console sink if Loguru is present
try:
    from loguru import logger as _loguru

    # Remove all default sinks to avoid console output. Session code will add
    # a file sink per session when needed.
    try:
        _loguru.remove()
    except Exception:
        pass
except Exception:
    _loguru = None  # type: ignore

# 3) Re-export Tau2 modules used by Exgentic
from rich.console import Console  # noqa: E402
from tau2.agent.llm_agent import LLMAgent  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import Results, RunConfig, TerminationReason  # noqa: E402
from tau2.environment.tool import Tool  # noqa: E402
from tau2.metrics.agent_metrics import compute_metrics, is_successful  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.run import load_tasks, run_domain  # noqa: E402
from tau2.utils.display import ConsoleDisplay  # noqa: E402

# Tau2's llm_utils disables LiteLLM cache by default; re-enable Exgentic cache here.
from ...integrations.litellm.config import configure_litellm  # noqa: E402
from ...utils.settings import get_settings  # noqa: E402

configure_litellm(config=get_settings().to_litellm_config(), cache_only=True)

__all__ = [
    "Results",
    "RunConfig",
    "TerminationReason",
    "compute_metrics",
    "is_successful",
    "Tool",
    "registry",
    "load_tasks",
    "run_domain",
    "ConsoleDisplay",
    "Console",
    "LLMAgent",
    "AssistantMessage",
    "MultiToolMessage",
    "ToolCall",
    "ToolMessage",
    "UserMessage",
]
