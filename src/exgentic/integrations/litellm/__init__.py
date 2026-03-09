# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import os

from .health import acheck_model_accessible
from .proxy import LitellmProxy
from .trace_cost import load_trace_cost
from .trace_logger import (
    DEFAULT_FILE,
    FILE_ENV,
    TraceLogger,
    trace_logger,
)

# When running inside the LiteLLM proxy subprocess, eagerly initialise the
# Exgentic cache so that ``litellm.cache`` is set before any request arrives.
# The parent process sets EXGENTIC_PROXY_CACHE_INIT=true when it launches the
# proxy with disk caching enabled.
if os.environ.get("EXGENTIC_PROXY_CACHE_INIT", "").lower() in ("true", "1"):
    from ...utils.settings import get_settings

    get_settings()

__all__ = [
    "LitellmProxy",
    "trace_logger",
    "TraceLogger",
    "FILE_ENV",
    "DEFAULT_FILE",
    "load_trace_cost",
    "acheck_model_accessible",
]
