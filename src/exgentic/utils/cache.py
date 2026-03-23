# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Unified benchmark/agent cache directory helpers."""

from __future__ import annotations

from pathlib import Path

from .settings import get_settings


def benchmark_cache_dir(slug: str) -> Path:
    """Return the cache directory for a benchmark: ``{cache_dir}/{slug}/``."""
    return Path(get_settings().cache_dir).expanduser() / slug
