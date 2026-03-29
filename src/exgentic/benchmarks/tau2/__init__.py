# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import os
from pathlib import Path


def _resolve_tau2_data_dir() -> str:
    """Return the tau2 data directory path.

    Checks the cache directory first (populated by ``setup.sh``), then falls
    back to the legacy ``installation/`` path for backwards compatibility.
    """
    from ...utils.cache import benchmark_cache_dir

    cache_data = benchmark_cache_dir("tau2") / "data"
    if cache_data.is_dir():
        return str(cache_data)

    # Legacy path (pre-setup.sh installs that cloned into the package tree)
    legacy = Path(__file__).resolve().parent / "installation" / "tau2-bench" / "data"
    return str(legacy)


if "TAU2_DATA_DIR" not in os.environ:
    os.environ["TAU2_DATA_DIR"] = _resolve_tau2_data_dir()
