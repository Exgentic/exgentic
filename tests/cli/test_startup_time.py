# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Guard against CLI startup regressions.

Importing the CLI entry point must stay fast — heavy dependencies like
litellm and uvicorn should only load when actually running evaluations.
"""

from __future__ import annotations

import subprocess
import sys

# Maximum allowed import time in seconds.  Current baseline is ~0.25s;
# the threshold is generous to account for CI variability and cold caches.
MAX_IMPORT_SECONDS = 1.0


def test_cli_import_time():
    """Importing the CLI entry point must complete within the time budget."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import time; start = time.monotonic(); "
                "from exgentic.interfaces.cli.main import main; "
                "elapsed = time.monotonic() - start; "
                "print(f'{elapsed:.3f}'); "
                f"assert elapsed < {MAX_IMPORT_SECONDS}, "
                f"f'CLI import took {{elapsed:.2f}}s, max {MAX_IMPORT_SECONDS}s'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI import too slow:\n{result.stdout}{result.stderr}"


def test_no_litellm_on_import():
    """Litellm must not be imported as a side effect of loading the CLI."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from exgentic.interfaces.cli.main import main; "
                "import sys; "
                "assert 'litellm' not in sys.modules, "
                "'litellm was imported at CLI startup — use lazy imports'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"litellm imported at startup:\n{result.stdout}{result.stderr}"


def test_no_uvicorn_on_import():
    """Uvicorn must not be imported as a side effect of loading the CLI."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from exgentic.interfaces.cli.main import main; "
                "import sys; "
                "assert 'uvicorn' not in sys.modules, "
                "'uvicorn was imported at CLI startup — use lazy imports'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"uvicorn imported at startup:\n{result.stdout}{result.stderr}"
