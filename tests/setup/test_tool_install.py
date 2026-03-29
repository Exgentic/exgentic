# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Integration test that simulates ``uv tool install exgentic``.

The test creates an isolated venv (the way ``uv tool install`` would),
installs exgentic into it, then runs ``exgentic setup --benchmark <slug>``
from a clean working directory that has **no** ``pyproject.toml`` in any
parent — exactly the situation a user hits after a global tool install.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_uv_available = shutil.which("uv") is not None

# Root of the exgentic source tree (two levels up from this file).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-benchmark setup timeout in seconds (5 minutes).
_SETUP_TIMEOUT = 300


_ALL_BENCHMARKS = [
    "tau2",
    "gsm8k",
    "appworld",
    "bfcl",
    "browsecompplus",
    "hotpotqa",
    "swebench",
]


@pytest.mark.skipif(not _uv_available, reason="uv CLI not available")
@pytest.mark.parametrize("benchmark", _ALL_BENCHMARKS)
def test_setup_in_tool_install_venv(benchmark: str, tmp_path: Path) -> None:
    """Install exgentic into a fresh venv, then run ``exgentic setup``."""
    venv_dir = tmp_path / "venv"
    work_dir = tmp_path / "workdir"
    work_dir.mkdir()

    # 1. Create an isolated venv using uv.
    subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", f"{sys.version_info.major}.{sys.version_info.minor}"],
        check=True,
        capture_output=True,
        timeout=60,
    )

    # 2. Install exgentic from the local source tree into the venv.
    subprocess.run(
        ["uv", "pip", "install", str(_REPO_ROOT), "--python", str(venv_dir / "bin" / "python")],
        check=True,
        capture_output=True,
        timeout=300,
    )

    # 3. Locate the ``exgentic`` entry-point inside the venv.
    exgentic_bin = venv_dir / "bin" / "exgentic"
    assert exgentic_bin.exists(), f"exgentic CLI not found at {exgentic_bin}"

    # 4. Run ``exgentic setup --benchmark <slug>`` from the clean workdir.
    #    The working directory deliberately has no pyproject.toml ancestors.
    result = subprocess.run(
        [str(exgentic_bin), "setup", "--benchmark", benchmark, "--force"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=_SETUP_TIMEOUT,
    )

    assert result.returncode == 0, (
        f"exgentic setup --benchmark {benchmark} failed "
        f"(rc={result.returncode}).\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # 5. Verify the installation marker was written.
    venv_python = str(venv_dir / "bin" / "python")
    check = subprocess.run(
        [
            venv_python,
            "-c",
            (
                "from exgentic.utils.installation_tracker import is_installed; "
                f"assert is_installed({benchmark!r}, 'benchmark'), "
                f"'installation marker not found for {benchmark}'"
            ),
        ],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert check.returncode == 0, (
        f"Installation marker check failed for {benchmark}.\n" f"stdout:\n{check.stdout}\nstderr:\n{check.stderr}"
    )
