# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Venv environment: atomic build-then-rename into ``venv/``.

Every step of the pipeline runs inside a scratch directory, and the
scratch directory is renamed to ``venv/`` only after the whole build
succeeds. Readers that exec ``venv/bin/python`` therefore see either
nothing or a fully-populated venv — never a half-built state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .helpers import (
    build_subprocess_env,
    get_exgentic_version,
    install_packages,
    install_project,
    install_requirements,
    require_uv,
    run_setup_sh,
    validate_system_deps,
)

_SCRATCH_PREFIX = "venv.tmp."
_TRASH_PREFIX = "venv.old."


class VenvBackend:
    """Backend that creates an isolated Python venv with dependencies."""

    def install(
        self,
        env_dir: Path,
        *,
        module_path: str | None = None,
        **kwargs: object,
    ) -> dict:
        """Create a venv-based environment.

        Args:
            env_dir: Root directory for this environment.
            module_path: Dotted module path for locating package resources.
            **kwargs: Accepts ``project_root`` (Path) and ``packages`` (list).

        Returns:
            Empty dict (no extra marker data).
        """
        project_root: Path | None = kwargs.get("project_root")  # type: ignore[assignment]
        packages: list[str] | None = kwargs.get("packages")  # type: ignore[assignment]

        env_dir.mkdir(parents=True, exist_ok=True)
        venv_dir = env_dir / "venv"

        # Clean up scratch/trash dirs left behind by a prior crashed
        # install. Concurrent installs on the same env are already
        # serialized by EnvironmentManager's file lock, so this is safe.
        for child in env_dir.iterdir():
            if child.name.startswith((_SCRATCH_PREFIX, _TRASH_PREFIX)):
                shutil.rmtree(child, ignore_errors=True)

        scratch = env_dir / f"{_SCRATCH_PREFIX}{uuid.uuid4().hex}"
        try:
            uv = require_uv()
            # --relocatable: console-script shebangs must resolve their
            # interpreter relative to the script at runtime rather than
            # embedding the scratch path, or they become dead references
            # once we rename into place.
            subprocess.run(
                [
                    uv,
                    "venv",
                    str(scratch),
                    "--python",
                    f"{sys.version_info.major}.{sys.version_info.minor}",
                    "--relocatable",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            scratch_py = str(scratch / "bin" / "python")
            env = build_subprocess_env()
            if project_root is not None:
                install_project(uv, scratch_py, project_root, env)
            if packages:
                install_packages(uv, scratch_py, packages, env)
            if module_path is not None:
                install_requirements(uv, scratch_py, module_path, env)
                validate_system_deps(module_path)
                run_setup_sh(module_path, env_dir, venv_dir=scratch)

            _publish_atomic(scratch, venv_dir)
            scratch = None  # ownership transferred to venv_dir
        finally:
            if scratch is not None and scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)

        return {"exgentic_version": get_exgentic_version()}

    def exists(self, env_dir: Path, marker_data: dict) -> bool:
        """Check that the venv Python binary exists.

        With atomic publish, ``venv/bin/python`` can only exist if the
        entire install pipeline succeeded.
        """
        return (env_dir / "venv" / "bin" / "python").exists()

    def uninstall(self, env_dir: Path, marker_data: dict) -> None:
        """Remove the venv directory."""
        venv_dir = env_dir / "venv"
        if venv_dir.exists():
            shutil.rmtree(venv_dir)


def _publish_atomic(scratch: Path, venv_dir: Path) -> None:
    """Make *scratch* visible as *venv_dir* via atomic directory renames.

    POSIX ``rename`` on a directory is atomic but cannot replace a
    non-empty target, so we move any existing venv to a trash name
    first, rename the scratch in, and only then delete the trash. If
    the final rename fails, the prior venv is restored.
    """
    trash: Path | None = None
    if venv_dir.exists():
        trash = venv_dir.parent / f"{_TRASH_PREFIX}{uuid.uuid4().hex}"
        os.rename(venv_dir, trash)

    try:
        os.rename(scratch, venv_dir)
    except BaseException:
        # Best-effort restore: if the new venv couldn't be placed, move
        # the old one back so the user isn't left without any venv.
        if trash is not None and not venv_dir.exists():
            try:
                os.rename(trash, venv_dir)
                trash = None
            except OSError:
                pass
        raise
    finally:
        if trash is not None and trash.exists():
            shutil.rmtree(trash, ignore_errors=True)
