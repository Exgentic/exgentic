# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Tests for VenvBackend file lock during concurrent installs."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from exgentic.environment.venv import VenvBackend
from filelock import FileLock


@pytest.fixture
def env_dir(tmp_path: Path) -> Path:
    d = tmp_path / "envs" / "test_env"
    d.mkdir(parents=True)
    return d


def test_lock_file_created_during_install(env_dir: Path):
    """install() should create a .venv-install.lock file."""
    backend = VenvBackend()
    lock_path = env_dir / ".venv-install.lock"
    assert not lock_path.exists()

    try:
        backend.install(env_dir)
    except Exception:
        pass  # Install may fail without uv, that's OK

    assert lock_path.exists()


def test_install_blocked_while_lock_held(env_dir: Path):
    """install() should block when another process holds the lock."""
    lock_path = env_dir / ".venv-install.lock"
    lock = FileLock(str(lock_path), timeout=0)

    from filelock import Timeout as LockTimeout

    # Hold the lock from "another process".
    lock.acquire()
    try:
        backend = VenvBackend()
        # Patch timeout to 0 so install fails immediately instead of
        # waiting 600s for the lock we're holding.
        with patch("exgentic.environment.venv.FileLock", lambda p, timeout: FileLock(p, timeout=0)):
            with pytest.raises(LockTimeout):
                backend.install(env_dir)
    finally:
        lock.release()


def test_different_env_dirs_use_independent_locks(tmp_path: Path):
    """Installs to different env_dirs should not block each other."""
    dir_a = tmp_path / "env_a"
    dir_b = tmp_path / "env_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    backend = VenvBackend()
    results: list[str] = []
    lock = threading.Lock()

    def try_install(env_dir: Path, name: str):
        try:
            backend.install(env_dir)
        except Exception:
            pass
        with lock:
            results.append(name)

    t1 = threading.Thread(target=try_install, args=(dir_a, "a"))
    t2 = threading.Thread(target=try_install, args=(dir_b, "b"))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert sorted(results) == ["a", "b"]
