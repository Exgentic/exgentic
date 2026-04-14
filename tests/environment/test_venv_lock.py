# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Regression tests for concurrent install races in VenvBackend / EnvironmentManager."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from exgentic.environment.venv import VenvBackend

# ---------------------------------------------------------------------------
# Regression tests for the install-time races that caused the
# `ModuleNotFoundError: No module named 'filelock'` errors and the
# "appworld rebuilt 12 times per batch" issue.
#
# Bug 1: VenvBackend.install() published `venv/bin/python` before deps
# were installed, so workers that exec'd it mid-install crashed.
# Fix: build in a scratch dir, rename into place only after success.
#
# Bug 2: EnvironmentManager.install() checked is_installed() outside
# its lock and never re-checked inside, so N concurrent threads all
# committed to rebuilding the same env.
# Fix: wrap check-then-install in a FileLock with a double-check.
# ---------------------------------------------------------------------------


def test_atomic_publish_no_torn_state(tmp_path: Path, monkeypatch):
    """venv/bin/python is never visible at its final path until deps are installed."""
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    deps_installed = {"v": False}
    observations: list[bool] = []  # venv/bin/python existence at various mid-install moments

    def observe():
        observations.append((env_dir / "venv" / "bin" / "python").exists())

    def fake_subprocess_run(args, *a, **kw):
        # Simulate `uv venv <scratch>` — create bin/python inside the
        # scratch directory the caller passed, NOT the final venv/ dir.
        assert len(args) >= 2 and args[1] == "venv", f"unexpected: {args!r}"
        scratch = Path(args[2])
        (scratch / "bin").mkdir(parents=True, exist_ok=True)
        (scratch / "bin" / "python").touch()
        observe()
        return subprocess.CompletedProcess(args, 0, "", "")

    def fake_install_project(uv, py, root, env):
        observe()
        deps_installed["v"] = True

    monkeypatch.setattr("exgentic.environment.venv.subprocess.run", fake_subprocess_run)
    monkeypatch.setattr("exgentic.environment.venv.require_uv", lambda: "/fake/uv")
    monkeypatch.setattr("exgentic.environment.venv.install_project", fake_install_project)
    monkeypatch.setattr("exgentic.environment.venv.get_exgentic_version", lambda: "0.0.0")

    VenvBackend().install(env_dir, project_root=tmp_path)

    assert all(v is False for v in observations), "venv/bin/python visible mid-install"
    assert (env_dir / "venv" / "bin" / "python").exists()
    assert deps_installed["v"] is True


def test_concurrent_installs_rebuild_exactly_once(tmp_path: Path, monkeypatch):
    """Two threads racing into EnvironmentManager.install() trigger one backend build, not two."""
    from exgentic.environment import EnvType
    from exgentic.environment.manager import EnvironmentManager

    mgr = EnvironmentManager(base_dir=tmp_path)
    call_count = {"n": 0}
    count_lock = threading.Lock()
    first_entered = threading.Event()
    allow_first_to_finish = threading.Event()

    def slow_install(self, env_dir, **kwargs):
        with count_lock:
            call_count["n"] += 1
            idx = call_count["n"]
        if idx == 1:
            first_entered.set()
            allow_first_to_finish.wait(timeout=5)
        (env_dir / "venv" / "bin").mkdir(parents=True, exist_ok=True)
        (env_dir / "venv" / "bin" / "python").touch()
        return {"exgentic_version": "0.0.0"}

    monkeypatch.setattr(VenvBackend, "install", slow_install)
    monkeypatch.setattr("exgentic.environment.manager._now_iso", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr("exgentic.environment.helpers.get_exgentic_version", lambda: "0.0.0")

    errors: list[BaseException | None] = [None, None]

    def worker(idx: int):
        try:
            mgr.install("benchmarks/appworld", env_type=EnvType.VENV)
        except BaseException as e:
            errors[idx] = e

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1,))
    t1.start()
    first_entered.wait(timeout=5)
    t2.start()
    # Give T2 a moment to queue on the manager lock.
    threading.Event().wait(0.1)
    allow_first_to_finish.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors == [None, None], f"workers raised: {errors}"
    assert call_count["n"] == 1, (
        f"backend.install ran {call_count['n']} times for one env — "
        "the concurrent double-check under the manager lock is broken"
    )
