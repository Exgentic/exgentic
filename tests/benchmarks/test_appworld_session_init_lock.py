# SPDX-License-Identifier: Apache-2.0

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

from exgentic.benchmarks.appworld.appworld_eval import AppWorldSession


def test_create_world_serializes_parallel_initialization(monkeypatch):
    """Concurrent session startup should not initialize AppWorld concurrently."""
    state_lock = threading.Lock()
    active = 0
    overlap_detected = False

    class FakeAppWorld:
        def __init__(self, task_id: str, **_kwargs):
            nonlocal active, overlap_detected
            with state_lock:
                if active > 0:
                    overlap_detected = True
                active += 1
            time.sleep(0.05)
            with state_lock:
                active -= 1
            self.experiment_name = f"exp-{task_id}"

    fake_appworld_pkg = types.ModuleType("appworld")
    fake_environment_mod = types.ModuleType("appworld.environment")
    fake_environment_mod.AppWorld = FakeAppWorld

    monkeypatch.setitem(sys.modules, "appworld", fake_appworld_pkg)
    monkeypatch.setitem(sys.modules, "appworld.environment", fake_environment_mod)

    task_ids = [str(i) for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda task_id: AppWorldSession._create_world(task_id=task_id, env_kwargs={}), task_ids))

    assert overlap_detected is False
