# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

import json
import threading
from typing import Any, Optional

from ...core.orchestrator.observer import Observer


class ConfigsObserver(Observer):
    def __init__(self, run_id: str | None = None) -> None:
        super().__init__(run_id)
        self._lock = threading.Lock()
        self._run_config: Optional[Any] = None

    def on_run_start(self, run_config) -> None:
        with self._lock:
            self._run_config = run_config
        self._write_config(run_config)

    def on_run_success(self, results, run_config) -> None:
        with self._lock:
            self._run_config = run_config
        self._write_config(run_config)

    def on_run_error(self, error) -> None:
        with self._lock:
            run_config = self._run_config
        if run_config is None:
            return
        self._write_config(run_config)

    def _write_config(self, run_config) -> None:
        rp = self.paths
        config_path = rp.config
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    run_config.model_dump(mode="json"),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError:
            # Read-only runs should still be able to aggregate without persisting.
            return
