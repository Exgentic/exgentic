# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Minimal session-like object for Docker e2e tests.

Not a real Session subclass — avoids pulling in Session.__init__
which writes files.  We only need the method/property interface that
the ObjectProxy will forward over HTTP.
"""

from __future__ import annotations

import os


class DockerSession:
    """Minimal session-like object for Docker e2e tests."""

    def __init__(self, task_id: str, output_dir: str | None = None) -> None:
        self._task_id = task_id
        self._done = False
        self._good = 0
        self._steps = 0
        self._output_dir = output_dir

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        return f"Task {self._task_id}"

    @property
    def context(self) -> dict:
        return {"task_id": self._task_id}

    def start(self) -> dict:
        return {"result": "start"}

    def step(self, action_name: str) -> dict:
        self._steps += 1
        if action_name == "good":
            self._good += 1
            return {"result": "step"}
        if action_name == "finish":
            self._done = True
            return {"result": "finish"}
        return {"result": "step"}

    def done(self) -> bool:
        return self._done

    def score(self) -> dict:
        total = self._good
        return {"score": 1.0 if total > 0 else 0.0, "success": self._done and total > 0}

    def write_output(self, filename: str, content: str) -> str:
        """Write a file to the output dir.  Used to verify volume mounts."""
        out = self._output_dir or os.environ.get("EXGENTIC_OUTPUT_DIR", "/tmp")
        path = os.path.join(out, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return path

    def close(self) -> None:
        pass
