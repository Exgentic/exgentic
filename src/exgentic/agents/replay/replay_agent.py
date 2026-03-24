# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""ReplayAgent — replays a recorded trajectory against a benchmark.

Usage::

    evaluate(benchmark="gsm8k", agent="replay", agent_kwargs={"recording": "path/to/recording"})

A *recording* is a directory containing:
    trajectory.jsonl   — the recorded action/observation events
    session.json       — session manifest (task, context, actions schema)

These files are produced automatically by ``exgentic evaluate`` (under
``outputs/<run_id>/sessions/<session_id>/``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from ...core.agent import Agent
from ...core.agent_instance import AgentInstance
from ...core.types import Action, Observation


class ReplayAgentInstance(AgentInstance):
    """Replays recorded actions from a trajectory file."""

    def __init__(
        self,
        *,
        session_id: str,
        trajectory_path: Path,
    ) -> None:
        super().__init__(session_id=session_id)
        self._actions = self._load_actions(trajectory_path)
        self._step = 0

    def start(self, task, context, actions):
        """Receive work payload and build the action_types lookup."""
        super().start(task, context, actions)
        self._action_types = {at.name: at for at in self.actions}

    @staticmethod
    def _load_actions(trajectory_path: Path) -> list[dict]:
        """Extract action events from a trajectory JSONL file."""
        actions = []
        with open(trajectory_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("event") == "action":
                    actions.append(event["action"])
        return actions

    def react(self, observation: Observation | None) -> Action | None:
        if self._step >= len(self._actions):
            return None  # No more recorded actions — signal done

        action_data = self._actions[self._step]
        self._step += 1

        name = action_data.get("name", "")
        arguments = action_data.get("arguments", {})

        action_type = self._action_types.get(name)
        if action_type is None:
            from ...core.actions import build_unknown_action

            return build_unknown_action(name, arguments)

        return action_type.build_action(arguments)

    def close(self) -> None:
        pass


class ReplayAgent(Agent):
    """Agent that replays pre-recorded actions from a trajectory file."""

    display_name: ClassVar[str] = "Replay Agent"
    slug_name: ClassVar[str] = "replay"
    recording: str  # Path to the recording directory (or trajectory.jsonl file)
    runner: str | None = "direct"  # No external deps — run in host process

    @classmethod
    def get_instance_class(cls):
        return ReplayAgentInstance

    def get_instance_kwargs(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        recording_path = Path(self.recording)
        if recording_path.is_file():
            trajectory_path = recording_path
        else:
            trajectory_path = recording_path / "trajectory.jsonl"
        return {
            "session_id": session_id,
            "trajectory_path": trajectory_path,
        }
