# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""ReplaySession — replays recorded observations from a trajectory.

Used together with ReplayAgent to test the full execution loop
without needing any benchmark dependencies installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from json_schema_to_pydantic import create_model

from ...core.session import Session
from ...core.types import Action, ActionType, Observation, SessionScore, SingleAction, SingleObservation


def _action_type_from_schema(entry: dict) -> ActionType:
    """Reconstruct an ActionType from a session.json action entry."""
    schema = entry["arguments_schema"]
    args_model = create_model(schema)

    # Create a SingleAction subclass with the right name + arguments type
    action_cls = type(
        f"{entry['name']}_Action",
        (SingleAction,),
        {"__annotations__": {"name": str, "arguments": args_model}},
    )

    return ActionType(
        name=entry["name"],
        description=entry.get("description", ""),
        cls=action_cls,
        is_finish=entry.get("is_finish", False),
        is_message=entry.get("is_message", False),
        is_hidden=entry.get("is_hidden", False),
    )


class ReplaySession(Session):
    """Session that replays recorded observations from a trajectory file.

    Does not require any benchmark dependencies — everything is
    reconstructed from the recording (session.json + trajectory.jsonl).
    """

    def __init__(
        self,
        task_id: str,
        recording_dir: str,
        *,
        session_id: str | None = None,
    ) -> None:
        del task_id  # Task is fully reconstructed from the recording.
        recording = Path(recording_dir)
        manifest = json.loads((recording / "session.json").read_text())

        self._task_id_val = manifest.get("task_id", "")
        self._task_val = manifest.get("task", "")
        self._context_val = manifest.get("context", {})
        self._action_types = [_action_type_from_schema(a) for a in manifest.get("actions", [])]

        # Load recorded observations and score from trajectory/results
        self._observations: list[Any] = []
        trajectory = recording / "trajectory.jsonl"
        with open(trajectory, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event.get("event") == "observation" and event.get("observation") is not None:
                    self._observations.append(event["observation"])

        # Load recorded score
        results_path = recording / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            self._recorded_score = results.get("details", results)
        else:
            self._recorded_score = {"score": 0.0, "success": False, "is_finished": True}

        self._step_idx = 0
        self._done = False

        if session_id is not None:
            self._session_id = session_id

        super().__init__()

    @property
    def task_id(self) -> str:
        return self._task_id_val

    @property
    def task(self) -> str:
        return self._task_val

    @property
    def context(self) -> dict[str, Any]:
        return self._context_val

    @property
    def actions(self) -> list[ActionType]:
        return self._action_types

    def _next_observation(self) -> Observation | None:
        if self._step_idx < len(self._observations):
            obs = self._observations[self._step_idx]
            self._step_idx += 1
            return SingleObservation(result=obs.get("result") if isinstance(obs, dict) else obs)
        return None

    def start(self) -> Observation | None:
        return self._next_observation()

    def step(self, action: Action) -> Observation | None:
        obs = self._next_observation()
        if obs is None:
            self._done = True
        return obs

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        data = self._recorded_score
        return SessionScore(
            score=float(data.get("score", 0.0)),
            success=bool(data.get("success", False)),
            is_finished=data.get("is_finished", True),
            session_metrics=data.get("session_metrics", {}),
            session_metadata=data.get("session_metadata", {}),
        )

    def close(self) -> None:
        pass
