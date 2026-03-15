# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import secrets
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict, List, Optional

from ..observers.logging import get_logger
from ..utils.cost import CostReport
from ..utils.paths import SessionPaths
from .types import Action, ActionType, Observation, SessionScore


class Session(ABC):
    """Session interface - represents one task execution in one environment."""

    def __init__(self) -> None:
        # Persist configuration once per session if provided by subclass.
        self.save_config()
        self.save_manifest()

    @property
    def session_id(self) -> str:
        """Returns a unique id for the session."""
        if not hasattr(self, "_session_id"):
            self._session_id = secrets.token_hex(4)
        return self._session_id

    @property
    def paths(self) -> SessionPaths:
        """Convenience wrapper for all filesystem paths for this session."""
        if not hasattr(self, "_paths"):
            from .context import try_get_context

            ctx = try_get_context()
            if ctx is not None:
                self._paths = SessionPaths(
                    session_id=self.session_id,
                    run_id=ctx.run_id,
                    output_dir=ctx.output_dir,
                )
            else:
                self._paths = SessionPaths(session_id=self.session_id, run_id="default", output_dir="outputs")
        return self._paths

    @property
    def logger(self) -> Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(f"Session_{self.session_id}", str(self.paths.session_log))
        return self._logger

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dict for this session."""
        return {}

    def save_config(self) -> None:
        """Persist the session configuration to the standard config path."""
        config = self.get_config()
        config_path = self.paths.benchmark_config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def save_results(self, payload: Dict[str, Any]) -> None:
        """Persist a results payload to the standard per-session results path."""
        results_path = self.paths.benchmark_results
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def save_standard_results(self, score: SessionScore) -> None:
        """Persist the minimal standardized results payload (score/success)."""
        self.save_results({"score": score.score, "success": bool(score.success)})

    def save_manifest(self) -> None:
        """Persist task/context/actions metadata for the session."""
        actions_payload = []
        for action_type in self.actions:
            action_entry = {
                "name": action_type.name,
                "description": action_type.description,
                "is_finish": bool(action_type.is_finish),
                "is_message": bool(action_type.is_message),
                "is_hidden": bool(action_type.is_hidden),
            }
            schema = action_type.arguments.model_json_schema()  # type: ignore[attr-defined]
            action_entry["arguments_schema"] = schema
            actions_payload.append(action_entry)

        manifest = {
            "run_id": self.paths.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "context": self.context,
            "actions": actions_payload,
            "task_id": self.task_id,
        }
        path = self.paths.session_manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def get_cost(self) -> CostReport:
        """Estimated session cost; default 0.0."""
        return CostReport.initialize_empty()

    @property
    def task_id(self) -> str:
        """Task identifier."""
        return ""

    @property
    @abstractmethod
    def task(self) -> str:
        """Task description - benchmark defines what work to do."""
        pass

    @property
    @abstractmethod
    def context(self) -> Dict[str, Any]:
        """Task context - benchmark provides necessary information."""
        pass

    @property
    @abstractmethod
    def actions(self) -> List[ActionType]:
        """Available actions - benchmark defines action space."""
        pass

    @abstractmethod
    def start(self) -> Optional[Observation]:
        """Current observation - session maintains state."""
        pass

    @abstractmethod
    def step(self, action: Action) -> Optional[Observation]:
        """Execute action - session controls execution, returns None when done."""
        pass

    @abstractmethod
    def done(self) -> bool:
        """Check completion - session knows when task is finished."""
        pass

    @abstractmethod
    def score(self) -> Dict[str, Any]:
        """Evaluate performance - session/benchmark controls scoring.

        Should return at least two fields:

        "success" - True iff the session execution completed without an error
        "score" - The score of the session execution, should be None if an error occurred.
        """
        pass

    @abstractmethod
    def close(self):
        """Cleanup resources - session manages its own resources."""
        pass
