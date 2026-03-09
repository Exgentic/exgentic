# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict

from ..utils.paths import SessionPaths, get_run_paths
from ..utils.settings import ExecuterName
from .session import Session
from .types import SessionIndex


class Benchmark(BaseModel, ABC):
    """Benchmark interface - controls evaluation execution and provides sessions."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    subset: str | None = None
    seed: int = 300
    executer: ExecuterName | None = None
    use_cache: bool = True
    max_interactions: int | None = 150

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable benchmark name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def slug_name(self) -> str:
        """Stable identifier for paths and CLI selection."""
        raise NotImplementedError

    @abstractmethod
    def list_tasks(self) -> List[str]:
        """Return available task identifiers for this benchmark."""
        raise NotImplementedError

    @abstractmethod
    def create_session(self, index: SessionIndex) -> Session:
        """Create a session for a specific task."""
        raise NotImplementedError

    @abstractmethod
    def aggregate_sessions(self, sessions: List[SessionIndex]) -> Dict[str, Any]:
        """Aggregate results for the specified task sessions."""
        raise NotImplementedError

    def get_sessions_paths(self, sessions: List[SessionIndex]) -> List[SessionPaths]:
        """Return ``SessionPaths`` for each session index."""
        run_paths = get_run_paths()
        return [run_paths.session(s.session_id) for s in sessions]

    @property
    def subset_name(self) -> str:
        """Stable subset identifier for this benchmark run."""
        return str(self.subset) if self.subset else "unknown"

    def list_subsets(self) -> List[str]:
        """Return available subset identifiers for this benchmark."""
        subset = self.subset_name
        return [subset] if subset and subset != "unknown" else []

    def close(self) -> None:
        """Optional cleanup hook."""
        return
