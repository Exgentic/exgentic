# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from ...utils.paths import RunPaths


class Observer:
    events = None

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id
        self._paths: RunPaths | None = None

    @property
    def paths(self) -> RunPaths:
        if self._paths is None:
            from ..context import get_context

            ctx = get_context()
            if self._run_id is not None:
                self._paths = RunPaths(run_id=self._run_id, output_dir=ctx.output_dir)
            else:
                self._paths = RunPaths.from_context(ctx)
            self._run_id = self._paths.run_id
        return self._paths

    def on_run_start(self, run_config) -> None:
        return None

    def on_run_success(self, results, run_config) -> None:
        return None

    def on_run_error(self, error) -> None:
        return None

    def on_session_enter(self, session_id: str, task_id: str | None) -> None:
        """Fires right after ``session_scope`` is entered, before any service is spawned.

        Observers that need to publish context (e.g. an OTEL root span whose
        trace_id must land in the benchmark/agent runtime.json) must hook here
        rather than ``on_session_creation`` — the latter fires after benchmark
        services have already been spawned.
        """
        return

    def on_session_creation(self, session) -> None:
        return None

    def on_session_start(self, session, agent, observation) -> None:
        return None

    def on_react_success(self, session, action) -> None:
        return None

    def on_step_success(self, session, observation) -> None:
        return None

    def on_react_error(
        self,
        session,
        error,
    ) -> None:
        return None

    def on_step_error(self, session, error) -> None:
        return None

    def on_session_error(self, session, error) -> None:
        return None

    def on_session_success(self, session, score, agent) -> None:
        return None

    def on_session_scoring(self, session) -> None:
        return None

    def on_session_reuse(self, task_result) -> None:
        return None
