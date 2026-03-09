# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import threading
from typing import Iterable

from ...observers.handlers.configs import ConfigsObserver
from ...observers.handlers.file_logger import FileLoggerObserver
from ...observers.handlers.logger import ConsoleLoggerObserver
from ...observers.handlers.recap import RunRecapObserver
from ...observers.handlers.results import ResultsObserver
from ...observers.handlers.warnings import WarningsObserver
from ...utils.settings import get_settings
from ..context import get_context
from .controller import CleanupController, Controller, CoreController, LimitController
from .observer import Observer
from .termination import AgentError, BenchmarkError


class Tracker(Observer, Controller):
    def __init__(
        self,
        *,
        observers: Iterable[Observer] | None = None,
        controllers: Iterable[Controller] | None = None,
        use_defaults: bool = True,
        max_steps: int = 100,
        max_actions: int = 100,
    ) -> None:
        self._run_id = get_context().run_id
        self._observers: list[Observer] = []
        self._controllers: list[Controller] = []
        self._results: ResultsObserver | None = None
        self.events = None
        self._run_lock = threading.Lock()
        self._run_state: str | None = None
        if use_defaults:
            self._register_observer(ResultsObserver())
            self._register_observer(ConfigsObserver())
            self._register_observer(WarningsObserver())
            self._register_observer(FileLoggerObserver())
            self._register_observer(ConsoleLoggerObserver())
            self._register_observer(RunRecapObserver(console=False))
            if get_settings().otel_enabled:
                from ...observers.handlers.otel import OtelTracingObserver

                self._register_observer(OtelTracingObserver())
            self._register_controller(LimitController(max_steps=max_steps, max_actions=max_actions))
            self._register_controller(CleanupController())
        if observers:
            for observer in observers:
                self._register_observer(observer)
        if controllers:
            for controller in controllers:
                self._register_controller(controller)
        self._ensure_core_controller()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def session_results(self):
        if self._results is not None:
            return self._results.session_results()
        return []

    def results(self):
        if self._results is not None:
            return self._results.results()
        raise RuntimeError("No results observer configured to provide results.")

    def _register_observer(self, observer: Observer | None) -> None:
        if observer is None:
            return
        if observer in self._observers:
            return
        if isinstance(observer, ResultsObserver):
            self._results = observer
        if observer.events is not None:
            self.events = observer.events
        self._observers.append(observer)

    def _register_controller(self, controller: Controller | None) -> None:
        if controller is None:
            return
        if controller in self._controllers:
            return
        self._controllers.append(controller)

    def on_run_start(self, run_config) -> None:
        for observer in self._observers:
            observer.on_run_start(run_config)

    def on_run_success(self, results, run_config) -> None:
        if not self._mark_run_final("success"):
            return
        for observer in self._observers:
            observer.on_run_success(results, run_config)
        for controller in self._controllers:
            controller.on_run_success(results, run_config)

    def on_run_error(self, error) -> None:
        if not self._mark_run_final("error"):
            return
        for observer in self._observers:
            observer.on_run_error(error)
        for controller in self._controllers:
            controller.on_run_error(error)

    def on_session_creation(self, session) -> None:
        for observer in self._observers:
            observer.on_session_creation(session)

    def on_session_start(self, session, agent, observation) -> None:
        for observer in self._observers:
            observer.on_session_start(session, agent, observation)

    def on_react_success(self, session, action) -> None:
        for observer in self._observers:
            observer.on_react_success(session, action)
        for controller in self._controllers:
            controller.on_react_success(session, action)

    def on_step_success(self, session, observation) -> None:
        for observer in self._observers:
            observer.on_step_success(session, observation)
        for controller in self._controllers:
            controller.on_step_success(session, observation)

    def on_react_error(self, session, error) -> None:
        for observer in self._observers:
            observer.on_react_error(session, error)
        for controller in self._controllers:
            controller.on_react_error(session, error)
        raise AgentError(error)

    def on_step_error(self, session, error) -> None:
        for observer in self._observers:
            observer.on_step_error(session, error)
        for controller in self._controllers:
            controller.on_step_error(session, error)
        raise BenchmarkError(error)

    def on_session_error(self, session, error) -> None:
        for observer in self._observers:
            observer.on_session_error(session, error)

    def on_session_success(self, session, score, agent) -> None:
        for observer in self._observers:
            observer.on_session_success(session, score, agent)

    def on_session_scoring(self, session) -> None:
        for observer in self._observers:
            observer.on_session_scoring(session)

    def on_session_reuse(self, session_results) -> None:
        for observer in self._observers:
            observer.on_session_reuse(session_results)

    def _ensure_core_controller(self) -> None:
        cores = [item for item in self._controllers if isinstance(item, CoreController)]
        non_cores = [item for item in self._controllers if not isinstance(item, CoreController)]
        if not cores:
            self._controllers = [*non_cores, CoreController()]
            return
        self._controllers = [*non_cores, cores[-1]]

    def _mark_run_final(self, state: str) -> bool:
        with self._run_lock:
            if self._run_state is not None:
                return False
            self._run_state = state
            return True
