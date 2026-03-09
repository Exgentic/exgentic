# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import threading
import time
from typing import Dict

from ...core.orchestrator.observer import Observer
from ...core.orchestrator.termination import (
    AgentError,
    BenchmarkError,
    InvalidActionError,
    InvalidObservationError,
    RunCancelError,
    SessionCancelError,
)
from ...core.types import Action, Observation, SessionResults, SessionScore
from ...interfaces.registry import get_agent_entries, get_benchmark_entries
from ..logging import get_logger
from .session_ledger import SessionLedger


class FileLoggerObserver(Observer):
    def __init__(
        self,
        run_id: str | None = None,
        *,
        console: bool = False,
        logger=None,
    ) -> None:
        super().__init__(run_id)
        self._logger = logger
        self._console = console
        self._lock = threading.Lock()
        self._ledger = SessionLedger()
        self._session_reasons: Dict[str, str] = {}

    def _ensure_logger(self) -> None:
        if self._logger is not None:
            return
        rp = self.paths
        log_path = rp.tracker
        self._logger = get_logger(
            f"tracker.{self._run_id}",
            str(log_path),
            console=self._console,
            propagate=False,
        )

    def on_run_start(self, run_config) -> None:
        self._ensure_logger()
        run_id = self._run_id
        bench_entry = get_benchmark_entries().get(run_config.benchmark)
        agent_entry = get_agent_entries().get(run_config.agent)
        bench_name = bench_entry.display_name if bench_entry is not None else run_config.benchmark
        agent_name = agent_entry.display_name if agent_entry is not None else run_config.agent
        model_value = run_config.model or (run_config.agent_kwargs or {}).get("model")
        model_names = [str(model_value)] if model_value else None
        models_text = ", ".join(model_names) if model_names else ""
        if run_id is not None:
            self._logger.info("==== Exgentic Run %s ====", run_id)
        if models_text:
            self._logger.info(
                "Agent: %s (%s) | Benchmark: %s",
                agent_name,
                models_text,
                bench_name,
            )
        else:
            self._logger.info(
                "Agent: %s | Benchmark: %s",
                agent_name,
                bench_name,
            )

    def on_run_success(self, results, run_config) -> None:
        self._log_save()

    def on_run_error(self, error) -> None:
        self._log_error("run", None, error)
        self._log_save()

    def on_session_start(self, session, agent, observation) -> None:
        self._ensure_logger()
        session_id = session.session_id
        session_number = self._ledger.register(session_id)
        self._logger.info("▶️  Starting Session %s (logs: %s)", session_number, session.paths.root)
        if isinstance(observation, Observation):
            self._logger.info("⏺️  Recorded Start Session %s", session_number)

    def on_react_success(self, session, action) -> None:
        self._ensure_logger()
        if action is None:
            self._set_reason(session, "ended by agent (agent returned None action)")
            return
        if not isinstance(action, Action):
            self._set_reason(
                session,
                f"terminated by illegal action returned from agent: {action}",
            )
            self._log_error("react", session, InvalidActionError(action))
            return
        step_n, session_number = self._step(session)
        self._logger.info("⏩ Recorded Step %s Session %s", step_n, session_number)

    def on_step_success(self, session, observation) -> None:
        self._ensure_logger()
        if observation is not None and not isinstance(observation, Observation):
            self._set_reason(
                session,
                "terminated by illegal observation returned from session: " f"{observation}",
            )
            self._log_error("step", session, InvalidObservationError(observation))
            return
        if observation is None:
            self._set_reason(session, "ended by benchmark")

    def on_react_error(self, session, error) -> None:
        self._ensure_logger()
        if isinstance(error, InvalidActionError):
            self._set_reason(
                session,
                f"terminated by illegal action returned from agent: {error.action}",
            )
        else:
            self._set_reason(session, "terminated by agent exception")

    def on_step_error(self, session, error) -> None:
        self._ensure_logger()
        if isinstance(error, InvalidObservationError):
            self._set_reason(
                session,
                "terminated by illegal observation returned from session: " f"{error.observation}",
            )
        else:
            self._set_reason(session, "terminated by session exception")

    def on_session_error(self, session, error) -> None:
        self._ensure_logger()
        error_source = None
        if isinstance(error, AgentError):
            error_source = "agent"
        elif isinstance(error, BenchmarkError):
            error_source = "benchmark"
        if isinstance(error, (SessionCancelError, RunCancelError, KeyboardInterrupt)):
            self._set_reason(session, "cancelled by user", overwrite=True)
            error_source = "cancelled"
        else:
            if error_source == "agent":
                reason = "terminated by agent exception"
            elif error_source == "benchmark":
                reason = "terminated by benchmark exception"
            else:
                reason = "terminated by unexpected exception (see console)"
            self._set_reason(session, reason, overwrite=True)
            session_id = session.session_id if session else None
            detail = self._format_error_detail(error)
            if error_source == "agent":
                self._logger.error(
                    "Agent error in session %s: %s",
                    session_id or "-",
                    detail,
                )
            elif error_source == "benchmark":
                self._logger.error(
                    "Benchmark error in session %s: %s",
                    session_id or "-",
                    detail,
                )
            else:
                self._logger.error(
                    "Session error in session %s: %s",
                    session_id or "-",
                    detail,
                )
        score = SessionScore(score=0, success=False, is_finished=None)
        self._log_session(session, score)

    def on_session_success(self, session, score, agent) -> None:
        self._ensure_logger()
        self._log_session(session, score)

    def on_session_scoring(self, session) -> None:
        self._ensure_logger()
        session_id = session.session_id
        session_number = self._ledger.get_number(session_id)
        self._logger.info("⏳ Scoring Session %s (logs: %s)", session_number, session.paths.root)

    def _log_save(self) -> None:
        self._ensure_logger()
        rp = self.paths
        self._logger.info("💾 Saving results to %s", rp.root)

    def _step(self, session) -> tuple[int, int]:
        session_id = session.session_id
        step_n = self._ledger.increment_steps(session_id)
        session_number = self._ledger.get_number(session_id)
        return step_n, session_number

    def _log_session(self, session, score: SessionScore) -> None:
        session_id = session.session_id
        session_number = self._ledger.get_number(session_id)
        stats = self._ledger.pop_state(session_id)
        execution_time = time.time() - stats.started_at if stats is not None else 0.0
        steps = stats.steps if stats is not None else 0
        reason = self._pop_reason(session)
        self._logger.info("⏹️  Session %s %s.", session_number, reason)

        success = bool(score.success)
        value = score.score
        is_finished = score.is_finished
        score_text = f"{value}"
        success_emoji = self._success_emoji(success, value, is_finished)
        status = self._status_label(success, is_finished, reason)
        task_id = session.task_id
        task_id_str = f" | task_id: {task_id}" if task_id else ""
        self._logger.info(
            "%s Completed Session %s | status: %s | score: %s | steps: %s | time: %.1fs%s\n" "logs: %s",
            success_emoji,
            session_number,
            status,
            score_text,
            steps,
            execution_time,
            task_id_str,
            session.paths.root,
        )

    def _log_error(self, where, session, error) -> None:
        self._ensure_logger()
        session_id = session.session_id if session else None
        if session_id:
            self._logger.error(
                "error (%s) session=%s: %s",
                where,
                session_id,
                error,
            )
        else:
            self._logger.error("error (%s): %s", where, error)

    def on_session_reuse(self, session_results: SessionResults) -> None:
        self._ensure_logger()
        session_id = session_results.session_id
        session_number = self._ledger.mark_reuse(session_id)

        reason = "reused existing session"
        success = bool(session_results.success)
        value = session_results.score
        is_finished = session_results.is_finished
        score_text = f"{value}"
        success_emoji = self._success_emoji(success, value, is_finished)
        status = self._status_label(success, is_finished, reason)
        task_id = session_results.task_id
        task_id_str = f" | task_id: {task_id}" if task_id else ""
        execution_time = float(session_results.execution_time or 0.0)
        steps = int(session_results.steps or 0)
        sess_paths = self.paths.session(session_id)
        self._logger.info(
            "⏭️  Reused Session %s from existing results.",
            session_number,
        )
        self._logger.info(
            "%s Completed Session %s | status: %s | score: %s | steps: %s | time: %.1fs%s\n" "logs: %s",
            success_emoji,
            session_number,
            status,
            score_text,
            steps,
            execution_time,
            task_id_str,
            sess_paths.root,
        )

    @staticmethod
    def _format_error_detail(error: Exception | None) -> str:
        detail = error
        if isinstance(error, (AgentError, BenchmarkError)):
            detail = error.error
        if detail is None:
            return "unknown error"
        text = str(detail)
        if not text:
            return type(detail).__name__
        if isinstance(detail, Exception):
            return f"{type(detail).__name__}: {text}"
        return text

    def _set_reason(self, session, reason: str, *, overwrite: bool = False) -> None:
        session_id = session.session_id if session else None
        if session_id is None:
            return
        with self._lock:
            if session_id in self._session_reasons and not overwrite:
                return
            self._session_reasons[session_id] = reason

    def _pop_reason(self, session) -> str:
        session_id = session.session_id if session else None
        if session_id is None:
            return "ended"
        with self._lock:
            return self._session_reasons.pop(session_id, "ended")

    @staticmethod
    def _success_emoji(success: bool, value: float | None, is_finished: bool | None) -> str:
        if success:
            if value is not None and value == 1.0:
                return "✅"
            return "☑️ "
        if is_finished is True:
            return "☑️ "
        if is_finished is False:
            return "⚠️"
        return "❌"

    @staticmethod
    def _status_label(success: bool, is_finished: bool | None, reason: str) -> str:
        if success:
            return "success"
        if is_finished is True:
            return "unsuccessful"
        if is_finished is False:
            return "unfinished"
        reason_lower = reason.lower() if reason else ""
        if "illegal action" in reason_lower or "agent" in reason_lower:
            return "agent error"
        if (
            "benchmark" in reason_lower
            or "session exception" in reason_lower
            or "illegal observation" in reason_lower
            or "observation returned from session" in reason_lower
        ):
            return "benchmark error"
        if "cancelled" in reason_lower:
            return "cancelled"
        return "error"
