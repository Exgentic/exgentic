# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...core.orchestrator.observer import Observer
from ...core.orchestrator.termination import (
    AgentError,
    BenchmarkError,
    InvalidActionError,
    InvalidObservationError,
    RunCancelError,
    SessionCancelError,
)
from ...core.types import (
    Action,
    BenchmarkResults,
    Observation,
    RunResults,
    RunStatus,
    SessionExecutionStatus,
    SessionOutcomeStatus,
    SessionResults,
    SessionScore,
)
from ...interfaces.registry import get_agent_entries, get_benchmark_entries
from ...utils.cost import CostReport, accumulate_reports
from .session_ledger import SessionLedger


@dataclass
class _SessionData:
    action_count: int = 0
    invalid_action_count: int = 0
    agent: Any | None = None
    reason: Optional[str] = None


class ResultsObserver(Observer):
    def __init__(self, run_id: str | None = None) -> None:
        super().__init__(run_id)
        self._lock = threading.Lock()
        self._ledger = SessionLedger()
        self._sessions: Dict[str, _SessionData] = {}
        self._session_results: list[SessionResults] = []
        self._results: RunResults | None = None
        self._run_config: Optional[Any] = None
        self._final_results: Optional[Any] = None

    def on_run_start(self, run_config) -> None:
        with self._lock:
            self._run_config = run_config
        if self._run_id is None:
            self._run_id = run_config.run_id

    def on_session_start(self, session, agent, observation) -> None:
        session_id = session.session_id
        self._ledger.register(session_id)
        with self._lock:
            self._sessions[session_id] = _SessionData(agent=agent)
        if isinstance(observation, Observation):
            self._record_observation(session, observation, step=0, initial=True)

    def on_react_success(self, session, action) -> None:
        if action is None:
            self._set_reason(session, "ended by benchmark (agent returned None action)")
            return
        if not isinstance(action, Action):
            self._set_reason(
                session,
                f"terminated by illegal action returned from agent: {action}",
            )
            return
        self._record_action(session, action)

    def on_step_success(self, session, observation) -> None:
        if observation is not None and not isinstance(observation, Observation):
            self._set_reason(
                session,
                "terminated by illegal observation returned from session: " f"{observation}",
            )
            return
        self._record_observation(session, observation)
        if observation is None:
            self._set_reason(session, "ended by agent (session returned None observation)")

    def on_react_error(self, session, error) -> None:
        if isinstance(error, InvalidActionError):
            self._set_reason(
                session,
                f"terminated by illegal action returned from agent: {error.action}",
            )
        else:
            self._set_reason(session, "terminated by agent exception")

    def on_step_error(self, session, error) -> None:
        if isinstance(error, InvalidObservationError):
            self._set_reason(
                session,
                "terminated by illegal observation returned from session: " f"{error.observation}",
            )
        else:
            self._set_reason(session, "terminated by session exception")

    def on_session_error(self, session, error) -> None:
        error_source = None
        if isinstance(error, AgentError):
            error_source = "agent"
        elif isinstance(error, BenchmarkError):
            error_source = "benchmark"
        if isinstance(error, (SessionCancelError, RunCancelError, KeyboardInterrupt)):
            self._set_reason(session, "cancelled by user")
            error_source = "cancelled"
        else:
            self._set_reason(
                session,
                "terminated by unexpected exception (see console)",
            )
        root_error = error.error if isinstance(error, (AgentError, BenchmarkError)) else None
        error_message = str(root_error) if root_error else str(error)
        session_metadata = {"error": error_message}
        if error_source is not None:
            session_metadata["error_source"] = error_source
        score = SessionScore(
            score=0,
            success=False,
            is_finished=None,
            session_metadata=session_metadata,
        )
        self._record_session(session, score)

    def on_session_success(self, session, score, agent) -> None:
        self._record_session(session, score, agent=agent)

    def on_run_success(self, results, run_config) -> None:
        with self._lock:
            self._final_results = results
            self._run_config = run_config
        self._results = self._write_run_results()

    def on_run_error(self, error) -> None:
        self._results = self._write_run_results()

    def results(self) -> RunResults:
        if self._results is None:
            raise RuntimeError("Run results have not been computed yet.")
        return self._results

    def session_results(self) -> list[SessionResults]:
        return list(self._session_results)

    def on_session_reuse(self, session_results: SessionResults) -> None:
        with self._lock:
            self._session_results.append(session_results)

    def _record_action(self, session, action: Action) -> int:
        session_id = session.session_id
        single_actions = action.to_action_list()
        total_actions = len(single_actions)
        invalid_actions = 0
        for single_action in single_actions:
            report = single_action.validation
            if not report.valid or not report.name_valid or not report.args_valid:
                invalid_actions += 1
        step_n = self._ledger.increment_steps(session_id)
        session_number = self._ledger.get_number(session_id)
        with self._lock:
            data = self._sessions.get(session_id)
            if data is None:
                data = _SessionData()
                self._sessions[session_id] = data
            data.action_count += total_actions
            data.invalid_action_count += invalid_actions
        agent_cost, benchmark_cost = self._get_cost_snapshot(session_id, session)
        traj_path = session.paths.trajectory
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "action",
            "run_id": self._run_id,
            "session_id": session_id,
            "session_number": session_number,
            "task_id": session.task_id,
            "step": step_n,
            "action": json.loads(action.model_dump_json()),
            "initial": False,
            "agent_cost": agent_cost,
            "benchmark_cost": benchmark_cost,
        }
        with open(traj_path, "a", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
            f.write("\n")
        return step_n

    def _record_observation(
        self,
        session,
        observation: Optional[Observation],
        *,
        step: Optional[int] = None,
        initial: bool = False,
    ) -> None:
        session_id = session.session_id
        session_number = self._ledger.get_number(session_id)
        step_n = self._ledger.get_steps(session_id) if step is None else step
        agent_cost, benchmark_cost = self._get_cost_snapshot(session_id, session)
        traj_path = session.paths.trajectory
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(observation.model_dump_json()) if observation is not None else None
        event = {
            "event": "observation",
            "run_id": self._run_id,
            "session_id": session_id,
            "session_number": session_number,
            "task_id": session.task_id,
            "step": step_n,
            "observation": payload,
            "initial": initial,
            "agent_cost": agent_cost,
            "benchmark_cost": benchmark_cost,
        }
        with open(traj_path, "a", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
            f.write("\n")

    def _record_session(self, session, score: SessionScore, *, agent=None) -> None:
        session_id = session.session_id
        with self._lock:
            data = self._sessions.get(session_id)
        state = self._ledger.pop_state(session_id)
        execution_time = time.time() - state.started_at if state is not None else 0.0
        steps = state.steps if state is not None else 0
        action_count = data.action_count if data is not None else 0
        invalid_action_count = data.invalid_action_count if data is not None else 0
        success = bool(score.success)
        value = score.score
        is_finished = score.is_finished
        agent_cost_report = agent.get_cost() if agent is not None else CostReport.initialize_empty()
        benchmark_cost_report = session.get_cost()
        status = self._resolve_session_status(score)
        tr = SessionResults(
            session_id=session_id,
            success=success,
            score=value,
            is_finished=is_finished,
            status=status,
            steps=steps,
            action_count=action_count,
            invalid_action_count=invalid_action_count,
            agent_cost=agent_cost_report.total_cost,
            benchmark_cost=benchmark_cost_report.total_cost,
            execution_time=execution_time,
            details=score.model_dump(),
            cost_reports={
                "agent": agent_cost_report,
                "benchmark": benchmark_cost_report,
            },
            task_id=session.task_id,
        )
        self._pop_reason(session)
        with self._lock:
            self._session_results.append(tr)
            if session_id in self._sessions:
                del self._sessions[session_id]
        sess_paths = self.paths.session(session_id)
        sess_paths.results.parent.mkdir(parents=True, exist_ok=True)
        with open(sess_paths.results, "w", encoding="utf-8") as f:
            json.dump(tr.model_dump(), f, ensure_ascii=False, indent=2)
        error_message = score.session_metadata.get("error")
        if error_message:
            error_source = score.session_metadata.get("error_source")
            error_path = sess_paths.error_log
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, "w", encoding="utf-8") as f:
                if error_source:
                    f.write(f"source: {error_source}\n")
                f.write(str(error_message))

    def _resolve_session_status(self, score: SessionScore) -> SessionOutcomeStatus:
        error_source = score.session_metadata.get("error_source")
        if error_source == "cancelled":
            return SessionOutcomeStatus.CANCELLED
        if score.session_metadata.get("limit_reached"):
            if score.is_finished is True and score.success:
                return SessionOutcomeStatus.SUCCESS
            return SessionOutcomeStatus.LIMIT_REACHED
        if error_source in ("agent", "benchmark"):
            return SessionOutcomeStatus.ERROR
        if score.session_metadata.get("error"):
            return SessionOutcomeStatus.ERROR
        if score.is_finished is True:
            return SessionOutcomeStatus.SUCCESS if score.success else SessionOutcomeStatus.UNSUCCESSFUL
        if score.is_finished is False:
            return SessionOutcomeStatus.UNFINISHED
        return SessionOutcomeStatus.ERROR if not score.success else SessionOutcomeStatus.UNKNOWN

    def _write_run_results(self) -> RunResults:
        rp = self.paths
        with self._lock:
            results_snapshot = list(self._session_results)
            run_config = self._run_config
            bench_results_obj = self._final_results if isinstance(self._final_results, BenchmarkResults) else None
        if run_config is None:
            raise RuntimeError("Run config not recorded in results observer.")

        # Derive current session status snapshot for provenance.
        try:
            status = RunStatus.from_config(run_config)
        except Exception:
            status = None

        completed_sessions = None
        incomplete_sessions = None
        missing_sessions = None
        running_sessions = None
        aggregated_session_ids = None
        skipped_session_ids = None
        skipped_session_reasons = None
        missing_result_files = None

        if status is not None:
            completed = [s for s in status.session_statuses if s.status == SessionExecutionStatus.COMPLETED]
            incomplete = [s for s in status.session_statuses if s.status == SessionExecutionStatus.INCOMPLETE]
            missing = [s for s in status.session_statuses if s.status == SessionExecutionStatus.MISSING]
            running = [s for s in status.session_statuses if s.status == SessionExecutionStatus.RUNNING]
            completed_sessions = len(completed)
            incomplete_sessions = len(incomplete)
            missing_sessions = len(missing)
            running_sessions = len(running)
            aggregated_session_ids = [s.session_id for s in completed]
            skipped = incomplete + missing + running
            skipped_session_ids = [s.session_id for s in skipped]
            skipped_session_reasons = {s.session_id: str(s.status) for s in skipped}
            missing_result_files = [s.results_path for s in missing]

        total_sessions = len(results_snapshot)
        executed_session_ids = [r.session_id for r in results_snapshot]
        planned_sessions = None
        planned_session_ids = None
        if run_config.task_ids:
            planned_task_ids = list(run_config.task_ids)
            if run_config.num_tasks is not None:
                planned_task_ids = planned_task_ids[: int(run_config.num_tasks)]
            planned_sessions = len(planned_task_ids)
            planned_session_ids = [
                run_config.to_session_config(task_id).get_session_id() for task_id in planned_task_ids
            ]
        elif run_config.num_tasks is not None:
            planned_sessions = int(run_config.num_tasks)
        if planned_sessions is None:
            planned_sessions = total_sessions
        successful_sessions = sum(1 for r in results_snapshot if r.success)
        percent_successful = successful_sessions / total_sessions if total_sessions else None
        scores = [r.score for r in results_snapshot if r.score is not None]
        average_score = (sum(scores) / len(scores)) if scores else None

        finished_successful = sum(1 for r in results_snapshot if r.is_finished is True and bool(r.success))
        finished_unsuccessful = sum(1 for r in results_snapshot if r.is_finished is True and not bool(r.success))
        unfinished = sum(1 for r in results_snapshot if r.is_finished is False)
        errored = sum(1 for r in results_snapshot if r.is_finished is None)
        percent_finished_successful = finished_successful / total_sessions if total_sessions else None
        percent_finished_unsuccessful = finished_unsuccessful / total_sessions if total_sessions else None
        percent_unfinished = unfinished / total_sessions if total_sessions else None
        percent_error = errored / total_sessions if total_sessions else None
        percent_finished = (finished_successful + finished_unsuccessful) / total_sessions if total_sessions else None

        total_agent_cost = sum(r.agent_cost for r in results_snapshot) if total_sessions else 0.0
        total_benchmark_cost = sum(r.benchmark_cost for r in results_snapshot) if total_sessions else 0.0
        total_run_cost = total_agent_cost + total_benchmark_cost
        if results_snapshot:
            agent_reports = [r.cost_reports["agent"] for r in results_snapshot]
            benchmark_reports = [r.cost_reports["benchmark"] for r in results_snapshot]
            try:
                accumulated_agent_report = accumulate_reports(agent_reports)
            except ValueError:
                accumulated_agent_report = CostReport.initialize_empty()
                for report in agent_reports:
                    accumulated_agent_report.accumulate_from(report)
            try:
                accumulated_benchmark_report = accumulate_reports(benchmark_reports)
            except ValueError:
                accumulated_benchmark_report = CostReport.initialize_empty()
                for report in benchmark_reports:
                    accumulated_benchmark_report.accumulate_from(report)
        else:
            accumulated_agent_report = CostReport.initialize_empty()
            accumulated_benchmark_report = CostReport.initialize_empty()
        average_agent_cost = (total_agent_cost / total_sessions) if total_sessions else None
        average_benchmark_cost = (total_benchmark_cost / total_sessions) if total_sessions else None

        steps = [tr.steps for tr in results_snapshot]
        avg_steps = sum(steps) / len(steps) if steps else None
        action_counts = [tr.action_count for tr in results_snapshot]
        avg_action_count = sum(action_counts) / len(action_counts) if action_counts else None
        invalid_action_counts = [tr.invalid_action_count for tr in results_snapshot]
        avg_invalid_action_count = (
            sum(invalid_action_counts) / len(invalid_action_counts) if invalid_action_counts else None
        )
        total_action_count = sum(action_counts) if action_counts else 0
        total_invalid_action_count = sum(invalid_action_counts) if invalid_action_counts else 0
        avg_invalid_action_percent = (
            (total_invalid_action_count / total_action_count * 100) if total_action_count else None
        )

        bench_score: Optional[float] = bench_results_obj.score if bench_results_obj is not None else None

        model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")
        model_names = [str(model_name)] if model_name else None

        max_workers = run_config.max_workers
        if max_workers is None:
            max_workers_env = os.environ.get("EXGENTIC_MAX_WORKERS")
            if max_workers_env:
                try:
                    max_workers = int(max_workers_env)
                except ValueError:
                    max_workers = None

        bench_entry = get_benchmark_entries().get(run_config.benchmark)
        agent_entry = get_agent_entries().get(run_config.agent)
        bench_name = bench_entry.display_name if bench_entry is not None else run_config.benchmark
        agent_name = agent_entry.display_name if agent_entry is not None else run_config.agent

        results_obj = RunResults(
            benchmark_name=str(bench_name),
            benchmark_slug_name=str(run_config.benchmark),
            agent_name=str(agent_name),
            agent_slug_name=str(run_config.agent),
            model_name=str(model_name) if model_name is not None else None,
            model_names=model_names,
            subset_name=str(run_config.subset) if run_config.subset is not None else None,
            total_sessions=total_sessions,
            planned_sessions=planned_sessions,
            planned_session_ids=planned_session_ids,
            executed_session_ids=executed_session_ids,
            max_workers=max_workers,
            successful_sessions=successful_sessions,
            benchmark_score=bench_score,
            benchmark_results=(bench_results_obj.model_dump() if bench_results_obj is not None else None),
            average_score=average_score,
            average_agent_cost=average_agent_cost,
            total_agent_cost=total_agent_cost,
            average_benchmark_cost=average_benchmark_cost,
            total_benchmark_cost=total_benchmark_cost,
            total_run_cost=total_run_cost,
            session_results=results_snapshot,
            accumulated_agent_report=accumulated_agent_report,
            accumulated_benchmark_report=accumulated_benchmark_report,
            average_steps=avg_steps,
            average_action_count=avg_action_count,
            average_invalid_action_count=avg_invalid_action_count,
            average_invalid_action_percent=avg_invalid_action_percent,
            percent_finished=percent_finished,
            percent_successful=percent_successful,
            percent_finished_successful=percent_finished_successful,
            percent_finished_unsuccessful=percent_finished_unsuccessful,
            percent_unfinished=percent_unfinished,
            percent_error=percent_error,
            aggregation_mode="completed_only",
            completed_sessions=completed_sessions,
            incomplete_sessions=incomplete_sessions,
            missing_sessions=missing_sessions,
            running_sessions=running_sessions,
            aggregated_session_ids=aggregated_session_ids,
            skipped_session_ids=skipped_session_ids,
            skipped_session_reasons=skipped_session_reasons,
            missing_result_files=missing_result_files,
        )

        try:
            rp.results.parent.mkdir(parents=True, exist_ok=True)
            with open(rp.results, "w", encoding="utf-8") as f:
                json.dump(results_obj.model_dump(), f, ensure_ascii=False, indent=2)
            if bench_results_obj is not None:
                rp.benchmark_results.parent.mkdir(parents=True, exist_ok=True)
                with open(rp.benchmark_results, "w", encoding="utf-8") as f:
                    json.dump(
                        bench_results_obj.model_dump(),
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
        except OSError:
            # Allow aggregation in read-only output directories.
            pass
        return results_obj

    def _set_reason(self, session, reason: str, *, overwrite: bool = False) -> None:
        session_id = session.session_id if session else None
        if session_id is None:
            return
        with self._lock:
            data = self._sessions.get(session_id)
            if data is None:
                return
            if data.reason is not None and not overwrite:
                return
            data.reason = reason

    def _pop_reason(self, session) -> str:
        session_id = session.session_id if session else None
        if session_id is None:
            return "ended"
        with self._lock:
            data = self._sessions.get(session_id)
            if data is None or data.reason is None:
                return "ended"
            reason = data.reason
            data.reason = None
            return reason

    def _get_cost_snapshot(self, session_id: str, session) -> tuple[float, float]:
        with self._lock:
            data = self._sessions.get(session_id)
            agent = data.agent if data is not None else None
        agent_cost_report = CostReport.initialize_empty()
        benchmark_cost_report = CostReport.initialize_empty()
        if agent is not None:
            try:
                agent_cost_report = agent.get_cost()
            except Exception:
                agent_cost_report = CostReport.initialize_empty()
        try:
            benchmark_cost_report = session.get_cost()
        except Exception:
            benchmark_cost_report = CostReport.initialize_empty()
        return agent_cost_report.total_cost, benchmark_cost_report.total_cost


FileSystemObserver = ResultsObserver
