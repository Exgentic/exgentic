# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import threading
import time
from collections import deque
from queue import Queue
from typing import Any

from pydantic import BaseModel

from ...core.agent import Agent
from ...core.context import get_context
from ...core.orchestrator.observer import Observer
from ...core.orchestrator.termination import (
    AgentError,
    BenchmarkError,
    RunCancelError,
    SessionCancelError,
)
from ...core.types import BenchmarkResults, SessionScore
from ...utils.paths import RunPaths


class DashboardEventsObserver(Observer):
    """Emit dashboard-friendly run events to a queue."""

    def __init__(self) -> None:
        self.events: Queue[dict[str, Any]] = Queue(maxsize=10000)
        self._thread_session: dict[int, str] = {}
        self._session_steps: dict[str, int] = {}
        self._session_started_at: dict[str, float] = {}
        self._session_agents: dict[str, Agent] = {}
        self._events_lock = threading.Lock()
        self._pending_events: deque = deque()
        self._last_batch_time = time.time()
        self._batch_interval = 0.02
        self._pending_run_meta = True

    def _emit(self, etype: str, **payload: Any) -> None:
        evt = {"type": etype, "ts": time.time(), **payload}

        current_time = time.time()

        if etype in ("session_started", "session_finished", "run_meta", "saved"):
            try:
                self.events.put_nowait(evt)
            except Exception:
                try:
                    self.events.get_nowait()
                    self.events.put_nowait(evt)
                except Exception:
                    pass
            return

        with self._events_lock:
            if len(self._pending_events) >= 100:
                self._pending_events.popleft()
            self._pending_events.append(evt)

            if current_time - self._last_batch_time >= self._batch_interval or len(self._pending_events) >= 10:
                self._flush_batch()
                self._last_batch_time = current_time

    def _flush_batch(self) -> None:
        while self._pending_events:
            try:
                evt = self._pending_events.popleft()
                self.events.put_nowait(evt)
            except Exception:
                try:
                    self.events.get_nowait()
                    self.events.put_nowait(evt)
                except Exception:
                    break

    def on_run_start(self, run_config) -> None:
        self._emit_run_meta()

    def on_session_start(self, session, agent: Agent, observation) -> None:
        sid = session.session_id
        tid = threading.get_ident()
        with self._events_lock:
            if tid is not None:
                self._thread_session[int(tid)] = sid
            self._session_steps[sid] = 0
            self._session_started_at[sid] = time.time()
            self._session_agents[sid] = agent
        self._emit("session_started", session_id=sid)

    def on_react_success(self, session, action) -> None:
        step_n = None
        sid = None
        tid = threading.get_ident()
        if tid is not None:
            with self._events_lock:
                sid = self._thread_session.get(int(tid))
                if sid:
                    step_n = self._session_steps.get(sid, 0) + 1
                    self._session_steps[sid] = step_n

        def _action_summary(a: Any) -> str:
            try:
                from ...core.types import ParallelAction, SingleAction

                if isinstance(a, SingleAction):
                    return f"{a.name}"[:100]
                if isinstance(a, ParallelAction):
                    return f"parallel[{len(a.actions)} actions]"
            except Exception:
                pass
            return str(a)[:100]

        def _action_obj(a: Any):
            try:
                from ...core.types import ParallelAction, SingleAction

                if isinstance(a, SingleAction):
                    args = a.arguments.model_dump()
                    if isinstance(args, dict) and len(str(args)) > 500:
                        args = {k: (v if len(str(v)) < 50 else f"{str(v)[:50]}...") for k, v in list(args.items())[:5]}
                    return {"type": "single", "name": a.name, "arguments": args}
                if isinstance(a, ParallelAction):
                    items = []
                    for i, x in enumerate(a.actions[:5]):
                        try:
                            args = x.arguments.model_dump()
                            if isinstance(args, dict) and len(str(args)) > 200:
                                args = {
                                    k: (v if len(str(v)) < 30 else f"{str(v)[:30]}...")
                                    for k, v in list(args.items())[:3]
                                }
                            items.append({"name": x.name, "arguments": args})
                        except Exception:
                            items.append(str(x)[:100])
                        if i >= 4:
                            break
                    return {"type": "parallel", "actions": items}
            except Exception:
                return None
            return None

        if sid:
            agent_cost, benchmark_cost = self._get_cost_snapshot(sid, session)
            execution_time = self._get_execution_time(sid)
            self._emit(
                "step",
                event="action",
                session_id=sid,
                n=step_n,
                action=_action_summary(action),
                action_obj=_action_obj(action),
                execution_time=execution_time,
                agent_cost=agent_cost,
                benchmark_cost=benchmark_cost,
            )

    def on_step_success(self, session, observation) -> None:
        sid = None
        tid = threading.get_ident()
        if tid is not None:
            with self._events_lock:
                sid = self._thread_session.get(int(tid))

        def _safe_json(obj: Any) -> str:
            try:
                if isinstance(obj, BaseModel):
                    data = obj.model_dump()
                    if isinstance(data, dict) and len(str(data)) > 1000:
                        truncated = {
                            k: (v if len(str(v)) < 100 else f"{str(v)[:100]}...") for k, v in list(data.items())[:10]
                        }
                        return json.dumps(truncated, ensure_ascii=False)
                    return json.dumps(data, ensure_ascii=False)
                obj_str = str(obj)
                if len(obj_str) > 1000:
                    obj_str = obj_str[:1000] + "..."
                return json.dumps(obj_str, default=str, ensure_ascii=False)
            except Exception:
                return str(obj)[:500]

        def _obs_obj(o: Any):
            try:
                if isinstance(o, BaseModel):
                    data = o.model_dump()
                    if isinstance(data, dict) and len(str(data)) > 1000:
                        truncated = {
                            k: (v if len(str(v)) < 100 else f"{str(v)[:100]}...") for k, v in list(data.items())[:10]
                        }
                        return truncated
                    return data
                safe_json = _safe_json(o)
                return json.loads(safe_json)
            except Exception:
                return None

        if sid:
            step_n = self._session_steps.get(sid, 0)
            agent_cost, benchmark_cost = self._get_cost_snapshot(sid, session)
            execution_time = self._get_execution_time(sid)
            self._emit(
                "observation",
                event="observation",
                session_id=sid,
                step=step_n,
                observation=_obs_obj(observation),
                initial=False,
                execution_time=execution_time,
                agent_cost=agent_cost,
                benchmark_cost=benchmark_cost,
            )

    def on_session_success(self, session, score: SessionScore, agent: Agent) -> None:
        sid = session.session_id
        self._flush_pending()
        self._cleanup_thread_session()
        success = bool(score.success)
        value = score.score
        details = score.model_dump()
        with self._events_lock:
            steps = self._session_steps.pop(sid, 0)
            started_at = self._session_started_at.pop(sid, None)
            self._session_agents.pop(sid, None)
        execution_time = time.time() - started_at if started_at is not None else None
        agent_cost = 0.0
        benchmark_cost = 0.0
        if agent is not None:
            try:
                report = agent.get_cost()
                agent_cost = float(report.total_cost)
            except Exception:
                agent_cost = 0.0
        try:
            report = session.get_cost()
            benchmark_cost = float(report.total_cost)
        except Exception:
            benchmark_cost = 0.0
        self._emit(
            "session_finished",
            session_id=sid,
            success=success,
            score=value,
            details=details,
            steps=steps,
            execution_time=execution_time,
            agent_cost=agent_cost,
            benchmark_cost=benchmark_cost,
            is_finished=score.is_finished,
        )

    def on_session_error(self, session, error) -> None:
        sid = session.session_id
        self._flush_pending()
        self._cleanup_thread_session()
        error_source = None
        if isinstance(error, AgentError):
            error_source = "agent"
        elif isinstance(error, BenchmarkError):
            error_source = "benchmark"
        elif isinstance(error, (SessionCancelError, RunCancelError, KeyboardInterrupt)):
            error_source = "cancelled"
        root_error = error.error if isinstance(error, (AgentError, BenchmarkError)) else None
        error_message = str(root_error) if root_error else str(error)
        details = {"error": error_message}
        if error_source is not None:
            details["error_source"] = error_source
        with self._events_lock:
            steps = self._session_steps.pop(sid, 0)
            started_at = self._session_started_at.pop(sid, None)
            self._session_agents.pop(sid, None)
        execution_time = time.time() - started_at if started_at is not None else None
        agent_cost = 0.0
        benchmark_cost = 0.0
        try:
            report = session.get_cost()
            benchmark_cost = float(report.total_cost)
        except Exception:
            benchmark_cost = 0.0
        self._emit(
            "session_finished",
            session_id=sid,
            success=False,
            score=None,
            details=details,
            steps=steps,
            execution_time=execution_time,
            agent_cost=agent_cost,
            benchmark_cost=benchmark_cost,
            is_finished=None,
            error_source=error_source,
        )

    def _get_execution_time(self, session_id: str) -> float | None:
        with self._events_lock:
            started_at = self._session_started_at.get(session_id)
        if started_at is None:
            return None
        return time.time() - started_at

    def _get_cost_snapshot(self, session_id: str, session) -> tuple[float, float]:
        with self._events_lock:
            agent = self._session_agents.get(session_id)
        agent_cost = 0.0
        benchmark_cost = 0.0
        if agent is not None:
            try:
                report = agent.get_cost()
                agent_cost = float(report.total_cost)
            except Exception:
                agent_cost = 0.0
        try:
            report = session.get_cost()
            benchmark_cost = float(report.total_cost)
        except Exception:
            benchmark_cost = 0.0
        return agent_cost, benchmark_cost

    def on_run_success(self, results: BenchmarkResults, run_config) -> None:
        payload = results.model_dump()
        self._emit("benchmark_recorded", results=payload)
        self._emit_saved_from_context()

    def on_run_error(self, error) -> None:
        self._emit_saved_from_context()

    def emit_saved(self, path: str) -> None:
        self._flush_pending()
        self._emit("saved", path=path)

    def _flush_pending(self) -> None:
        with self._events_lock:
            if self._pending_events:
                self._flush_batch()

    def _cleanup_thread_session(self) -> None:
        tid = threading.get_ident()
        if tid is not None:
            with self._events_lock:
                self._thread_session.pop(int(tid), None)

    def _emit_run_meta(self) -> None:
        if not self._pending_run_meta:
            return
        try:
            run_id = get_context().run_id
        except RuntimeError:
            return
        self._emit("run_meta", run_id=run_id)
        self._pending_run_meta = False

    def _emit_saved_from_context(self) -> None:
        try:
            ctx = get_context()
        except RuntimeError:
            return
        results_path = RunPaths.from_context(ctx).results
        self.emit_saved(str(results_path))
