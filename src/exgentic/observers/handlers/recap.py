# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ...core.orchestrator.observer import Observer
from ..logging import get_logger


@dataclass(frozen=True)
class RunRecapData:
    total_sessions: int
    successful_sessions: int
    success_rate: float | None
    finished_sessions: int | None
    benchmark_score: Any
    average_score: Any
    run_cost: Any
    avg_agent_cost: Any
    total_agent_cost: Any
    total_benchmark_cost: Any
    average_steps: Any
    execution_time: timedelta
    results_path: Path


class RunRecapMixin:
    def _load_recap_data(self, results_path: Path, start_time: datetime | None) -> RunRecapData | None:
        if not results_path.exists():
            return None

        try:
            with open(results_path, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            return None

        total = results.get("total_sessions", 0)
        succ = results.get("successful_sessions", 0)
        success_rate = results.get("percent_successful")
        final = results.get("benchmark_score")
        avg = results.get("average_score")
        run_cost = results.get("total_run_cost")
        avg_agent_cost = results.get("average_agent_cost")
        total_agent_cost = results.get("total_agent_cost")
        total_benchmark_cost = results.get("total_benchmark_cost")
        avg_steps = results.get("average_steps")

        session_results = results.get("session_results") or []
        if session_results:
            total = len(session_results)
            succ = sum(1 for r in session_results if r.get("success"))
            if total:
                success_rate = succ / total
        finished = sum(1 for r in session_results if r.get("is_finished")) if session_results else None

        started = start_time or datetime.now()
        execution_time = datetime.now() - started
        return RunRecapData(
            total_sessions=total,
            successful_sessions=succ,
            success_rate=success_rate,
            finished_sessions=finished,
            benchmark_score=final,
            average_score=avg,
            run_cost=run_cost,
            avg_agent_cost=avg_agent_cost,
            total_agent_cost=total_agent_cost,
            total_benchmark_cost=total_benchmark_cost,
            average_steps=avg_steps,
            execution_time=execution_time,
            results_path=results_path,
        )

    def _format_money(self, value: Any) -> str:
        return f"${value:.4f}" if isinstance(value, (int, float)) else "-"


class RunRecapObserver(Observer, RunRecapMixin):
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
        self._start_time: datetime | None = None

    def _ensure_logger(self) -> None:
        if self._logger is not None:
            return
        rp = self.paths
        log_path = rp.tracker
        self._logger = get_logger(
            f"tracker.recap.{self._run_id}",
            str(log_path),
            console=self._console,
            propagate=False,
        )

    def on_run_start(self, run_config) -> None:
        self._ensure_logger()
        self._start_time = datetime.now()

    def on_run_success(self, results, run_config) -> None:
        self._log_recap()

    def on_run_error(self, error) -> None:
        self._log_recap()

    def _log_recap(self) -> None:
        self._ensure_logger()
        results_path = self.paths.results
        data = self._load_recap_data(results_path, self._start_time)
        if data is None:
            return

        finished_str = f" | Finished: {data.finished_sessions}" if data.finished_sessions is not None else ""
        success_rate_str = f" | Success%: {data.success_rate:.2%}" if data.success_rate is not None else ""

        final_str = f"{data.benchmark_score}" if data.benchmark_score is not None else "-"
        avg_str = f"{data.average_score}" if data.average_score is not None else "-"
        run_cost_str = self._format_money(data.run_cost)
        avg_agent_cost_str = self._format_money(data.avg_agent_cost)
        total_agent_cost_str = self._format_money(data.total_agent_cost)
        total_benchmark_cost_str = self._format_money(data.total_benchmark_cost)

        recap = (
            f"📊 Sessions: {data.total_sessions} | Successes: {data.successful_sessions}"
            f"{success_rate_str}{finished_str}\n"
            f"🏁 Scores: Final={final_str} | Avg={avg_str}\n"
            f"💰 Costs: Run={run_cost_str} | Avg Agent={avg_agent_cost_str} | "
            f"Agent Total={total_agent_cost_str} | Benchmark Total={total_benchmark_cost_str}\n"
            f"🐾 Average number of steps={data.average_steps}\n"
            f"🕐 Total execution time={data.execution_time}\n"
            f"📄 Results: {data.results_path}"
        )
        self._logger.info(recap)
