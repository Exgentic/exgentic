# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""AppWorld evaluator and session classes.

These classes import the ``appworld`` package at method level.  They are only
ever instantiated inside the isolated runner subprocess, so the heavy
dependency is never required in the host process.

The light ``AppWorldBenchmark`` class lives in ``appworld_benchmark.py`` and
must remain importable without the ``appworld`` package installed.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    create_model,
)

from ...core.actions import ActionsHandler
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    EmptyObservation,
    FinishAction,
    MessageAction,
    Observation,
    SessionIndex,
    SessionScore,
    SingleAction,
)
from ...utils.paths import get_run_id, get_run_paths
from ...utils.settings import get_settings
from .appworld_benchmark import AppWorldObservation

settings = get_settings()
logger = logging.getLogger(__name__)

APPWORLD_TOTAL_TASKS = {
    "train": 90,
    "dev": 57,
    "test_normal": 168,
    "test_challenge": 417,
}

if TYPE_CHECKING:
    from appworld.environment import AppWorld  # type: ignore


class AppWorldSession(Session):
    """Session that hosts AppWorld directly (no separate WorldProcess).

    Tools are derived from the AppWorld task API docs and mapped to ActionTypes.
    Each step calls AppWorld.requester.request(app, api, **args) and wraps the
    result as observations, surfacing API errors as structured payloads.
    """

    CACHE_DIR: ClassVar[str] = "./appworld_disk_cache"
    TASK_OUTPUT_SUBDIR: ClassVar[str] = "task_output"
    SCORES_FILE_NAME: ClassVar[str] = "scores.json"

    def __init__(
        self,
        task_id: str,
        session_id: str | None = None,
        env_kwargs: dict[str, Any] | None = None,
        use_cache: bool = True,
        tool_name_separator: str = ".",
        max_interactions: int | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id
        self._env_kwargs = env_kwargs or {}
        self._tool_name_separator = tool_name_separator
        self._max_interactions = max_interactions
        self._action_count = 0
        self._registry = ActionsHandler(
            logger=self.logger,
            warn_on_validation_error=False,
            warn_on_unknown_action=False,
            handle_validation_error=lambda action, _msg: self.apply_action(action),
            handle_unknown_action=self.apply_action,
        )
        self._step_count: int = 0
        self._done: bool = False
        self._world_closed: bool = False
        self._cached_score: SessionScore | None = None
        self._task_id = str(task_id)

        # Build experiment_name from run_id + session_id for AppWorld output isolation.
        self._env_kwargs.setdefault("experiment_name", f"{get_run_id()}__{self.session_id}")
        self._env_kwargs.setdefault("max_interactions", max_interactions)

        # Construct AppWorld in-process (lazy import to defer side effects)
        from appworld import update_root  # type: ignore
        from appworld.common.constants import DEFAULT_EXPERIMENT_NAME  # type: ignore
        from appworld.common.path_store import path_store  # type: ignore
        from appworld.environment import AppWorld  # type: ignore

        # Point appworld at the correct data directory before loading the task.
        cache = Path(settings.cache_dir).expanduser()
        update_root(str(cache / "appworld"))

        # Patch appworld's SQLite connection helper to allow cross-thread usage.
        # The venv runner serves via uvicorn which may dispatch requests across
        # threads, but appworld's @lru_cache'd connections default to
        # check_same_thread=True, causing ProgrammingError.
        self._patch_appworld_sqlite()

        self._world: AppWorld = AppWorld(task_id=self._task_id, **self._env_kwargs)
        self._experiment_name: str = self._world.experiment_name or DEFAULT_EXPERIMENT_NAME
        self._task_output_dir: Path = (
            Path(path_store.experiment_outputs) / self._experiment_name / "tasks" / self._task_id
        )

        self.logger.info(f"Task ID: {task_id}")
        super().__init__()

    @staticmethod
    def _patch_appworld_sqlite() -> None:
        """Patch appworld's SQLite helpers to tolerate cross-thread access.

        appworld's ``get_direct_sqlite3_connection`` creates connections with
        the default ``check_same_thread=True``, then caches them via
        ``@lru_cache``.  When the uvicorn runner dispatches requests on
        different threads, reusing those connections raises
        ``sqlite3.ProgrammingError``.  We patch the function and clear the
        cache so fresh connections are created with ``check_same_thread=False``.
        """
        import sqlite3 as _sqlite3

        from appworld.apps.lib.models import db as _appworld_db  # type: ignore

        _original = _appworld_db.get_direct_sqlite3_connection

        if getattr(_original, "_exgentic_patched", False):
            return

        def _safe_connect(db_app_path: str) -> _sqlite3.Connection:
            conn = _sqlite3.connect(db_app_path, check_same_thread=False)
            conn.execute("PRAGMA mmap_size = 268435456")
            return conn

        _safe_connect._exgentic_patched = True  # type: ignore[attr-defined]
        _appworld_db.get_direct_sqlite3_connection = _safe_connect
        # Clear the lru_cache so stale thread-bound connections aren't reused.
        _appworld_db.get_direct_cached_sqlite3_connection.cache_clear()

    def get_config(self) -> dict[str, Any]:
        return {
            "task_id": self._task_id,
            "env_kwargs": self._env_kwargs,
            "tool_name_separator": self._tool_name_separator,
            "max_interactions": self._max_interactions,
        }

    @property
    def world(self) -> AppWorld:
        return self._world

    @property
    def task(self) -> str:
        self.logger.info(f"Task: {self.world.task.instruction}")
        return "Task from supervisor:\n" + self.world.task.instruction

    @property
    def context(self) -> dict[str, Any]:
        if self.world.task is None:
            raise ValueError("AppWorld task is not initialized")

        allowed = ", ".join(app for app in self.world.task.allowed_apps if app != "api_docs")
        return {
            "policy": (
                "This environment provides a set of applications,"
                " each exposing a predefined set of APIs that may"
                " be used to perform tasks on behalf of the"
                " supervisor. The applications include:"
                f" {allowed}.\n"
                " The available applications and their APIs are"
                " fixed for the task.\n"
                "\n"
                "Supervisor account credentials (such as emails,"
                " usernames, and passwords) are available through"
                " the supervisor application's APIs and are"
                " accessed from there when required.\n"
                "\n"
                "If an application requires an access token to"
                " perform authenticated operations, the access"
                " token is obtained by calling that application's"
                " authentication/login API using the credentials"
                " retrieved from the supervisor application."
                " Access tokens are not provided by the supervisor"
                " application.\n"
                "\n"
                "References to people (e.g., friends, family,"
                " roommates) correspond to entries in the"
                " phone_contacts application.\n"
                "References to files or storage correspond to the"
                " file_system application, not the local machine"
                " filesystem.\n"
                "\n"
                "Time-based instructions (e.g., 'this month',"
                " 'yesterday') are interpreted with full calendar"
                " boundary ranges.\n"
                "If an API returns paginated results, all pages"
                " constitute the complete result.\n"
                "\n"
                "The environment consists only of the provided"
                " applications and their documented APIs and"
                " parameters. No additional endpoints, methods,"
                " arguments, or capabilities are assumed beyond"
                " those explicitly defined.\n"
                "\n"
                "When task execution is finished, the designated"
                " task-completion API is used to signal completion."
                " If the task requires a final answer value, the"
                " answer is returned through that completion API."
                " If the task cannot be completed using the"
                " available applications and APIs, the task may be"
                " marked as failed."
            ),
            "supervisor": dict(self.world.task.supervisor),
            # "app_descriptions": self.world.task.app_descriptions,
            # "allowed_apps": self.world.task.allowed_apps,
            "datetime": self.world.task.datetime.isoformat(),
        }

    @property
    def actions(self) -> list[ActionType]:
        if not self._registry.actions:
            # Build ActionTypes from AppWorld function_calling docs to leverage enriched auth parameters
            from ...adapters.schemas.json_schema import make_args_model_from_json_schema

            tools_specs = self.world.task.api_docs.function_calling()
            for tool in tools_specs:
                function = tool["function"]
                raw_name = function["name"]
                separator = self._tool_name_separator
                name = raw_name.replace("__", separator)
                app, api = name.split(separator, 1)
                if app == "api_docs":
                    continue
                if api == "show_active_task":
                    continue

                args_model = make_args_model_from_json_schema(name, function["parameters"])

                if raw_name == "supervisor__complete_task":
                    finish_act = create_model(
                        "AppWorldFinishAction",
                        __base__=FinishAction,
                        arguments=(args_model, ...),
                    )
                    self._registry.add_action(
                        name="finish",
                        description=function["description"],
                        action_cls=finish_act,
                        handler=self.apply_action,
                        is_finish=True,
                    )
                else:
                    act = create_model(
                        f"{name}_Action",
                        __base__=SingleAction,
                        name=(Literal[name], name),
                        arguments=(args_model, ...),
                    )
                    self._registry.add_action(
                        name=name,
                        description=function["description"],
                        action_cls=act,
                        handler=self.apply_action,
                    )
        return self._registry.actions

    @property
    def task_id(self) -> str:
        return str(self._task_id)

    @property
    def _actions_names(self) -> set[str]:
        return {a.name for a in self.actions}

    def _to_observation(self, raw: Any, invoking: list[SingleAction] | None = None) -> Observation:
        return AppWorldObservation(invoking_actions=invoking or [], result=raw)

    def start(self) -> Observation | None:
        self.logger.info(f"session_start id={self.session_id} task_id={self._task_id}")
        # Empty initial observation; task details are provided via task/context.
        return EmptyObservation()

    def _is_message_action(self, action: SingleAction) -> bool:
        if isinstance(action, MessageAction) or action.name == "message":
            return True
        return False

    def apply_action(self, action: SingleAction):
        if self._is_message_action(action):
            self._step_count += 1
            return AppWorldObservation(
                invoking_actions=[action],
                result="Error: Sending a message is not allowed. Please use only one of the available actions.",
            )
        # if action.name not in self._actions_names:
        #     return AppWorldObservation(invoking_actions=[action], result="Wrong name: {action.name}")
        # Map benchmark-level finish to the supervisor.complete_task endpoint
        effective_name = action.name
        if action.name == "finish":
            separator = self._tool_name_separator
            effective_name = f"supervisor{separator}complete_task"

        separator = self._tool_name_separator
        parts = effective_name.split(separator, 1)
        if len(parts) != 2:
            parts = effective_name, ""
        app_name, api_name = parts

        arguments = action.arguments
        if isinstance(arguments, BaseModel):
            arguments = arguments.model_dump()

        self.logger.info(f"App: {app_name}, Function: {api_name}, Arguments: {arguments}")

        try:
            out = self.world.requester.request(app_name, api_name, **arguments)
        except Exception as e:
            try:
                e = json.loads(str(e).split("\n")[-1])["message"]
            except json.JSONDecodeError:
                pass
            out = "Error: " + str(e)
        finally:
            self._step_count += 1

        self.logger.info(f"Output: {out}")

        return AppWorldObservation(
            invoking_actions=[action],
            result=out,
        )

    @staticmethod
    def _max_interactions_error(observation: Observation) -> bool:
        for obs in observation.to_observation_list():
            result = obs.result
            if isinstance(result, str) and "Maximum number of executions" in result:
                return True
        return False

    def step(self, action: Action) -> Observation | None:
        if self._done:
            return None

        if self._max_interactions is not None:
            incoming = len(action.to_action_list())
            if self._action_count + incoming > self._max_interactions:
                self.logger.warning(
                    "AppWorld local max_interactions reached (%s/%s); terminating session",
                    self._action_count,
                    self._max_interactions,
                )
                return None

        observation = self._registry.execute(action)
        if observation is None:
            return None
        if self._max_interactions is not None:
            self._action_count += len(action.to_action_list())
        if self._max_interactions_error(observation):
            self.logger.warning(
                "AppWorld max_interactions reached (%s/%s); terminating session",
                self.world.num_interactions,
                self.world.max_interactions,
            )
            return None
        return observation

    def done(self) -> bool:
        return self.world.task_completed()

    def score(self) -> SessionScore:
        if self._cached_score is not None:
            return self._cached_score
        # World was already saved and closed in close(); compute the actual evaluation score now.
        from appworld.apps.lib.models.db import CachedDBHandler
        from appworld.evaluator import evaluate_task

        test_tracker = evaluate_task(
            task_id=self._task_id,
            experiment_name=self._experiment_name,
            suppress_errors=True,
            save_report=False,
        )
        score_value = float(test_tracker.pass_percentage) / 100.0
        self.logger.info(
            "Evaluation results: pass_percentage=%s pass_count=%s fail_count=%s num_tests=%s success=%s",
            test_tracker.pass_percentage,
            test_tracker.pass_count,
            test_tracker.fail_count,
            test_tracker.num_tests,
            test_tracker.success,
        )

        # Finished when close() already captured task completion.
        finished = bool(self._done)
        # Reset cached DB handler for this task so later aggregate evaluation can run.
        CachedDBHandler.reset(self._task_id)
        # Surface benchmark evaluation details for downstream analysis.
        session_metrics = {
            "pass_percentage": test_tracker.pass_percentage,
            "pass_count": test_tracker.pass_count,
            "fail_count": test_tracker.fail_count,
            "num_tests": test_tracker.num_tests,
            "difficulty": test_tracker.difficulty,
            "success": test_tracker.success,
        }
        tracker_dict = test_tracker.to_dict(stats_only=False)
        scores_path = self.paths.benchmark_dir / self.SCORES_FILE_NAME
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_id": self._task_id,
                    "session_id": self.session_id,
                    "test_tracker": tracker_dict,
                    **session_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        session_metadata = {"test_tracker": tracker_dict}
        return SessionScore(
            score=score_value,
            success=test_tracker.success,
            is_finished=finished,
            session_metrics=session_metrics,
            session_metadata=session_metadata,
        )

    def close(self):
        # Save AppWorld task state and mirror logs
        self.logger.info(
            "Closing AppWorld session: steps=%s done=%s world_closed=%s",
            self._step_count,
            self._done,
            self._world_closed,
        )
        if self._world_closed:
            self.logger.warning("AppWorld session close called more than once.")
        try:
            self.world.save()
        except Exception:
            self.logger.exception("AppWorld world.save failed")
            raise
        try:
            self._done = self.world.task_completed()
        except Exception:
            self.logger.exception("AppWorld task_completed check failed")
            raise
        logs_src = self._task_output_dir / "logs"
        if logs_src.exists():
            dest = self.paths.benchmark_dir / "logs"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                copytree(logs_src, dest, dirs_exist_ok=True)
            except Exception:
                self.logger.exception("AppWorld log copy failed")
                raise
        task_output_src = self._task_output_dir
        if task_output_src.exists():
            task_output_dest = self.paths.benchmark_dir / self.TASK_OUTPUT_SUBDIR
            try:
                if task_output_dest.exists():
                    shutil.rmtree(task_output_dest)
                copytree(task_output_src, task_output_dest)
            except Exception:
                self.logger.exception("AppWorld task output copy failed")
                raise
        # Write a standardized results.json plus AppWorld-specific fields
        try:
            sc = self.score()
        except Exception:
            self.logger.exception("AppWorld evaluation failed")
            raise
        self._cached_score = sc
        try:
            self.save_results(
                {
                    "score": sc.score,
                    "success": sc.success,
                    "session_id": self.session_id,
                    "task_id": self._task_id,
                    "completed": self._done,
                    "steps": self._step_count,
                }
            )
        except Exception:
            self.logger.exception("AppWorld save_results failed")
            raise
        self.logger.info(
            "Session Finished | Success: %s, steps=%s, score=%s",
            self._done,
            self._step_count,
            sc.score,
        )
        try:
            self.world.close()
        except Exception:
            self.logger.exception("AppWorld world.close failed")
            raise
        self._world_closed = True
        experiment_root = self._task_output_dir.parent.parent
        if experiment_root.name == self._experiment_name:
            try:
                shutil.rmtree(experiment_root)
            except Exception:
                self.logger.exception("AppWorld temp experiment cleanup failed")


class AppWorldEvaluator(Evaluator):
    """Evaluator for AppWorld -- task discovery, session config, and aggregation."""

    def __init__(self, subset: str = "test_normal") -> None:
        self._subset = subset

    def _ensure_appworld_root(self) -> None:
        from appworld import update_root  # type: ignore

        cache = Path(settings.cache_dir).expanduser()
        root = str(cache / "appworld")
        update_root(root)

    def list_tasks(self) -> list[str]:
        from appworld.task import load_task_ids  # type: ignore

        self._ensure_appworld_root()
        items: list[str] | None = load_task_ids(self._subset)
        if not items:
            return []
        return [str(t) for t in items]

    def _stage_task_outputs(
        self,
        *,
        task_ids: list[str],
        task_to_session: dict[str, str],
        temp_output_dir: Path,
    ) -> None:
        run_paths = get_run_paths()
        for task_id in task_ids:
            session_id = task_to_session.get(task_id)
            if not session_id:
                raise FileNotFoundError(f"Missing session mapping for AppWorld task '{task_id}'.")
            session_task_output = run_paths.session(session_id).benchmark_dir / AppWorldSession.TASK_OUTPUT_SUBDIR
            if not session_task_output.exists():
                raise FileNotFoundError(
                    f"Missing staged task output for task='{task_id}' session='{session_id}' at {session_task_output}"
                )
            dest_task_dir = temp_output_dir / "tasks" / task_id
            dest_task_dir.parent.mkdir(parents=True, exist_ok=True)
            copytree(session_task_output, dest_task_dir)

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        from appworld.evaluator import Metric, TestTracker  # type: ignore

        self._ensure_appworld_root()

        if not sessions:
            return BenchmarkResults(
                benchmark_name="appworld",
                total_tasks=0,
                score=0.0,
                metrics={},
            )

        run_paths = get_run_paths()
        task_id_to_test_tracker: dict[str, TestTracker] = {}
        for session in sessions:
            task_id = str(session.task_id)
            results_path = run_paths.session(session.session_id).results
            scores_path = run_paths.session(session.session_id).benchmark_dir / AppWorldSession.SCORES_FILE_NAME

            if scores_path.exists():
                with open(scores_path, encoding="utf-8") as f:
                    scores_payload = json.load(f)
                tracker = scores_payload.get("test_tracker")
            else:
                logger.warning(
                    "Missing AppWorld scores file for task_id=%s session_id=%s at %s; "
                    "falling back to session metadata in %s",
                    task_id,
                    session.session_id,
                    scores_path,
                    results_path,
                )
                with open(results_path, encoding="utf-8") as f:
                    payload = json.load(f)
                tracker = (payload.get("details") or {}).get("session_metadata", {}).get("test_tracker")
            if not isinstance(tracker, dict):
                raise ValueError(
                    "Missing test_tracker in aggregation source for "
                    f"task_id={task_id} session_id={session.session_id}. "
                    f"Checked {scores_path} and fallback {results_path}."
                )

            task_id_to_test_tracker[task_id] = TestTracker.from_dict(tracker, suppress_errors=False)

        evaluation_dict = Metric.compute_metrics(task_id_to_test_tracker, include_details=True)
        report = Metric.build_report(evaluation_dict)
        return BenchmarkResults(
            benchmark_name="appworld",
            total_tasks=len(sessions),
            score=evaluation_dict["aggregate"]["task_goal_completion"] / 100,
            metrics=report,
        )
