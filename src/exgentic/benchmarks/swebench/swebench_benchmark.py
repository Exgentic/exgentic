# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import logging
import subprocess
import textwrap
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core import Benchmark
from ...core.actions import ActionsHandler
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    FinishAction,
    Observation,
    SessionIndex,
    SingleAction,
    SingleObservation,
)
from ...utils.logging import hook_loggers_into_session
from ...utils.paths import get_run_id
from ...utils.settings import ExgenticSettings, get_settings
from . import swebench_evaluation, swebench_logs, swebench_metrics

# =============================================================================
# Configuration
# =============================================================================

_CONFIG: dict[str, Any] = None


def get_config() -> dict[str, Any]:
    import yaml

    global _CONFIG
    if _CONFIG is None:
        path = Path(__file__).parent / "config.yaml"
        _CONFIG = yaml.safe_load(path.read_text())
    return _CONFIG


# =============================================================================
# Actions
# =============================================================================


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to execute")


class BashAction(SingleAction):
    name: Literal["bash"] = "bash"
    arguments: BashArgs


class SubmitPatchArgs(BaseModel):
    summary: str = Field(description="Brief textual summary of the fix (no diff/patch)")


class SubmitPatchAction(FinishAction):
    name: Literal["finish"] = "finish"
    arguments: SubmitPatchArgs


# =============================================================================
# Action Handlers
# =============================================================================


def run_bash(
    command: str,
    env,
    timeout: int | None = None,
    size_limit: int | None = None,
    timeout_template: str = "",
) -> dict[str, Any]:
    """Execute bash command in environment."""
    try:
        output = env.execute(command=command, timeout=timeout or env.config.timeout)
    except subprocess.TimeoutExpired as e:
        exception_output = e.output.decode("utf-8", errors="replace") if e.output else ""
        msg = textwrap.dedent(timeout_template).format_map({"command": command, "exception_output": exception_output})
        output = {"output": msg, "returncode": 124}

    # Truncate large outputs
    if size_limit and len(output["output"]) > size_limit:
        head, tail = size_limit // 2, size_limit - size_limit // 2
        omitted = f"\n\n- OMITTED {len(output['output']) - size_limit} chars -\n\n"
        output["output"] = output["output"][:head] + omitted + output["output"][-tail:]

    return output


def generate_patch(env, cwd: str, base_commit: str) -> str:
    """Generate patch from staged changes."""
    command = f"git add -A && git diff --staged {base_commit} | cat"
    output = env.execute(command=command, cwd=cwd)
    return output["output"]


# =============================================================================
# Session
# =============================================================================


class SWEBenchSession(Session):
    """Session for a single SWE-bench task."""

    def __init__(
        self,
        settings: ExgenticSettings,
        instance: dict[str, Any],
        subset: str,
        max_interactions: int | None = None,
        require_submit_for_patch_evaluation: bool = True,
        session_id: str | None = None,
    ) -> None:
        cfg = get_config()

        self._instance = instance
        self._subset = subset
        self._instance_id = instance["instance_id"]
        if session_id is not None:
            self._session_id = session_id

        self._registry = ActionsHandler(
            logger=self.logger,
            warn_on_validation_error=False,
            warn_on_unknown_action=True,
            handle_validation_error=None,
            handle_unknown_action=None,
        )

        # State
        self._step_count = 0
        self._done = False
        self._max_interactions = max_interactions
        self._require_submit_for_patch_evaluation = require_submit_for_patch_evaluation
        self._action_count = 0
        self._score = None
        self._final_patch: str | None = None

        # Environment (set in start())
        self.env = None
        from swebench.harness.constants import DOCKER_WORKDIR

        self.container_repo_dir = DOCKER_WORKDIR
        self.container_base_commit: str | None = None

        # Config
        self._timeout = cfg["session"]["timeout"]
        self._environment_pull_timeout = int(cfg["session"].get("environment_pull_timeout", 600))
        self._observation_size_limit = cfg["session"]["observation_size_limit"]
        self._timeout_template = cfg["session"]["timeout_template"]
        self._task_prompt = cfg["task_prompt"]
        self._eval_config = cfg["evaluation"]

        self.logger.info(
            f"INIT | Session initialized | dataset: {self._subset:<30} "
            f"| instance_id: {self._instance_id:<20} | repo: {self._instance['repo']}"
        )

        hook_loggers_into_session(
            self.logger,
            logger_names=cfg["logging"].get("hook_loggers", []),
            level=logging._nameToLevel.get(cfg["logging"]["log_level"], logging.INFO),
        )

        # Call parent to save session manifest (session.json)
        super().__init__()

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> Observation | None:
        self.logger.info("START | Session start")
        self._setup_environment()
        return SingleObservation(invoking_actions=[], result=None)

    def done(self) -> bool:
        return self._done

    def close(self):
        self.logger.info("CLOSE | Session closing...")
        super().close()
        self.logger.debug("CLOSE | Cleaning agent environment")
        del self.env
        self.logger.info("CLOSE | Session closed")

    # -------------------------------------------------------------------------
    # Step Execution
    # -------------------------------------------------------------------------

    def _handle_bash(self, action: BashAction) -> str:
        self.logger.info(f"STEP | {self._step_count:<3} | ACTION | bash | command: {action.arguments.command}")
        result = run_bash(
            command=action.arguments.command,
            env=self.env,
            timeout=self._timeout,
            size_limit=self._observation_size_limit,
            timeout_template=self._timeout_template,
        )
        self.logger.info(
            f"STEP | {self._step_count:<3} | RESULT | bash | "
            f"returncode: {result['returncode']} | output_len: {len(result['output'])}"
        )
        return result

    def _handle_submit_patch(self, action: SubmitPatchAction) -> str:
        self.logger.info(f"STEP | {self._step_count:<3} | ACTION | submit_patch | summary: {action.arguments.summary}")
        self._final_patch = generate_patch(
            env=self.env,
            cwd=self.container_repo_dir,
            base_commit=self.container_base_commit,
        )
        self._done = True
        return None

    def step(self, action: Action) -> Observation | None:
        if self._done:
            return None

        if self._max_interactions is not None:
            incoming = len(action.to_action_list())
            if self._action_count + incoming > self._max_interactions:
                self.logger.warning(
                    "STEP | {self._step_count:<3} | Max interactions reached (%s/%s); terminating session",
                    self._action_count,
                    self._max_interactions,
                )
                return None

        self._step_count += 1
        observation = self._registry.execute(action)
        if observation is None:
            return None

        # Post-execute: always track action count (useful for metrics)
        self._action_count += len(action.to_action_list())

        return observation

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def score(self) -> swebench_logs.SessionScore:
        """Compute and cache the session score."""
        if self._score is not None:
            return self._score

        self.logger.info("SCORE | Computing session score")

        harness_result = None
        if self._done or not self._require_submit_for_patch_evaluation:
            if self._final_patch is None and not self._done:
                self.logger.info("SCORE | submit_patch not called; generating patch from current workspace")
                self._final_patch = self._generate_current_patch()
            harness_result = self._run_harness()
        else:
            self.logger.info(
                "SCORE | Skipping harness - submit_patch not called (require_submit_for_patch_evaluation=true)"
            )

        # Flush logs before parsing
        for handler in self.logger.handlers:
            handler.flush()

        # Build score
        self._score = swebench_logs.build_score(
            paths=self.paths,
            num_actions=self._action_count,
            max_interactions=self._max_interactions,
            instance_id=self._instance_id,
            harness_data=harness_result,
        )

        # Finished when agent called submit_patch (graceful completion)
        self._score.is_finished = self._done

        results_path = self.paths.benchmark_results
        swebench_logs.write_results(results_path, self._score)
        self.logger.info(
            "SCORE | Final | success=%s | score=%s | is_finished=%s",
            self._score.success,
            self._score.score,
            self._score.is_finished,
        )

        return self._score

    def _run_harness(self) -> swebench_evaluation.HarnessResult | None:
        """Run harness evaluation, handling errors."""
        if self._final_patch is None:
            self.logger.info("SCORE | Harness evaluation skipped: no patch available for evaluation")
            return None
        try:
            return swebench_evaluation.run_harness(
                patch=self._final_patch,
                instance_id=self._instance_id,
                subset=self._subset,
                paths=self.paths,
                eval_config=self._eval_config,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.exception(f"SCORE | Harness evaluation failed: {e}")
            return None

    def _generate_current_patch(self) -> str | None:
        """Generate patch from current working tree for non-submit evaluation mode."""
        if self.env is None or self.container_base_commit is None:
            self.logger.warning("SCORE | Cannot generate patch: environment/base commit unavailable")
            return None
        try:
            return generate_patch(
                env=self.env,
                cwd=self.container_repo_dir,
                base_commit=self.container_base_commit,
            )
        except Exception as e:
            self.logger.exception(f"SCORE | Patch generation failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def actions(self) -> list[ActionType]:
        if not self._registry.actions:
            self._registry.add_action(
                name="bash",
                description="Run a bash command in the repo root and get the output",
                action_cls=BashAction,
                handler=self._handle_bash,
            )
            self._registry.add_action(
                name="finish",
                description=(
                    "Finish the task by submitting a brief"
                    " summary. The system automatically computes"
                    " the git patch from the repository changes."
                ),
                action_cls=SubmitPatchAction,
                handler=self._handle_submit_patch,
                is_finish=True,
            )
        return self._registry.actions

    @property
    def task(self) -> str:
        return self._task_prompt.format_map(
            {
                "container_repo_dir": self.container_repo_dir,
                "problem_statement": self._instance["problem_statement"],
            }
        )

    @property
    def context(self) -> dict[str, str]:
        return {}

    @property
    def task_id(self) -> str:
        return str(self._instance_id)

    # -------------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------------

    def _setup_environment(self) -> None:
        """Initialize the Docker environment for the task."""
        self.logger.info("ENV | Setting up environment")
        from ...utils.logging import capture_stdio_to_session

        with capture_stdio_to_session(self.logger):
            import minisweagent
            import yaml
            from minisweagent.run.extra.swebench import get_sb_environment

            config_path = Path(minisweagent.__file__).parent / "config" / "extra" / "swebench.yaml"
            config = yaml.safe_load(config_path.read_text()).copy()
            config["environment"]["cwd"] = self.container_repo_dir
            config["environment"]["pull_timeout"] = self._environment_pull_timeout

            self.env = get_sb_environment(config=config, instance=self._instance)

            result = self.env.execute(command="git rev-parse HEAD", cwd=self.container_repo_dir)
            self.container_base_commit = result["output"].strip()

        if self.container_base_commit != self._instance["base_commit"]:
            self.logger.error(
                f"ENV | Base commit mismatch: expected {self._instance['base_commit']} "
                f"| got {self.container_base_commit}"
            )


# =============================================================================
# Evaluator
# =============================================================================


class SWEBenchEvaluator(Evaluator):
    """Evaluation logic for SWE-bench — task discovery, session config, aggregation."""

    def __init__(
        self,
        subset: str,
        require_submit_for_patch_evaluation: bool = True,
        max_interactions: int | None = None,
    ) -> None:
        self._subset = subset
        self._require_submit_for_patch_evaluation = require_submit_for_patch_evaluation
        self._max_interactions = max_interactions
        self._dataset: Any = None
        self._instances_by_id: dict[str, dict[str, Any]] = {}

    def list_tasks(self) -> list[str]:
        self._ensure_dataset()
        return list(self._instances_by_id.keys())

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        self._ensure_dataset()
        task_id_str = str(index.task_id)
        instance = self._instances_by_id.get(task_id_str)
        if instance is None:
            raise KeyError(f"Unknown SWE-bench task id: {index.task_id}")
        return {
            "settings": get_settings(),
            "instance": instance,
            "subset": self._subset,
            "max_interactions": self._max_interactions,
            "require_submit_for_patch_evaluation": self._require_submit_for_patch_evaluation,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores: list[float] = []
        session_ids = [s.session_id for s in sessions]
        for paths in self.get_sessions_paths(sessions):
            if not paths.benchmark_results.exists():
                raise FileNotFoundError(
                    f"Missing results for planned session '{paths.session_id}' at {paths.benchmark_results}"
                )
            with open(paths.benchmark_results, encoding="utf-8") as f:
                payload = json.load(f)
            scores.append(float(payload.get("score", 0.0)))

        metrics = swebench_metrics.collect_metrics(get_run_id(), session_ids)

        return BenchmarkResults(
            benchmark_name="swebench",
            total_tasks=len(sessions),
            score=sum(scores) / len(scores) if scores else 0.0,
            metrics=metrics["funnel"],
        )

    def _ensure_dataset(self) -> None:
        if self._dataset is not None:
            return
        if self._subset is None:
            raise ValueError("subset must be configured for SWE-bench.")
        from datasets import load_dataset

        dataset = load_dataset(self._subset, split="test")
        instances = list(dataset)
        if not instances:
            raise RuntimeError(
                f"SWE-bench dataset '{self._subset}' returned 0 instances. Check dataset availability and HF auth."
            )
        self._dataset = instances
        self._instances_by_id = {str(inst["instance_id"]): inst for inst in instances}


# =============================================================================
# Benchmark
# =============================================================================


class SWEBenchBenchmark(Benchmark):
    """Benchmark configuration for SWE-bench evaluation."""

    display_name: ClassVar[str] = "SWE-bench"
    slug_name: ClassVar[str] = "swebench"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_evaluator_class(cls):
        return SWEBenchEvaluator

    @classmethod
    def get_session_class(cls):
        return SWEBenchSession

    subset: str | None = None
    require_submit_for_patch_evaluation: bool = True
    docker_socket: bool = True  # SWE-bench sessions create sibling Docker containers

    def model_post_init(self, __context):
        cfg = get_config()
        benchmark_cfg = cfg["benchmark"]
        session_cfg = cfg["session"]
        if self.subset is None:
            self.subset = benchmark_cfg["subset"]
        if self.runner is None:
            self.runner = benchmark_cfg.get("runner")
        if "seed" in benchmark_cfg:
            self.seed = benchmark_cfg["seed"]
        if (
            "require_submit_for_patch_evaluation" in benchmark_cfg
            and "require_submit_for_patch_evaluation" not in self.model_fields_set
        ):
            self.require_submit_for_patch_evaluation = bool(benchmark_cfg["require_submit_for_patch_evaluation"])
        if self.max_interactions is None:
            self.max_interactions = session_cfg.get("max_interactions")

    def get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "require_submit_for_patch_evaluation": self.require_submit_for_patch_evaluation,
            "max_interactions": self.max_interactions,
        }
