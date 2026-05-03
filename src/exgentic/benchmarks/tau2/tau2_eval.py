# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""TAU2 evaluator, session, and proxy-agent classes.

These classes import tau2 (via tau2_shim) at module level.  They are only
ever instantiated inside the isolated venv subprocess, so the heavy
``tau2`` dependency is never required in the host process.
"""

from __future__ import annotations

import builtins
import contextvars
import json
import logging
import os
import threading
import traceback
from pathlib import Path
from shutil import move
from typing import TYPE_CHECKING, Any

from ...adapters.actions.chat import ChatActionContext
from ...adapters.executors.proxy import PairableProxyAgent, PairableProxySession
from ...adapters.schemas.openai import openai_tools_to_action_types
from ...core.actions import ActionsHandler
from ...core.evaluator import Evaluator
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    MessageAction,
    SessionIndex,
    SessionScore,
    SingleAction,
    SingleObservation,
)
from ...integrations.litellm.config import configure_litellm
from ...integrations.litellm.health import check_model_accessible_sync
from ...observers.logging import (
    add_loguru_file_sink,
    attach_library_logger_to_handler,
    close_logger,
    get_logger,
    remove_loguru_sink,
    restore_library_logger,
)
from ...utils.cost import CostReport, LiteLLMCostReport, UpdatableCostReport
from ...utils.paths import get_run_id
from ...utils.settings import get_settings
from .tau2_shim import (
    AssistantMessage,
    Console,
    ConsoleDisplay,
    LLMAgent,
    MultiToolMessage,
    Results,
    RunConfig,
    TerminationReason,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    compute_metrics,
    is_successful,
    load_tasks,
    registry,
    run_domain,
)

if TYPE_CHECKING:
    pass

# Resolve settings once per module
settings = get_settings()
logger = get_logger(__name__)


def current_run_id() -> str:
    return get_run_id()


PROXY_AGENT_NAME = "proxy_agent"
TAU2_TOTAL_TASKS = {
    "mock": 9,
    "retail": 114,
    "airline": 50,
    "telecom": 114,
}


def _echo_action(action: SingleAction) -> SingleAction:
    """Return the action unchanged so registry can normalize/validate without altering behavior."""
    return action


def tau_message_to_user_tool_message(message: Any) -> dict[str, Any]:
    """Convert Tau2 message objects into a generic chat-style payload."""
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": str(message.id),
            "content": message.content,
        }
    if isinstance(message, MultiToolMessage):
        return [{"role": "tool", "tool_call_id": str(m.id), "content": m.content} for m in message.tool_messages]
    return {"content": str(message)}


def assistant_message_to_tau_message(msg_dict: dict[str, Any]) -> AssistantMessage:
    """Convert a generic assistant message dict into a Tau2 AssistantMessage."""
    tool_calls: list[ToolCall] | None = None
    if "tool_calls" in msg_dict:
        tool_calls = [ToolCall(**p) for p in msg_dict["tool_calls"]]
    return AssistantMessage(role="assistant", content=msg_dict.get("content"), tool_calls=tool_calls)


class TAU2Session(PairableProxySession):
    """Proxy session whose score reads a single-task results file (runs remote)."""

    def __init__(
        self,
        task_id: str,
        subset: str,
        user_simulator_model: str,
        llm_temperature_user: float,
        llm_user_input_cost_per_token: float | None,
        llm_user_output_cost_per_token: float | None,
        max_steps: int,
        max_errors: int,
        num_trials: int,
        seed: int,
        use_cache: bool,
        session_id: str | None = None,
        user_simulator_litellm_params_extra: dict[str, Any] | None = None,
    ):
        if session_id is not None:
            self._session_id = session_id
        self._user_simulator_litellm_params_extra: dict[str, Any] = dict(user_simulator_litellm_params_extra or {})
        llm_args_user: dict[str, Any] = {
            "temperature": llm_temperature_user,
            "caching": settings.litellm_caching,
        }
        if llm_user_input_cost_per_token is not None:
            llm_args_user["input_cost_per_token"] = llm_user_input_cost_per_token
        if llm_user_output_cost_per_token is not None:
            llm_args_user["output_cost_per_token"] = llm_user_output_cost_per_token
        llm_args_user.update(self._user_simulator_litellm_params_extra)
        self._cfg = RunConfig(
            domain=subset,
            user="user_simulator",
            task_set_name=None,
            task_ids=[str(task_id)],
            num_tasks=1,
            agent=PROXY_AGENT_NAME,
            llm_agent="unknown",
            llm_args_agent={},
            llm_user=user_simulator_model,
            llm_args_user=llm_args_user,
            num_trials=num_trials,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            log_level=settings.log_level,
            max_concurrency=1,
            is_remote=False,
            save_to=None,
        )
        self.use_cache = use_cache
        self._cfg.save_to = self.session_id

        self.tools: list[Tool] = []
        self.domain_policy: str = ""
        self._registry: ActionsHandler | None = None
        self.file_path: str | None = None
        self._runner_thread: threading.Thread | None = None
        self._runner_error: Exception | None = None
        self._chat_ctx = ChatActionContext()
        self._user_input_tokens = 0
        self._user_output_tokens = 0
        self._user_total_cost = 0.0

        self._registry = ActionsHandler(
            logger=self.logger,
            warn_on_validation_error=False,
            warn_on_unknown_action=False,
            handle_validation_error=lambda action, _msg: SingleObservation(invoking_actions=[action], result=action),
        )
        self._registry.add_action(
            name="message",
            description="Send a message to the user.",
            action_cls=MessageAction,
            handler=_echo_action,
            is_message=True,
        )

        # Load TAU2 environment tools to expose actions
        environment_constructor = registry.get_env_constructor(self._cfg.domain)
        environment = environment_constructor()
        self.domain_policy = environment.get_policy()
        tools = environment.get_tools()
        openai_tools = [t.openai_schema for t in tools]
        action_types = openai_tools_to_action_types(openai_tools)
        self._registry.add_actions(action_types, _echo_action)

        # Persist config/manifest after actions/context are initialized
        super().__init__()
        self.logger.debug(f"Init session PID:{os.getpid()}")

        with open(self.paths.benchmark_task, "w", encoding="utf-8") as f:
            payload = self.task  # pydantic RunConfig
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with open(self.paths.benchmark_context, "w", encoding="utf-8") as f:
            payload = self.context  # pydantic RunConfig
            json.dump(payload, f, ensure_ascii=False, indent=2)

        from . import get_tau2_data_dir

        base = Path(get_tau2_data_dir()) / "simulations"
        base.mkdir(parents=True, exist_ok=True)
        self.file_path = str((base / f"{self._cfg.save_to}.json").resolve())
        self.results_file = self.paths.benchmark_results

        # Check user simulator model accessibility before starting Tau2 runner
        check_model_accessible_sync(
            self._cfg.llm_user,
            logger=self.logger,
            litellm_params_extra=self._user_simulator_litellm_params_extra or None,
        )

        # Start Tau2 runner
        self.logger.debug("Staging for pairing")
        self.stage_for_pairing()
        self.logger.debug("Staged OK")

        def _runner():
            self.logger.debug(f"Runner started PID:{os.getpid()}")
            agent_name = self._cfg.agent
            if agent_name not in registry.get_agents():
                registry.register_agent(TAU2ProxyAgent, agent_name)
            # Prepare session log path and redirect Tau2 console + prints
            log_fh = open(self.paths.benchmark_dir / "tau2_session.log", "a", encoding="utf-8")
            prev_console = ConsoleDisplay.console
            prev_print = builtins.print
            prev_input = builtins.input
            tau2_logger_state = None
            loguru_sink_id = None

            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    ConsoleDisplay.console = Console(
                        file=log_fh,
                        force_terminal=False,
                        color_system=None,
                        highlight=False,
                    )
                    tau2_logger_state = attach_library_logger_to_handler("tau2", handler)
                    # Also route Loguru logs to the Tau2 session file if Loguru is used by Tau2
                    loguru_sink_id = add_loguru_file_sink(log_fh, level="DEBUG", colorize=False)
                    break

            def _file_print(*args, **kwargs):
                if "file" not in kwargs:
                    kwargs["file"] = log_fh
                return prev_print(*args, **kwargs)

            builtins.print = _file_print

            # Prevent interactive prompts from causing EOFError in non-interactive runs
            def _no_input(*args, **kwargs):
                return ""

            builtins.input = _no_input
            try:
                if self.file_path and Path(self.file_path).exists():
                    self.logger.info(
                        "Removing existing TAU2 simulation file before run: %s",
                        self.file_path,
                    )
                    Path(self.file_path).unlink()
                self.logger.info("Starting TAU2 run domain")
                run_domain(self._cfg)
                self.logger.info("TAU2 run completed")
            except Exception as e:
                self.logger.error(f"TAU2 run FAILED with Exception: {e}")
                trace_str = traceback.format_exc()
                self.logger.error(trace_str)
                self._runner_error = e
            finally:
                builtins.print = prev_print
                builtins.input = prev_input
                ConsoleDisplay.console = prev_console
                # Restore tau2 logger handlers
                if tau2_logger_state is not None:
                    (
                        tau2_logger,
                        prev_tau2_handlers,
                        prev_tau2_propagate,
                    ) = tau2_logger_state
                    restore_library_logger(tau2_logger, prev_tau2_handlers, prev_tau2_propagate)
                remove_loguru_sink(loguru_sink_id)
                log_fh.flush()
                log_fh.close()
                # Ensure any waiting session.step() unblocks when Tau2 run completes
                self.logger.debug("Sending terminal observation to unblock session.step()")
                self.put_observation(None)
                # If session ended before pairing with agent, release the pairing semaphore
                # to avoid deadlocks
                self.unstage_for_pairing()
                self.logger.debug("TAU2 runner thread finishing")

        # Copy the parent's contextvars so the daemon thread inherits
        # the exgentic Context (run_id, session_id, output_dir, etc.).
        ctx_copy = contextvars.copy_context()
        t = threading.Thread(target=ctx_copy.run, args=(_runner,), daemon=True)
        t.start()
        self._runner_thread = t

    def get_config(self) -> dict[str, Any]:
        return self._cfg.model_dump()

    # Proxy -> Exgentic observation mapping
    def update_message(self, message: Any) -> None:
        if isinstance(message, UserMessage):
            usage = message.usage or {}
            self._user_input_tokens += usage.get("prompt_tokens", 0)
            self._user_output_tokens += usage.get("completion_tokens", 0)
            if message.cost is not None:
                self._user_total_cost += float(message.cost)
        payload = tau_message_to_user_tool_message(message)
        obs = self._chat_ctx.message_to_observation(payload)
        self.put_observation(obs)

    @property
    def task(self) -> str:
        return (
            "You are a customer service agent that helps the"
            " user according to the <policy> provided below."
            " Try to be helpful and always follow the policy."
        )

    @property
    def context(self) -> dict[str, Any]:
        return {"policy": self.domain_policy}

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    @property
    def task_id(self) -> str:
        if self._cfg.task_ids:
            return str(self._cfg.task_ids[0])
        return ""

    def close(self):
        self.logger.debug("Closing session")
        try:
            super().close()  # This sets self.completed = True
            t = self._runner_thread
            self.logger.debug(f"Thread state: alive={t.is_alive() if t else 'None'}")
            if t and t.is_alive():
                self.logger.debug("Waiting for runner thread")
                t.join(timeout=10.0)
                if t.is_alive():
                    self.logger.warning("Runner thread did not exit cleanly, continuing anyway")
            self.logger.debug("Thread join completed")
            if not Path(self.results_file).exists():
                if self.file_path and Path(self.file_path).exists():
                    self.logger.debug("Moving results file")
                    move(self.file_path, self.results_file)
                    self.logger.debug("File move completed")
                else:
                    self.logger.error("Results file not found")
                    raise FileNotFoundError(
                        f"TAU2 results file not found for session {self.session_id}: {self.file_path}"
                    )

            self.logger.debug("Closing logging")
            if not t or not t.is_alive():
                close_logger(self.logger)
        finally:
            # Surface runner thread error after cleanup to avoid leaking resources
            if self._runner_error is not None:
                raise RuntimeError(
                    f"TAU2 runner thread failed with error: {self._runner_error}"
                ) from self._runner_error

    def get_cost(self) -> CostReport:
        def _custom_token_cost(input_tokens: int, output_tokens: int) -> float | None:
            args = self._cfg.llm_args_user or {}
            input_rate = args.get("input_cost_per_token")
            output_rate = args.get("output_cost_per_token")
            if input_rate is None and output_rate is None:
                return None
            input_cost = input_tokens * float(input_rate or 0.0)
            output_cost = output_tokens * float(output_rate or 0.0)
            return input_cost + output_cost

        def _report_from_usage(input_tokens: int, output_tokens: int) -> CostReport | None:
            if input_tokens == 0 and output_tokens == 0:
                return None
            custom_total = _custom_token_cost(input_tokens, output_tokens)
            if custom_total is not None:
                report = UpdatableCostReport.initialize_empty(model_name=self._cfg.llm_user)
                report.add_cost(custom_total)
                return report
            return LiteLLMCostReport.from_token_counts(
                model_name=self._cfg.llm_user,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        def _report_from_messages(messages: list[Any]) -> CostReport | None:
            total_cost = 0.0
            has_cost = False
            input_tokens = 0
            output_tokens = 0
            for message in messages:
                if message.role != "user":
                    continue
                usage = message.usage or {}
                input_tokens += usage.get("prompt_tokens", 0)
                output_tokens += usage.get("completion_tokens", 0)
                if message.cost is not None:
                    total_cost += float(message.cost)
                    has_cost = True
            if has_cost and total_cost > 0:
                report = UpdatableCostReport.initialize_empty(model_name=self._cfg.llm_user)
                report.add_cost(total_cost)
                return report
            return _report_from_usage(input_tokens, output_tokens)

        def _load_results(path: str | None) -> Results | None:
            if not path:
                return None
            results_path = Path(path)
            if not results_path.exists():
                return None
            try:
                return Results.load(path)
            except Exception:
                return None

        res = _load_results(self.results_file)
        if res is None:
            res = _load_results(self.file_path)
        if res is not None and res.simulations:
            sim = res.simulations[-1]
            report = _report_from_messages(sim.messages)
            if report is not None:
                return report

        if self._user_total_cost > 0:
            report = UpdatableCostReport.initialize_empty(model_name=self._cfg.llm_user)
            report.add_cost(self._user_total_cost)
            return report

        report = _report_from_usage(self._user_input_tokens, self._user_output_tokens)
        if report is not None:
            return report

        return LiteLLMCostReport.initialize_empty(model_name=self._cfg.llm_user)

    def score(self) -> SessionScore:
        # Check if the runner thread encountered an error and surface it
        if self._runner_error is not None:
            raise RuntimeError(f"TAU2 runner thread failed with error: {self._runner_error}") from self._runner_error
        # Ensure the results file is in place.  score() may be called before
        # close() by the framework, so move the tau2 simulation output now.
        if not Path(self.results_file).exists():
            t = self._runner_thread
            if t and t.is_alive():
                t.join(timeout=30.0)
            if self.file_path and Path(self.file_path).exists():
                Path(self.results_file).parent.mkdir(parents=True, exist_ok=True)
                move(self.file_path, self.results_file)
        # No simulation file or empty simulations → agent produced no
        # useful actions.  Score as unsuccessful rather than crashing.
        if not Path(self.results_file).exists():
            self.logger.warning("No simulation file produced; scoring as unsuccessful (score=0).")
            return SessionScore(score=0.0, success=False, is_finished=True)
        res = Results.load(self.results_file)
        if not res.simulations:
            self.logger.warning("No simulations in results; scoring as unsuccessful (score=0).")
            return SessionScore(score=0.0, success=False, is_finished=True)

        sim = res.simulations[-1]

        self.paths.benchmark_dir.mkdir(parents=True, exist_ok=True)
        with open(self.paths.benchmark_dir / "dialog.log", "w", encoding="utf-8") as f:
            prev_console = ConsoleDisplay.console
            ConsoleDisplay.console = Console(file=f, force_terminal=False, color_system=None)
            ConsoleDisplay.display_simulation(sim)
            ConsoleDisplay.console = prev_console

        self.logger.info("Computing score")
        self.logger.info(f"Score: {sim.reward_info.reward}")
        # Finished only when Tau2 reports an agent/user stop termination.
        termination = sim.termination_reason
        # Default to not finished unless Tau2 says the run ended cleanly.
        graceful = False
        if isinstance(termination, TerminationReason):
            graceful = termination in (
                TerminationReason.AGENT_STOP,
                TerminationReason.USER_STOP,
            )
        elif isinstance(termination, str):
            graceful = termination in ("agent_stop", "user_stop")
        session_metadata: dict[str, Any] = {}
        session_metrics: dict[str, Any] = {}
        if sim.reward_info is not None:
            session_metadata["reward_info"] = sim.reward_info.model_dump(mode="json")
            session_metrics["reward"] = sim.reward_info.reward
            if sim.reward_info.db_check is not None:
                session_metrics["db_check_db_match"] = sim.reward_info.db_check.db_match
                session_metrics["db_check_db_reward"] = sim.reward_info.db_check.db_reward
        return SessionScore(
            score=sim.reward_info.reward,
            success=is_successful(sim.reward_info.reward),
            is_finished=graceful,
            session_metrics=session_metrics,
            session_metadata=session_metadata,
        )


class TAU2ProxyAgent(LLMAgent, PairableProxyAgent[TAU2Session]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str | None = None,
        llm_args: dict | None = None,
    ):
        sess = self.adopt_staged_session()
        sess.logger.debug(f"Agent adopted PID:{os.getpid()}")
        sess.tools = tools
        sess.domain_policy = domain_policy
        self.session = sess
        configure_litellm(config=settings.to_litellm_config(), cache_only=True)
        super().__init__(tools, domain_policy, llm, llm_args)

    def generate_next_message(self, message: Any, state: TAU2Session | None):
        self.session.logger.info(repr(message))
        return self.handle_observation(message, state)

    def get_init_state(self, message_history: list | None = None) -> TAU2Session:  # type: ignore[override]
        return self.session

    # BaseProxyAgent hooks
    def create_session(self, first_observation: Any) -> TAU2Session:
        return self.session

    def update_session_observation(self, session: TAU2Session, observation: Any) -> None:
        session.update_message(observation)

    def action_to_response(self, action: Any | None, observation: Any, session: TAU2Session):
        if action is None:
            message = (
                AssistantMessage(role="assistant", content="__done__", tool_calls=None),
                session,
            )
            return message

        actions = self._expand_actions(action, session)

        msg_dict = session._chat_ctx.actions_to_assistant_message(actions)  # type: ignore[attr-defined]
        message = assistant_message_to_tau_message(msg_dict)
        session.logger.info(repr(message))
        return (message, session)

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message.

        By default the agent does not stop.
        """
        return message.content == "__done__"

    # Registry helpers -------------------------------------------------------
    def _expand_actions(self, action: Action, session: TAU2Session) -> list[SingleAction]:
        """Normalize raw Action into a list of SingleAction via registry."""
        registry = session._registry  # type: ignore[attr-defined]
        expanded: list[SingleAction] = []
        for raw in action.to_action_list():
            # Handlers are no-ops; this call just normalizes/validates the action shape.
            obs = registry.normalize(raw)
            if obs is None:
                continue
            for so in obs.to_observation_list():
                res = so.result
                if isinstance(res, SingleAction):
                    expanded.append(res)
                elif so.invoking_actions:
                    expanded.extend(so.invoking_actions)
                elif isinstance(res, Action):
                    expanded.append(res)  # type: ignore[arg-type]
        return expanded


class TAU2Evaluator(Evaluator):
    """Evaluation logic for TAU2 -- task discovery, session kwargs, aggregation."""

    def __init__(
        self,
        subset: str,
        score_path: str | None = None,
    ):
        self._subset = subset
        self._score_path = score_path

    def list_tasks(self) -> list[str]:
        tasks = load_tasks(task_set_name=self._subset)
        return [str(t.id) for t in tasks]

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        """Aggregate per-session Tau2 result files and expose a final score.

        - Computes Tau2 metrics via ``compute_metrics`` for detailed reporting.
        - Derives a top-level ``score`` as the mean per-session reward to provide
          a single scalar suitable for tracker summaries and comparisons.
        """
        files: list[Path] = []
        for paths in self.get_sessions_paths(sessions):
            fp = paths.benchmark_results
            if fp.exists():
                files.append(fp)

        base: Results | None = None
        all_sims = []
        task_map: dict[str, Any] = {}
        for fp in files:
            r = Results.load(fp)
            if base is None:
                base = r
            if len(r.tasks) != 1:
                raise ValueError(f"Expected exactly 1 task per result file, got {len(r.tasks)} in {fp}")
            if len(r.simulations) > 1:
                raise ValueError(f"Expected at most 1 simulation per result file, got {len(r.simulations)} in {fp}")
            if len(r.simulations) == 0:
                continue
            all_sims.extend(r.simulations)
            for t in r.tasks:
                task_map[t.id] = t

        total_sessions = len(sessions)

        # No simulations at all → return score=0 instead of crashing.
        if len(all_sims) == 0 or base is None:
            return BenchmarkResults(
                benchmark_name=f"tau2-{self._subset}",
                total_tasks=total_sessions,
                score=0.0,
                metrics={"avg_reward": 0.0},
            )

        combined = Results(info=base.info, tasks=list(task_map.values()), simulations=all_sims)
        m = compute_metrics(combined)

        return BenchmarkResults(
            benchmark_name=f"tau2-{self._subset}",
            total_tasks=total_sessions,
            score=m.avg_reward,
            metrics=m.as_dict(),
        )
