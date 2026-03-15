# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import abc
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Optional

from ...adapters.agents.mcp_agent import MCPAgentInstance
from ...core.agent import Agent
from ...core.agent_instance import AgentInstance
from ...core.context import context_env
from ...core.types import ActionType, ModelSettings
from ...integrations.litellm import LitellmProxy
from ...integrations.litellm.health import check_model_accessible_sync
from ...integrations.litellm.trace_cost import load_trace_cost
from ...utils.cost import UpdatableCostReport
from .command_runner import (
    BaseCLIConfig,
    CLIResult,
    DockerRunner,
    ExecutionBackend,
    PodmanRunner,
    ProcessRunner,
)


class BaseCLIWrapper(abc.ABC):
    """Shared helpers for headless CLI wrappers.

    Subclasses implement build_env/build_command; run() handles the rest.
    """

    config_prefix = "cli_"
    spawn_error_message = "Spawn failed"

    def __init__(
        self,
        env: Optional[dict[str, str]] = None,
        log_path: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        runner: ExecutionBackend = ExecutionBackend.DOCKER,
    ) -> None:
        self.env = env or os.environ.copy()
        self.config_dir = config_dir
        self.log_path = log_path
        self._last_run_context: dict[str, Any] = {}
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        if runner == ExecutionBackend.PROCESS:
            self.runner = ProcessRunner(log_path=log_path, logger=self._logger)
        elif runner == ExecutionBackend.PODMAN:
            self.runner = PodmanRunner(log_path=log_path, logger=self._logger)
        elif runner == ExecutionBackend.DOCKER:
            self.runner = DockerRunner(log_path=log_path, logger=self._logger)
        else:
            raise ValueError(f"runner value: {runner} is not supported!")

    def run(self, *, prompt: str, config: BaseCLIConfig) -> CLIResult:
        cfg_root = self._resolve_config_root(self.config_prefix)
        self._last_run_context = {
            "cfg_root": cfg_root,
            "prompt": prompt,
            "config": config,
        }
        env = self.build_env(cfg_root=cfg_root, prompt=prompt, config=config)
        env.update(context_env())
        if config.env:
            env = {**env, **config.env}
        cmd = self.build_command(cfg_root=cfg_root, prompt=prompt, config=config)
        return self.runner.run(
            cmd=cmd,
            env=env,
            cfg_root=cfg_root,
            config=config,
            spawn_error_message=self.spawn_error_message,
        )

    def close(self) -> None:
        self.runner.close()

    def _resolve_config_root(self, prefix: str) -> Path:
        if self.config_dir is not None:
            return Path(self.config_dir)
        return Path(tempfile.mkdtemp(prefix=prefix))

    def _log_warning(self, message: str) -> None:
        """Best-effort logger wrapper to avoid attribute errors on exit."""
        try:
            self._logger.warning(message)
        except Exception:
            logging.getLogger(__name__).warning(message)

    # Abstract hooks -------------------------------------------------

    @abc.abstractmethod
    def build_env(self, *, cfg_root: Path, prompt: str, config: Any) -> dict[str, str]:
        ...

    @abc.abstractmethod
    def build_command(self, *, cfg_root: Path, prompt: str, config: Any) -> list[str]:
        ...


class ProxyBackedMCPAgentInstance(MCPAgentInstance, abc.ABC):
    """Base class for MCP agents that launch a LiteLLM proxy and delegate to a CLI wrapper."""

    def __init__(
        self,
        session_id: str,
        task: str,
        context: dict[str, Any],
        actions: list[ActionType],
        model_id: str,
        *,
        max_steps: int = 150,
        model_alias: Optional[str] = None,
        runner: ExecutionBackend = ExecutionBackend.PROCESS,
        model_settings: ModelSettings | None = None,
    ) -> None:
        super().__init__(session_id, task, context, actions)
        self.model_id = model_id
        self.max_steps = max_steps
        self._proxy_log_dir = self.paths.agent_dir / "litellm_proxy"
        self._trace_log_path = self._proxy_log_dir / "trace.jsonl"
        self._proxy: Optional[LitellmProxy] = None
        self._cli: Optional[BaseCLIWrapper] = None
        self._model_alias = model_alias
        self.runner = runner
        if model_settings is None:
            self.model_settings = ModelSettings()
        elif isinstance(model_settings, ModelSettings):
            self.model_settings = model_settings
        else:
            raise ValueError("model_settings must be a ModelSettings instance.")

        # Check model accessibility
        check_model_accessible_sync(self.model_id, logger=self.logger)

    @property
    @abc.abstractmethod
    def cli_display_name(self) -> str:
        ...

    @abc.abstractmethod
    def _build_cli(self) -> BaseCLIWrapper:
        ...

    @abc.abstractmethod
    def _run_cli(
        self,
        cli: BaseCLIWrapper,
        prompt: str,
        mcp_host: str,
        mcp_port: int,
        proxy: LitellmProxy,
    ) -> Any:
        ...

    def close_mcp_agent(self) -> None:
        self.logger.info("Closing CLI before MCP shutdown")
        if self._cli is not None:
            self._cli.close()
            self._cli = None
        self.logger.info("Closing LiteLLM proxy before MCP shutdown")
        if self._proxy is not None:
            self._proxy.close()
            self._proxy = None
        super().close_mcp_agent()

    def run_mcp_agent(self, mcp_host: str, mcp_port: int) -> Any:
        prompt = self._build_prompt()
        proxy_log = self._proxy_log_dir / "litellm_proxy.log"
        alias_map = self._proxy_alias_map()

        # Log model parameters for validation
        params_info = []
        if self.model_settings.temperature is not None:
            params_info.append(f"temperature={self.model_settings.temperature}")
        if self.model_settings.max_tokens is not None:
            params_info.append(f"max_tokens={self.model_settings.max_tokens}")
        if self.model_settings.top_p is not None:
            params_info.append(f"top_p={self.model_settings.top_p}")
        params_str = f" with {', '.join(params_info)}" if params_info else ""

        self.logger.info(
            "Starting LiteLLM proxy for model %s%s (log: %s)",
            self.model_id,
            params_str,
            proxy_log,
        )
        proxy_log.parent.mkdir(parents=True, exist_ok=True)
        proxy = LitellmProxy(
            model=self.model_id,
            log_path=str(proxy_log),
            usage_log_path=str(self._trace_log_path),
            model_alias_map=alias_map or None,
            model_settings=self.model_settings,
        )
        self._proxy = proxy
        try:
            proxy.start()
        except Exception:
            self.logger.exception("LiteLLM proxy failed to start")
            raise
        self.logger.info("LiteLLM proxy started at %s", proxy.base_url)

        try:
            cli = self._build_cli()
            self._cli = cli
            self.logger.info(
                "Launching %s (log: %s) against MCP http://%s:%s/mcp using proxy %s",
                self.cli_display_name,
                cli.log_path,
                mcp_host,
                mcp_port,
                proxy.base_url,
            )
            stdout = self._run_cli(cli, prompt, mcp_host, mcp_port, proxy)
            self.logger.info("%s run finished", self.cli_display_name)
            return stdout
        except Exception as e:
            self.logger.exception("%s run failed: %s", self.cli_display_name, e)
            raise
        finally:
            if self._cli is not None:
                self._cli.close()
                self._cli = None
            proxy.close()
            self._proxy = None
            self._drain_server()

    def get_cost(self) -> UpdatableCostReport:
        cost = load_trace_cost(self._trace_log_path, self.model_id)
        report = UpdatableCostReport.initialize_empty(model_name=self.model_id)
        report.add_cost(cost)
        return report

    def _build_prompt(self) -> str:
        prompt = ""
        if self.context:
            prompt += f"Context: {self.context}\n\n"

        finish_hint = ""
        finish_tools = [a.name for a in self.actions if a.is_finish]
        if finish_tools:
            finish_hint = f" Use the designated finish tool(s): {', '.join(finish_tools)}."

        instructions = (
            "Complete this task using the available environment tools. Each tool corresponds to an action "
            " you can take in the task environment.\n"
            "# Important: You are on solo mode. Do not reply back or message unless its through a dedicated "
            "environment tool call, every such attempt will finish the session with failure.\n"
            "All your actions on with regard to the task must go through environment tool calls."
        )

        if finish_hint:
            instructions += (
                f"{finish_hint} Always conclude by invoking the designated finish tool for this task environment."
            )
        prompt += f"{instructions}\n"

        if self.initial_observation is not None and not self.initial_observation.is_empty():
            text = str(self.initial_observation).strip()
            if text:
                prompt += f"\nFirst Observation: {text}\n"

        return prompt + self.task

    def close(self) -> None:
        if self._cli is not None:
            self._cli.close()
            self._cli = None
        if self._proxy is not None:
            self._proxy.close()
            self._proxy = None
        self._drain_server()
        super().close()

    def _proxy_alias_map(self) -> dict[str, str]:
        if not self._model_alias:
            return {}
        return {self._model_alias: self.model_id}

    def _drain_server(self) -> None:
        """Best-effort stop/join of the MCP server thread to avoid teardown crashes."""
        server = self._mcp_server
        if server is None:
            return
        try:
            server.stop(raise_on_timeout=False)
        except Exception as exc:
            self._log_warning(f"Error while stopping MCP server: {exc}")

    def _log_warning(self, message: str) -> None:
        try:
            self._logger.warning(message)
        except Exception:
            logging.getLogger(__name__).warning(message)


class ProxyBackedAgent(Agent):
    """Minimal agent factory helper for proxy-backed CLI agents."""

    agent_cls: ClassVar[type[ProxyBackedMCPAgentInstance] | None] = None
    model: str
    max_steps: int = 150
    runner: ExecutionBackend = ExecutionBackend.PROCESS
    model_settings: ModelSettings | None = None

    def assign(
        self,
        task: str,
        context: dict[str, Any],
        actions: list[ActionType],
        session_id: str,
    ) -> AgentInstance:
        agent_cls = self.__class__.agent_cls
        if agent_cls is None:
            raise ValueError("agent_cls must be set on the agent class.")
        return agent_cls(
            session_id=session_id,
            task=task,
            context=context,
            actions=actions,
            model_id=self.model,
            max_steps=self.max_steps,
            runner=self.runner,
            model_settings=self.model_settings,
        )

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model)]
