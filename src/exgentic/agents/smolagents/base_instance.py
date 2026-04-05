# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import functools
import logging
from abc import abstractmethod
from collections.abc import Callable

from rich.console import Console
from smolagents import LiteLLMModel
from smolagents.models import is_rate_limit_error
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.tools import Tool, tool
from smolagents.utils import AgentError, Retrying

from ...adapters.agents.code_agent import CodeAgentInstance
from ...core.types import ModelSettings, RetryStrategy
from ...integrations.litellm.health import check_model_accessible_sync
from ...observers.logging import close_logger
from ...utils.cost import CostReport, LiteLLMCostReport
from ...utils.settings import get_settings

settings = get_settings()


class SmolagentBaseAgentInstance(CodeAgentInstance):
    def __init__(
        self,
        session_id: str,
        model_id: str,
        max_steps: int = 150,
        model_settings: ModelSettings | None = None,
        retry_on_all_errors: bool = True,
    ):
        super().__init__(session_id)
        self.model_id = model_id
        self.max_steps = max_steps
        if model_settings is None:
            self.model_settings = ModelSettings()
        elif isinstance(model_settings, ModelSettings):
            self.model_settings = model_settings
        else:
            raise ValueError("model_settings must be a ModelSettings instance.")
        self._retry_on_all_errors = retry_on_all_errors
        self._agent = None
        self._model = None

        # Check model accessibility
        check_model_accessible_sync(self.model_id, logger=self.logger)

    def run_code_agent(self, functions: list[Callable]) -> None:
        def _wrap_tool(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except RuntimeError as exc:
                    if "after close" in str(exc):
                        agent_logger = self.get_smolagent_logger()
                        if agent_logger is None:
                            agent_logger = AgentLogger(
                                console=Console(),
                                level=LogLevel.ERROR,
                            )
                        raise AgentError("Agent interrupted (session closed).", agent_logger) from exc
                    raise

            return wrapper

        tools = [tool(_wrap_tool(function)) for function in functions]
        return self.run_smolagent(tools=tools)

    def get_smolagent_logger(self):
        smolagent_logger = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                console = Console(
                    file=handler.stream,
                    force_terminal=False,
                    color_system=None,
                    highlight=False,
                )
                smolagent_logger = AgentLogger(console=console, level=LogLevel.DEBUG)
        return smolagent_logger

    def get_internal_model(self):
        if self._model is None:
            temperature = self.model_settings.temperature
            self._model = LiteLLMModel(
                model_id=self.model_id,
                temperature=temperature if temperature is not None else 1.0,
                max_tokens=self.model_settings.max_tokens,
                caching=settings.litellm_caching,
            )
            num_retries = self.model_settings.num_retries or 0
            max_attempts = num_retries + 1 if num_retries > 0 else 1
            retry_strategy = self.model_settings.retry_strategy.value
            exponential_base = 2.0 if retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF.value else 1.0
            log_level = logging._nameToLevel.get(settings.log_level, logging.INFO)
            self._model.retryer = Retrying(
                max_attempts=max_attempts,
                wait_seconds=self.model_settings.retry_after,
                exponential_base=exponential_base,
                jitter=False,
                retry_predicate=self.retry_predicate,
                reraise=True,
                before_sleep_logger=(self.logger, log_level),
                after_logger=None,
            )
        return self._model

    def retry_predicate(self, exc: BaseException) -> bool:
        if self._retry_on_all_errors:
            return True
        return is_rate_limit_error(exc)

    @abstractmethod
    def run_smolagent(self, tools: list[Tool]):
        raise NotImplementedError

    def close(self):
        self.logger.info("Interrupting Smolagent...")
        if self._agent is not None:
            self._agent.interrupt()
        super().close()
        self.logger.debug("Closing logger.")
        close_logger(self.logger)

    def get_cost(self) -> CostReport:
        if self._agent is None:
            return LiteLLMCostReport.initialize_empty(model_name=self.model_id)

        token_usage = self._agent.monitor.get_total_token_counts()

        return LiteLLMCostReport.from_token_counts(
            model_name=self.model_id,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
        )
