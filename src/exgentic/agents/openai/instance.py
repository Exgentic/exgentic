# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import asyncio
import logging
from typing import Any

import httpx
from agents import Agent as OpenAIAgent
from agents import Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.lifecycle import RunHooksBase
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
from agents.model_settings import ModelSettings as OpenAIModelSettings
from agents.model_settings import Reasoning
from agents.run import RunConfig
from agents.usage import Usage

from ...adapters.agents.mcp_agent import MCPAgentInstance
from ...core.types import ModelSettings, RetryStrategy
from ...integrations.litellm.health import acheck_model_accessible
from ...observers.logging import (
    attach_library_logger_to_handler,
    restore_library_logger,
)
from ...utils.cost import UpdatableCostReport, litellm_tokens_cost
from ...utils.settings import get_settings
from ...utils.sync import run_sync
from .openai_mcp_agent import MCPConfig

settings = get_settings()


class _UsageRunHooks(RunHooksBase[dict[str, Any], OpenAIAgent]):
    def __init__(self, record_usage) -> None:
        super().__init__()
        self._record_usage = record_usage

    async def on_llm_end(self, context, agent, response) -> None:
        self._record_usage(response.usage)


class RetryingLitellmModel(LitellmModel):
    def __init__(
        self,
        model: str,
        *,
        num_retries: int,
        retry_after: float,
        retry_strategy: str | RetryStrategy,
    ):
        super().__init__(model=model)
        if isinstance(retry_strategy, RetryStrategy):
            retry_strategy = retry_strategy.value
        if retry_strategy not in ("exponential_backoff_retry", "constant_retry"):
            raise ValueError(f"Unsupported retry_strategy: {retry_strategy}")
        if num_retries < 0:
            raise ValueError("num_retries must be >= 0")
        if retry_after < 0:
            raise ValueError("retry_after must be >= 0")
        self._num_retries = num_retries
        self._retry_after = retry_after
        self._retry_strategy = retry_strategy

    async def _fetch_response(self, *args, **kwargs):
        for attempt in range(self._num_retries + 1):
            try:
                return await super()._fetch_response(*args, **kwargs)
            except Exception as exc:
                if attempt >= self._num_retries:
                    raise
                delay = self._retry_after
                if self._retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF.value:
                    delay *= 2**attempt
                logging.getLogger(__name__).warning(
                    "OpenAI MCP LiteLLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._num_retries + 1,
                    exc,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
        return None


class OpenAIMCPAgentInstance(MCPAgentInstance):
    """OpenAI Agents SDK + MCP (sync entrypoint, async core)."""

    def __init__(
        self,
        session_id: str,
        model_id: str,
        max_steps: int = 150,
        model_settings: ModelSettings | None = None,
        mcp_config: MCPConfig | dict | None = None,
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
        if mcp_config is None:
            self.mcp_config = MCPConfig()
        elif isinstance(mcp_config, dict):
            self.mcp_config = MCPConfig(**mcp_config)
        else:
            self.mcp_config = mcp_config
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._model_access_checked = False

    async def _check_model_access_once(self) -> None:
        if self._model_access_checked or self.mcp_config.skip_health_check:
            return
        self.logger.info("Running LiteLLM model health check (model=%s)", self.model_id)
        await acheck_model_accessible(self.model_id)
        self._model_access_checked = True

    def _record_usage(self, usage: Usage | None) -> None:
        if usage is None:
            return
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens

    def run_mcp_agent(self, mcp_host: str, mcp_port: int) -> Any:
        # Run async core on the shared loop (sync API)
        return run_sync(self.run_mcp_agent_async(mcp_host, mcp_port), timeout=600.0)

    async def run_mcp_agent_async(self, mcp_host: str, mcp_port: int) -> Any:
        RunConfig.tracing_disabled = True

        prompt = self._build_prompt()

        file_handler = next(
            (h for h in self.logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        logger_states: list[tuple] = []

        try:
            if file_handler:
                logger_states += [
                    attach_library_logger_to_handler("agents", file_handler),
                    attach_library_logger_to_handler(__name__, file_handler),
                ]
                logging.getLogger("agents").setLevel(logging.INFO)
            await self._check_model_access_once()

            # Create custom httpx client factory with extended timeout
            def httpx_client_factory(headers=None, timeout=None, auth=None):
                if (
                    self.mcp_config.http_timeout_seconds is None
                    and self.mcp_config.sse_read_timeout_seconds is None
                    and self.mcp_config.http_connect_timeout_seconds is None
                ):
                    client_timeout = httpx.Timeout(None)
                else:
                    client_timeout = httpx.Timeout(
                        self.mcp_config.http_timeout_seconds,
                        connect=self.mcp_config.http_connect_timeout_seconds,
                        read=self.mcp_config.sse_read_timeout_seconds,
                    )
                return httpx.AsyncClient(
                    headers=headers,
                    timeout=client_timeout,
                    auth=auth,
                )

            mcp_params: dict[str, Any] = {
                "url": f"http://{mcp_host}:{mcp_port}/mcp",
                "httpx_client_factory": httpx_client_factory,
                "terminate_on_close": self.mcp_config.terminate_on_close,
            }
            if self.mcp_config.headers is not None:
                mcp_params["headers"] = self.mcp_config.headers
            if self.mcp_config.http_timeout_seconds is not None:
                mcp_params["timeout"] = self.mcp_config.http_timeout_seconds
            if self.mcp_config.sse_read_timeout_seconds is not None:
                mcp_params["sse_read_timeout"] = self.mcp_config.sse_read_timeout_seconds

            async with MCPServerStreamableHttp(
                params=MCPServerStreamableHttpParams(**mcp_params),
                cache_tools_list=self.mcp_config.cache_tools_list,
                name=self.mcp_config.name,
                client_session_timeout_seconds=self.mcp_config.client_session_timeout_seconds,
                use_structured_content=self.mcp_config.use_structured_content,
                max_retry_attempts=self.mcp_config.max_retry_attempts,
                retry_backoff_seconds_base=self.mcp_config.retry_backoff_seconds_base,
                message_handler=self.mcp_config.message_handler,
            ) as mcp_server:
                temperature = self.model_settings.temperature
                reasoning_effort = self.model_settings.reasoning_effort
                openai_model_settings = OpenAIModelSettings(
                    temperature=temperature if temperature is not None else 1.0,
                    max_tokens=self.model_settings.max_tokens,
                    top_p=self.model_settings.top_p,
                    reasoning=(Reasoning(effort=reasoning_effort) if reasoning_effort is not None else None),
                )
                num_retries = self.model_settings.num_retries or 0
                retry_after = self.model_settings.retry_after
                retry_strategy = self.model_settings.retry_strategy.value
                openai_model_settings.extra_args = {
                    "caching": settings.litellm_caching,
                    "max_retries": 0 if num_retries > 0 else 5,
                }
                agent = OpenAIAgent(
                    name="Assistant",
                    instructions=prompt,
                    model=RetryingLitellmModel(
                        model=self.model_id,
                        num_retries=num_retries,
                        retry_after=retry_after,
                        retry_strategy=retry_strategy,
                    ),
                    model_settings=openai_model_settings,
                    mcp_servers=[mcp_server],
                )
                self.logger.info(
                    "Starting OpenAI MCP agent run (model=%s, task=%s, max_turns=%s)",
                    self.model_id,
                    self.task,
                    self.max_steps,
                )
                hooks = _UsageRunHooks(self._record_usage)
                try:
                    result = await Runner.run(
                        agent,
                        self.task,
                        max_turns=self.max_steps,
                        run_config=RunConfig(tracing_disabled=True),
                        hooks=hooks,
                    )
                except Exception:
                    self.logger.exception("OpenAI MCP agent run failed")
                    raise
                if self._total_input_tokens == 0 and self._total_output_tokens == 0:
                    for resp in result.raw_responses:
                        self._record_usage(resp.usage)
                self.logger.info("OpenAI MCP agent run finished: %s", result)
                return result
        finally:
            for state in logger_states:
                if state:
                    restore_library_logger(*state)
            if file_handler:
                file_handler.flush()

    def get_cost(self) -> UpdatableCostReport:
        report = UpdatableCostReport.initialize_empty(model_name=self.model_id)
        if self._total_input_tokens == 0 and self._total_output_tokens == 0:
            return report

        cost = litellm_tokens_cost(
            model_name=self.model_id,
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
        ).total_cost
        report.add_cost(cost)
        return report

    def _build_prompt(self) -> str:
        prompt = ""
        if self.context:
            prompt += f"Context: {self.context}\n\n"

        prompt += (
            "Complete this task using the available tools. Each tool corresponds to an action "
            "you can take in the environment. Do not respond or ask clarification questions "
            "unless done through a dedicated tool, and only if such tool exist. "
            "Any plain message that is not a tool call will end the run in failure.\n"
        )

        if self.initial_observation is not None and not self.initial_observation.is_empty():
            text = str(self.initial_observation).strip()
            if text:
                prompt += f"\nFirst Observation: {text}\n"

        return prompt
