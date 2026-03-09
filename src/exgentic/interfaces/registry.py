# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..core.agent import Agent
    from ..core.benchmark import Benchmark


@dataclass(frozen=True)
class RegistryEntry:
    slug_name: str
    display_name: str
    module: str
    attr: str
    kind: str
    subsets: tuple[str, ...] = ()
    subset_arg: str | None = None
    task_ids_arg: str | None = None
    task_id_type: str | None = None

    def is_available(self) -> bool:
        return importlib.util.find_spec(self.module) is not None

    def load(self) -> type:
        try:
            module = importlib.import_module(self.module)
        except Exception as exc:
            raise ImportError(f"Failed to import {self.kind} '{self.slug_name}' from {self.module}: {exc}") from exc
        try:
            return getattr(module, self.attr)
        except AttributeError as exc:
            raise ImportError(f"Missing {self.kind} class '{self.attr}' in {self.module}") from exc


BENCHMARKS: dict[str, RegistryEntry] = {
    "tau2": RegistryEntry(
        slug_name="tau2",
        display_name="Tau Bench 2",
        module="exgentic.benchmarks.tau2.tau2_benchmark",
        attr="TAU2Benchmark",
        kind="benchmark",
        subsets=("mock", "retail", "airline", "telecom"),
        subset_arg="subset",
        task_id_type="int",
    ),
    "appworld": RegistryEntry(
        slug_name="appworld",
        display_name="AppWorld",
        module="exgentic.benchmarks.appworld.appworld_benchmark",
        attr="AppWorldBenchmark",
        kind="benchmark",
        subsets=("train", "dev", "test_normal"),
        subset_arg="subset",
        task_id_type="str",
    ),
    "gsm8k": RegistryEntry(
        slug_name="gsm8k",
        display_name="GSM8k",
        module="exgentic.benchmarks.gsm8k.gsm8k_benchmark",
        attr="GSM8kBenchmark",
        kind="benchmark",
        subsets=("main",),
        subset_arg="subset",
        task_id_type="int",
    ),
    "hotpotqa": RegistryEntry(
        slug_name="hotpotqa",
        display_name="HotpotQA",
        module="exgentic.benchmarks.hotpotqa.hotpotqa_benchmark",
        attr="HotpotQABenchmark",
        kind="benchmark",
        subsets=("distractor",),
        subset_arg="subset",
        task_id_type="int",
    ),
    "browsecompplus": RegistryEntry(
        slug_name="browsecompplus",
        display_name="BrowseCompPlus",
        module="exgentic.benchmarks.browsecompplus.browsecomp_benchmark",
        attr="BrowseCompPlusBenchmark",
        kind="benchmark",
        subsets=("main",),
        subset_arg="subset",
        task_id_type="int",
    ),
    "swebench": RegistryEntry(
        slug_name="swebench",
        display_name="SWE-bench",
        module="exgentic.benchmarks.swebench.swebench_benchmark",
        attr="SWEBenchBenchmark",
        kind="benchmark",
        subsets=(),
        subset_arg="subset",
        task_id_type="str",
    ),
}

AGENTS: dict[str, RegistryEntry] = {
    "tool_calling": RegistryEntry(
        slug_name="tool_calling",
        display_name="LiteLLM Tool Calling",
        module="exgentic.agents.litellm_tool_calling.litellm_tool_calling_agent",
        attr="LiteLLMToolCallingAgent",
        kind="agent",
    ),
    "smolagents_tool": RegistryEntry(
        slug_name="smolagents_tool",
        display_name="SmolAgents Tool Calling",
        module="exgentic.agents.smolagents.tool_calling_agent",
        attr="SmolagentToolCallingAgent",
        kind="agent",
    ),
    "smolagents_code": RegistryEntry(
        slug_name="smolagents_code",
        display_name="SmolAgents Code",
        module="exgentic.agents.smolagents.code_agent",
        attr="SmolagentCodeAgent",
        kind="agent",
    ),
    "openai_solo": RegistryEntry(
        slug_name="openai_solo",
        display_name="OpenAI Solo",
        module="exgentic.agents.openai.openai_mcp_agent",
        attr="OpenAIMCPAgent",
        kind="agent",
    ),
    "claude_code": RegistryEntry(
        slug_name="claude_code",
        display_name="Claude Code CLI",
        module="exgentic.agents.cli.claude.agent",
        attr="ClaudeCodeAgent",
        kind="agent",
    ),
    "codex_cli": RegistryEntry(
        slug_name="codex_cli",
        display_name="Codex CLI",
        module="exgentic.agents.cli.codex.agent",
        attr="CodexAgent",
        kind="agent",
    ),
    "gemini_cli": RegistryEntry(
        slug_name="gemini_cli",
        display_name="Gemini CLI",
        module="exgentic.agents.cli.gemini.agent",
        attr="GeminiAgent",
        kind="agent",
    ),
}


def get_benchmark_entries() -> dict[str, RegistryEntry]:
    return dict(BENCHMARKS)


def get_agent_entries() -> dict[str, RegistryEntry]:
    return dict(AGENTS)


def get_benchmark_subsets(slug_name: str) -> list[str]:
    entry = BENCHMARKS.get(slug_name)
    if entry is None:
        raise KeyError(f"Unknown benchmark slug '{slug_name}'")
    return list(entry.subsets)


def get_benchmark_subset_arg(slug_name: str) -> str | None:
    entry = BENCHMARKS.get(slug_name)
    if entry is None:
        raise KeyError(f"Unknown benchmark slug '{slug_name}'")
    return entry.subset_arg


def apply_subset_kwargs(slug_name: str, subset: str | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    if subset is None:
        return kwargs
    subsets = get_benchmark_subsets(slug_name)
    if subsets and subset not in subsets:
        raise ValueError(f"Unknown subset '{subset}' for '{slug_name}'. Available: {', '.join(subsets)}")
    subset_arg = get_benchmark_subset_arg(slug_name)
    if subset_arg:
        if subset_arg in kwargs and kwargs[subset_arg] != subset:
            raise ValueError(f"Conflicting subset selection: {subset_arg}={kwargs[subset_arg]} " f"but subset={subset}")
        merged = dict(kwargs)
        merged[subset_arg] = subset
        return merged
    if subsets and subset != subsets[0]:
        raise ValueError(
            f"Benchmark '{slug_name}' does not support subset selection; " f"default subset is '{subsets[0]}'."
        )
    return kwargs


def apply_task_kwargs(slug_name: str, tasks: list[str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    if not tasks:
        return kwargs
    entry = BENCHMARKS.get(slug_name)
    if entry is None:
        raise KeyError(f"Unknown benchmark slug '{slug_name}'")
    if not entry.task_ids_arg:
        raise ValueError(f"Benchmark '{slug_name}' does not support task filtering.")
    if entry.task_id_type == "int":
        try:
            coerced = [int(v) for v in tasks]
        except Exception as exc:
            raise ValueError(f"Invalid task for '{slug_name}': {tasks}. Expected integers.") from exc
    else:
        coerced = [str(v) for v in tasks]
    if entry.task_ids_arg in kwargs and kwargs[entry.task_ids_arg] != coerced:
        raise ValueError(
            f"Conflicting task selection: {entry.task_ids_arg}={kwargs[entry.task_ids_arg]} " f"but tasks={coerced}"
        )
    merged = dict(kwargs)
    merged[entry.task_ids_arg] = coerced
    return merged


def load_benchmark(slug_name: str) -> type[Benchmark]:
    entry = BENCHMARKS.get(slug_name)
    if entry is None:
        raise KeyError(f"Unknown benchmark slug '{slug_name}'")
    cls = entry.load()
    _validate_entry(entry, cls)
    return cls  # type: ignore[return-value]


def load_agent(slug_name: str) -> type[Agent]:
    entry = AGENTS.get(slug_name)
    if entry is None:
        raise KeyError(f"Unknown agent slug '{slug_name}'")
    cls = entry.load()
    _validate_entry(entry, cls)
    return cls  # type: ignore[return-value]


def _validate_entry(entry: RegistryEntry, cls: type) -> None:
    try:
        slug = cls.slug_name
    except AttributeError as exc:
        raise ValueError(f"{entry.kind} class '{entry.attr}' is missing slug_name") from exc
    if str(slug) != entry.slug_name:
        raise ValueError(
            f"{entry.kind} slug mismatch: registry '{entry.slug_name}' "
            f"!= class '{slug}' for {entry.module}.{entry.attr}"
        )
    try:
        display = cls.display_name
    except AttributeError as exc:
        raise ValueError(f"{entry.kind} class '{entry.attr}' is missing display_name") from exc
    if str(display) != entry.display_name:
        raise ValueError(
            f"{entry.kind} display_name mismatch: registry '{entry.display_name}' "
            f"!= class '{display}' for {entry.module}.{entry.attr}"
        )
    if not issubclass(cls, BaseModel):
        raise TypeError(f"{entry.kind} class '{entry.attr}' must be a Pydantic BaseModel.")


__all__ = [
    "RegistryEntry",
    "BENCHMARKS",
    "AGENTS",
    "get_benchmark_entries",
    "get_agent_entries",
    "get_benchmark_subsets",
    "get_benchmark_subset_arg",
    "apply_subset_kwargs",
    "apply_task_kwargs",
    "load_benchmark",
    "load_agent",
]
