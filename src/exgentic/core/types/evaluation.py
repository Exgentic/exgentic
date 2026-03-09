# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, StrictStr, model_validator

from ...interfaces.registry import (
    apply_subset_kwargs,
    load_agent,
    load_benchmark,
)
from ..context import try_get_context
from .model_settings import ModelSettings


def _validate_kwargs(kind: str, cls: type, kwargs: dict[str, Any]) -> None:
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        return
    cls.model_validate(kwargs)


def _compute_run_id(
    *,
    benchmark: str,
    agent: str,
    benchmark_kwargs: dict[str, Any],
    agent_kwargs: dict[str, Any],
) -> str:
    payload = {
        "benchmark": {
            "slug_name": benchmark,
            "params": benchmark_kwargs,
        },
        "agent": {
            "slug_name": agent,
            "params": agent_kwargs,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


class BaseEvaluationConfig(BaseModel):
    """Shared evaluation config with canonical normalization."""

    benchmark: StrictStr
    agent: StrictStr
    subset: Optional[str] = None
    output_dir: str = "./outputs"
    cache_dir: Optional[str] = None
    run_id: Optional[str] = None
    model: Optional[str] = None
    benchmark_kwargs: Optional[dict[str, Any]] = None
    agent_kwargs: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, values):
        if not isinstance(values, dict):
            return values
        payload = dict(values)
        benchmark = payload.get("benchmark")
        agent = payload.get("agent")
        benchmark_kwargs = payload.get("benchmark_kwargs") or {}
        agent_kwargs = payload.get("agent_kwargs") or {}
        if not isinstance(benchmark_kwargs, dict):
            raise TypeError("benchmark_kwargs must be a dict")
        if not isinstance(agent_kwargs, dict):
            raise TypeError("agent_kwargs must be a dict")

        subset = payload.get("subset")
        if subset is not None and benchmark:
            benchmark_kwargs = apply_subset_kwargs(str(benchmark), str(subset), dict(benchmark_kwargs))
        else:
            benchmark_kwargs = dict(benchmark_kwargs)

        agent_kwargs = dict(agent_kwargs)
        if "model_settings" in agent_kwargs:
            model_cfg = agent_kwargs["model_settings"]
            if model_cfg is None:
                pass
            elif isinstance(model_cfg, ModelSettings):
                pass
            elif isinstance(model_cfg, dict):
                agent_kwargs["model_settings"] = ModelSettings(**model_cfg)
            else:
                raise ValueError("agent.model_settings must be a ModelSettings or dict.")

        model = payload.get("model")
        if model is not None:
            if "model" in agent_kwargs and agent_kwargs["model"] != model:
                raise ValueError("Conflicting model selection: " f"model={agent_kwargs['model']} but model={model}")
            agent_kwargs["model"] = model

        if benchmark:
            bench_cls = load_benchmark(str(benchmark))
            _validate_kwargs("benchmark", bench_cls, benchmark_kwargs)
        if agent:
            agent_cls = load_agent(str(agent))
            _validate_kwargs("agent", agent_cls, agent_kwargs)

        payload["benchmark_kwargs"] = benchmark_kwargs
        payload["agent_kwargs"] = agent_kwargs
        if payload.get("run_id") is None and benchmark and agent:
            ctx = try_get_context()
            payload["run_id"] = (ctx.run_id if ctx else None) or _compute_run_id(
                benchmark=str(benchmark),
                agent=str(agent),
                benchmark_kwargs=benchmark_kwargs,
                agent_kwargs=agent_kwargs,
            )
        return payload

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_context(self):
        from ..context import run_scope

        return run_scope(
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            run_id=self.run_id,
        )

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls.model_validate(payload)
