# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from typing import Any, Optional, Self, Sequence, TypeVar

from pydantic import BaseModel, computed_field

name_map = {
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "DeepSeek-V3.2": "deepseek/deepseek-v3.2",
    "DeepSeek-V3": "deepseek/deepseek-v3",
    "Kimi-K2.5": "moonshot/kimi-k2.5",
}


class TokensCost(BaseModel):
    input_cost: float
    output_cost: float
    total_cost: float


def _cost_per_token(*, model: str, prompt_tokens: int, completion_tokens: int):
    from litellm.cost_calculator import cost_per_token

    return cost_per_token(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def litellm_cost_per_token(model_name: str):
    return _cost_per_token(model=model_name, prompt_tokens=1, completion_tokens=1)


def litellm_tokens_cost(input_tokens: int, output_tokens: int, model_name: str) -> TokensCost:
    for src, dst in name_map.items():
        model_name = model_name.replace(src, dst)

    parts = model_name.lower().split("/")
    for i in range(len(parts)):
        model = "/".join(parts[-i:])
        try:
            input_cost, output_cost = _cost_per_token(
                model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
            )
            return TokensCost(
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
            )
        except Exception:
            continue
    raise ValueError(
        f"No pricing info found for model '{model_name}'. "
        f"Add it to the name_map in exgentic/utils/cost.py or check litellm model support."
    )


class CostReport(BaseModel):
    _total_cost: float = 0
    model_name: str = ""

    @classmethod
    def initialize_empty(cls, model_name: str = "") -> Self:
        """Create an empty cost report with zero costs."""
        return cls(model_name=model_name)

    @computed_field
    @property
    def total_cost(self) -> float:
        return self._total_cost

    def accumulate_from(self, other: Self) -> None:
        # Accumulate by total_cost only
        self._total_cost += float(other.total_cost)


class UpdatableCostReport(CostReport):
    """A simple accumulator that supports adding arbitrary cost amounts."""

    def add_cost(self, new_cost: float) -> None:
        self._total_cost += new_cost


class LLMCostReport(CostReport):
    """Represents a cost report for LLM usage.

    Usage:
        # Explicit definition
        report = CostReport(model_name="gpt-4", input_tokens=100, output_tokens=50, input_cost=0.02, output_cost=0.03)

        # Empty report
        empty_report = CostReport.initialize_empty("gpt-4")
    """

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @classmethod
    def initialize_empty(cls, model_name: str = "") -> Self:
        """Create an empty cost report with zero tokens and costs."""
        return cls(
            model_name=model_name,
            input_tokens=0,
            output_tokens=0,
            input_cost=0,
            output_cost=0,
        )

    @computed_field
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @computed_field
    @property
    def total_tokens(self) -> float:
        return self.input_tokens + self.output_tokens

    def update_cost(self, input_tokens, output_tokens, input_cost, output_cost) -> None:
        """Update the report with additional tokens and costs."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.input_cost += input_cost
        self.output_cost += output_cost

    def accumulate_from(self, other: Self) -> None:
        """Accumulate costs and tokens from other."""
        self.input_tokens += int(other.input_tokens)
        self.output_tokens += int(other.output_tokens)
        self.input_cost += float(other.input_cost)
        self.output_cost += float(other.output_cost)


class LiteLLMCostReport(LLMCostReport):
    """Specialized cost report that calculates costs using LiteLLM pricing.

    Additional Features:
        - Auto-calculates cost if not provided.
        - Provides helper methods to update cost from token counts.

    Usage:
        report = LiteLLMCostReport.from_token_counts("gpt-4", 100, 50)
        report.update_cost_from_tokens(20, 10)
    """

    output_cost: Optional[float] = None
    input_cost: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.output_cost is None or self.input_cost is None:
            cost_data = LiteLLMCostReport.get_litellm_tokens_cost(
                self.input_tokens, self.output_tokens, model_name=self.model_name
            )
            self.output_cost = cost_data.output_cost
            self.input_cost = cost_data.input_cost

    @classmethod
    def from_token_counts(cls, model_name, input_tokens, output_tokens) -> "LiteLLMCostReport":
        """Create a report from token counts, auto-calculating costs."""
        cost_data = cls.get_litellm_tokens_cost(input_tokens, output_tokens, model_name=model_name)
        return cls(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=cost_data.input_cost,
            output_cost=cost_data.output_cost,
        )

    def update_cost_from_tokens(self, new_input_tokens, new_output_tokens):
        """Update cost based on new token counts."""
        new_cost_data = LiteLLMCostReport.get_litellm_tokens_cost(new_input_tokens, new_output_tokens, self.model_name)
        self.update_cost(
            new_input_tokens,
            new_output_tokens,
            new_cost_data.input_cost,
            new_cost_data.output_cost,
        )

    @classmethod
    def get_litellm_tokens_cost(cls, input_tokens, output_tokens, model_name) -> TokensCost:
        """Fetch cost data from LiteLLM pricing API."""
        if input_tokens == 0 and output_tokens == 0:
            return TokensCost(input_cost=0, output_cost=0, total_cost=0)
        return litellm_tokens_cost(input_tokens, output_tokens, model_name=model_name)


T = TypeVar("T", bound=CostReport)


def accumulate_reports(reports: Sequence[T]) -> T:
    """Accumulate a sequence of same-typed cost reports into a single report of the same type.

    Behavior:
      - Enforces that all items are of the same concrete type.
      - Preserves `model_name` if identical across items; else uses 'mixed'.
      - Uses the report's own `accumulate_from` implementation to merge.

    Args:
      reports: non-empty sequence of reports, all of the same type.

    Returns:
      A new report of the same type, containing the accumulated data.

    Raises:
      ValueError: if the list is empty or contains mixed types.
    """
    if not reports:
        raise ValueError("The reports list cannot be empty.")

    first = reports[0]
    first_type = type(first)

    if not all(type(r) is first_type for r in reports):
        raise ValueError("All reports must be of the same concrete type.")

    # Preserve model_name if consistent; otherwise use 'mixed'
    first_model_name = first.model_name
    same_model_name = all(r.model_name == first_model_name for r in reports)
    acc_report = first_type.initialize_empty(model_name=first_model_name if same_model_name else "mixed")

    # Fold via the type's own accumulation logic
    for r in reports:
        acc_report.accumulate_from(r)

    return acc_report
