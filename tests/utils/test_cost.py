# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import os
import sys

import pytest

# Make package importable
sys.path.insert(0, os.path.abspath("src"))
from exgentic.utils.cost import TokensCost, litellm_tokens_cost  # noqa: E402

# All models mentioned across examples and scripts
EXAMPLE_MODELS = [
    "watsonx/meta-llama/llama-3-3-70b-instruct",
    "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
    "watsonx/openai/gpt-oss-120b",
    "openai/Azure/gpt-4.1",
    "openrouter/openai/gpt-oss-120b",
]


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "watsonx/meta-llama/llama-3-3-70b-instruct",
            TokensCost(
                input_cost=7.099999999999999e-05,
                output_cost=7.099999999999999e-05,
                total_cost=0.00014199999999999998,
            ),
        ),
        (
            "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
            TokensCost(
                input_cost=0.00019999999999999998,
                output_cost=0.00019999999999999998,
                total_cost=0.00039999999999999996,
            ),
        ),
        (
            "watsonx/openai/gpt-oss-120b",
            TokensCost(
                input_cost=1.4999999999999999e-05,
                output_cost=5.9999999999999995e-05,
                total_cost=7.5e-05,
            ),
        ),
        (
            "openai/Azure/gpt-4.1",
            TokensCost(
                input_cost=0.00019999999999999998,
                output_cost=0.0007999999999999999,
                total_cost=0.001,
            ),
        ),
        (
            "openai/GCP/gemini-2.5-flash",
            TokensCost(
                input_cost=2.9999999999999997e-05,
                output_cost=0.00025,
                total_cost=0.00028,
            ),
        ),
        (
            "openai/GCP/claude-haiku-4-5-20251001",
            TokensCost(
                input_cost=9.999999999999999e-05,
                output_cost=0.0005,
                total_cost=0.0006000000000000001,
            ),
        ),
        (
            "openrouter/openai/gpt-oss-120b",
            TokensCost(
                input_cost=1.8e-05,
                output_cost=7.999999999999999e-05,
                total_cost=9.8e-05,
            ),
        ),
    ],
)
def test_cost_per_token(name, expected):
    cost = litellm_tokens_cost(model_name=name, input_tokens=100, output_tokens=100)
    print(name, cost)
    assert cost == expected
