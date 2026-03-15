# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

## CLI usage:
# exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 1 \
#   --model gpt-4o --set benchmark.user_simulator_model=gpt-4o
## Python API usage:
# from exgentic import RunConfig, evaluate
# evaluate(RunConfig(benchmark="tau2", agent="tool_calling", subset="retail", num_tasks=1,
#   model="gpt-4o", benchmark_kwargs={"user_simulator_model": "gpt-4o"}))
## Direct class usage (this script):
# TAU2Benchmark + LiteLLMToolCallingAgent

from exgentic import LiteLLMToolCallingAgent, TAU2Benchmark, evaluate


def main() -> None:
    benchmark = TAU2Benchmark(subset="retail", user_simulator_model="gpt-4o")
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=1)


if __name__ == "__main__":
    main()
