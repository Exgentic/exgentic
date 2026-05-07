# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

## CLI usage:
# framework evaluate --benchmark swebench --agent tool_calling --num-tasks 30 \
#   --model gpt-4o
## Python API usage:
# from framework import RunConfig, evaluate
# evaluate(RunConfig(benchmark="swebench", agent="tool_calling", num_tasks=30,
#   model="gpt-4o"))
## Direct class usage (this script):
# SWEBenchBenchmark + LiteLLMToolCallingAgent

from framework import LiteLLMToolCallingAgent, SWEBenchBenchmark, evaluate


def main() -> None:
    benchmark = SWEBenchBenchmark()
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=30)


if __name__ == "__main__":
    main()
