# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

## CLI usage:
# framework evaluate --benchmark appworld --agent tool_calling --subset test_normal --num-tasks 3 \
#   --model gpt-4o
## Python API usage:
# from framework import RunConfig, evaluate
# evaluate(RunConfig(benchmark="appworld", agent="tool_calling", subset="test_normal", num_tasks=3,
#   model="gpt-4o"))
## Direct class usage (this script):
# AppWorldBenchmark + LiteLLMToolCallingAgent

from framework import AppWorldBenchmark, LiteLLMToolCallingAgent, evaluate


def main() -> None:
    benchmark = AppWorldBenchmark(subset="test_normal")
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=3)


if __name__ == "__main__":
    main()
