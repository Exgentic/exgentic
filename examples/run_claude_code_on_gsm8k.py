# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

## CLI usage:
# framework evaluate --benchmark gsm8k --agent claude_code --num-tasks 1 \
#   --model gpt-4o
## Python API usage:
# from framework import RunConfig, evaluate
# evaluate(RunConfig(benchmark="gsm8k", agent="claude_code", num_tasks=1,
#   model="gpt-4o"))
## Direct class usage (this script):
# GSM8kBenchmark + ClaudeCodeAgent

from framework import ClaudeCodeAgent, GSM8kBenchmark, evaluate


def main() -> None:
    benchmark = GSM8kBenchmark()
    agent = ClaudeCodeAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=1)


if __name__ == "__main__":
    main()
