# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

## CLI usage:
# exgentic evaluate --benchmark bfcl --agent tool_calling --subset simple_python --num-tasks 5 \
#   --model gpt-4o
## Python API usage:
# from exgentic import RunConfig, evaluate
# evaluate(RunConfig(benchmark="bfcl", agent="tool_calling", subset="simple_python",
#   num_tasks=5, model="gpt-4o"))
## Direct class usage (this script):
# BFCLBenchmark + LiteLLMToolCallingAgent

from exgentic import BFCLBenchmark, LiteLLMToolCallingAgent, evaluate


def main() -> None:
    benchmark = BFCLBenchmark(subset="simple_python")
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=5)


if __name__ == "__main__":
    main()
