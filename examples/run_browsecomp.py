# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

## CLI usage:
# framework evaluate --benchmark browsecompplus --agent tool_calling --subset main --num-tasks 3 \
#   --model gpt-4o
## Python API usage:
# from framework import RunConfig, evaluate
# evaluate(RunConfig(benchmark="browsecompplus", agent="tool_calling", subset="main", num_tasks=3,
#   model="gpt-4o"))
## Direct class usage (this script):
# BrowseCompPlusBenchmark + LiteLLMToolCallingAgent

from framework import BrowseCompPlusBenchmark, LiteLLMToolCallingAgent, evaluate


def main() -> None:
    benchmark = BrowseCompPlusBenchmark()
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=3)


if __name__ == "__main__":
    main()
