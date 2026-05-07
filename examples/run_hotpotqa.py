# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

## CLI usage:
# framework evaluate --benchmark hotpotqa --agent smolagents_tool --subset distractor --num-tasks 3 \
#   --model gpt-4o --set benchmark.with_search_tools=true
## Python API usage:
# from framework import RunConfig, evaluate
# evaluate(RunConfig(benchmark="hotpotqa", agent="smolagents_tool", subset="distractor", num_tasks=3,
#   model="gpt-4o", benchmark_kwargs={"with_search_tools": True}))
## Direct class usage (this script):
# HotpotQABenchmark + SmolagentToolCallingAgent

from framework import HotpotQABenchmark, SmolagentToolCallingAgent, evaluate


def main() -> None:
    benchmark = HotpotQABenchmark(with_search_tools=True)
    agent = SmolagentToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=3)


if __name__ == "__main__":
    main()
