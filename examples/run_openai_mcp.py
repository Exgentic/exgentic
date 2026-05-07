# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

## CLI usage:
# framework evaluate --benchmark tau2 --agent openai_solo --subset retail --task 4 \
#   --model gpt-4o --set benchmark.user_simulator_model=gpt-4o
## Python API usage:
# from framework import RunConfig, evaluate
# evaluate(RunConfig(benchmark="tau2", agent="openai_solo", subset="retail", task_ids=["4"],
#   model="gpt-4o",
#   benchmark_kwargs={"user_simulator_model": "gpt-4o"}))
## Direct class usage (this script):
# TAU2Benchmark + OpenAIMCPAgent

from framework import OpenAIMCPAgent, TAU2Benchmark, evaluate


def main() -> None:
    benchmark = TAU2Benchmark(
        subset="retail",
        user_simulator_model="gpt-4o",
    )
    agent = OpenAIMCPAgent(model="gpt-4o")
    evaluate(
        benchmark=benchmark,
        agent=agent,
        output_dir="./outputs",
        task_ids=["4"],
    )


if __name__ == "__main__":
    main()
