# SWE-bench

Evaluate agents on real-world GitHub issues using the [SWE-bench](https://github.com/princeton-nlp/SWE-bench) benchmark.

## Requirements

- Linux x86_64 with Docker
- Python 3.11 in a virtual environment
- Run setup once: `bash src/framework/benchmarks/swebench/setup.sh`

## Usage

```python
from framework.core.orchestrator import run
from framework.benchmarks.swebench.swebench_benchmark import SWEBenchBenchmark
from framework.agents.litellm_tool_calling.litellm_tool_calling_agent import LiteLLMToolCallingAgent

benchmark = SWEBenchBenchmark(
    subset="princeton-nlp/SWE-bench_Lite",
    num_tasks=10,
)
agent = LiteLLMToolCallingAgent(model="gpt-4o", max_steps=30)
results = run(benchmark, agent, output_dir="./outputs/swebench")
print("Avg score:", results.score)
```

## Output Structure

Each session writes to `outputs/<run_id>/sessions/<session_id>/benchmark/`:

```
session.log        - benchmark execution log
predictions.jsonl  - generated patch
logs/              - SWE-bench harness evaluation logs
results.json       - evaluation results (resolved, test pass rates)
```

## Notes

- Each task runs in a dedicated Docker container via `minisweagent`
- Patches are evaluated using the official SWE-bench harness
- Invalid patches (missing diff markers) are rejected without evaluation
- Heavy projects (e.g., astropy) may need increased Docker RAM/CPU limits
