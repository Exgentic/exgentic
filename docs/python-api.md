# Python API

Exgentic can be used as a library. The public API is importable directly from the `exgentic` package.

**Related docs:**
[docs/](./README.md) · [CLI Reference](./cli-reference.md) · [Output Format](./output-format.md) · [Batch Runs](./batch.md) · [Custom Models](./custom-models.md)

---

## Installation

```bash
uv add exgentic   # or: pip install exgentic
```

---

## Quick example

```python
from exgentic import evaluate

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=5,
    model="gpt-4o",
    benchmark_kwargs={"user_simulator_model": "gpt-4o"},
)

print(results.benchmark_score)
print(results.total_agent_cost)
```

---

## Core functions

All functions share the same config parameters. You can pass them as keyword arguments or as a pre-built `RunConfig` object.

### evaluate()

Run sessions and aggregate results. The standard function for most use cases.

```python
from exgentic import evaluate

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=10,
    model="gpt-4o",
    benchmark_kwargs={"user_simulator_model": "gpt-4o"},
    agent_kwargs={"model_settings": {"temperature": 0.2}},
    max_steps=100,
    max_actions=100,
    max_workers=4,
)
```

Returns: `RunResults` — see [Output Format](./output-format.md) for the full schema.

### execute()

Run sessions without aggregating results. Use this when you want to separate execution from aggregation (e.g. run on multiple machines, aggregate centrally).

```python
from exgentic import execute, aggregate

execute(benchmark="tau2", agent="tool_calling", subset="retail", num_tasks=10)
# ... copy outputs to central machine ...
results = aggregate(benchmark="tau2", agent="tool_calling", subset="retail", num_tasks=10)
```

Returns: `RunResults` with aggregation fields empty.

### aggregate()

Aggregate already-completed sessions without running anything. Reads `results.json` from each session directory and computes run-level statistics.

```python
from exgentic import aggregate

results = aggregate(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=10,
)
```

Returns: `RunResults`.

### status()

Get the current execution status of a run without running anything.

```python
from exgentic import status

run_status = status(benchmark="tau2", agent="tool_calling", subset="retail", num_tasks=10)
print(run_status.completed)   # number of completed sessions
print(run_status.running)     # number of currently-running sessions
print(run_status.missing)     # number of not-yet-started sessions
```

Returns: `RunStatus`.

### preview()

Get the execution plan for a run — which sessions would run, which would be reused, etc. — without executing.

```python
from exgentic import preview
from exgentic.interfaces.lib.api import RunConfig

config = RunConfig(benchmark="tau2", agent="tool_calling", subset="retail", num_tasks=10)
plan = preview(config)

print(plan.to_run)      # list of session configs that would run
print(plan.reuse)       # list of already-completed sessions
print(plan.missing)     # list of sessions with no output directory
```

Returns: `RunPlan`.

### results()

Load aggregated results from a completed run's `results.json` on disk.

```python
from exgentic import results
from exgentic.interfaces.lib.api import RunConfig

config = RunConfig(benchmark="tau2", agent="tool_calling", subset="retail", num_tasks=10)
run_results = results(config)
```

Returns: `RunResults`.

---

## Parameters

All core functions accept the same parameters (as kwargs or as a `RunConfig`/`SessionConfig` object).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark` | string \| Benchmark | required | Benchmark slug or instance |
| `agent` | string \| Agent | required | Agent slug or instance |
| `subset` | string | null | Benchmark subset |
| `task_ids` | list[string] | null | Explicit task IDs |
| `num_tasks` | int | null | Number of tasks to run |
| `model` | string | null | Model override (forwarded to the agent) |
| `output_dir` | string | `./outputs` | Results directory |
| `cache_dir` | string | null | Cache directory |
| `run_id` | string | auto | Deterministic run ID derived from config |
| `max_steps` | int | 100 | Maximum steps per session |
| `max_actions` | int | 100 | Maximum actions per session |
| `max_workers` | int | null | Parallel session workers |
| `overwrite_sessions` | bool | false | Re-run completed sessions |
| `benchmark_kwargs` | dict | null | Extra kwargs for the benchmark constructor |
| `agent_kwargs` | dict | null | Extra kwargs for the agent constructor |
| `observers` | list | null | Custom observers (see [Observers](./observers.md)) |
| `controllers` | list | null | Custom controllers (see [Observers](./observers.md)) |

---

## Discovery functions

### list_benchmarks()

```python
from exgentic import list_benchmarks

for b in list_benchmarks():
    print(b["slug_name"], b["display_name"], b["installed"])
```

Returns: `list[dict]` with keys `slug_name`, `display_name`, `installed`, `installed_at`.

### list_agents()

```python
from exgentic import list_agents

for a in list_agents():
    print(a["slug_name"], a["display_name"])
```

Returns: `list[dict]` with keys `slug_name`, `display_name`, `installed`, `installed_at`.

### list_subsets()

```python
from exgentic import list_subsets

subsets = list_subsets("tau2")
# ["retail", "airline", "banking"]
```

Returns: `list[str]`.

### list_tasks()

```python
from exgentic import list_tasks

tasks = list_tasks(benchmark="tau2", subset="retail")
# ["retail_1", "retail_2", ...]
```

Returns: `list[str]`.

---

## Setup functions

### setup_benchmark()

Install a benchmark's dependencies and run its `setup.sh`. Equivalent to `exgentic setup --benchmark <name>`.

```python
from exgentic.interfaces.lib.api import setup_benchmark

setup_benchmark("tau2")
setup_benchmark("tau2", force=True)        # reinstall even if already set up
setup_benchmark("tau2", runner="venv")     # install into isolated venv
```

### setup_agent()

```python
from exgentic.interfaces.lib.api import setup_agent

setup_agent("tool_calling")
setup_agent("tool_calling", force=True)
```

---

## Config objects

Use config objects when you want to construct a run programmatically, save configs to disk, or pass them around.

### RunConfig

```python
from exgentic.interfaces.lib.api import RunConfig

config = RunConfig(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=10,
    model="gpt-4o",
    max_steps=100,
    benchmark_kwargs={"user_simulator_model": "gpt-4o"},
    agent_kwargs={"model_settings": {"temperature": 0.2}},
)

# Save to disk
import json
Path("my_run.json").write_text(config.model_dump_json(indent=2))

# Load from disk
config2 = RunConfig.model_validate_json(Path("my_run.json").read_text())
```

### SessionConfig

For single-task runs:

```python
from exgentic.interfaces.lib.api import SessionConfig

config = SessionConfig(
    benchmark="tau2",
    agent="tool_calling",
    task_id="retail_1",
    subset="retail",
    model="gpt-4o",
)
```

---

## Model settings

Pass model settings through `agent_kwargs`:

```python
from exgentic import evaluate
from exgentic.core.types import ModelSettings

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=5,
    model="gpt-4o",
    agent_kwargs={
        "model_settings": ModelSettings(
            temperature=0.2,
            max_tokens=4096,
            num_retries=3,
        )
    },
)
```

Or as a plain dict (equivalent):

```python
agent_kwargs={
    "model_settings": {
        "temperature": 0.2,
        "max_tokens": 4096,
        "num_retries": 3,
    }
}
```

See [Custom Models](./custom-models.md) for the full `ModelSettings` reference.

---

## Custom observers

Pass observers to receive live callbacks during a run:

```python
from exgentic import evaluate
from exgentic.core.orchestrator.observer import Observer

class PrintObserver(Observer):
    def on_session_success(self, session, score, agent):
        print(f"Session {session.session_id}: score={score.score}")

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    num_tasks=5,
    observers=[PrintObserver()],
)
```

See [Observers](./observers.md) for the full interface reference.

---

## See also

- [Custom Models](./custom-models.md) — LLM provider setup and `ModelSettings`
- [Output Format](./output-format.md) — `RunResults` and `SessionResults` schema
- [Observers](./observers.md) — custom event callbacks
- [Batch Runs](./batch.md) — programmatic equivalents of batch commands
- [CLI Reference](./cli-reference.md) — CLI alternative to the Python API
- [docs/](./README.md) — documentation index
