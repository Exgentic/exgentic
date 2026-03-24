# Output Format

Every evaluation writes structured results to a directory under `outputs/<run_id>/`. This document covers the file layout and the schema of every results file.

**Related docs:**
[docs/](./README.md) · [Python API](./python-api.md) · [Batch Runs](./batch.md) · [CLI Reference](./cli-reference.md)

---

## Directory layout

```
outputs/<run_id>/
├── results.json                    # Aggregated run-level results
├── benchmark_results.json          # Benchmark-specific aggregated results
├── run/
│   ├── config.json                 # Snapshot of RunConfig used for this run
│   ├── run.log                     # Main execution log
│   └── warnings.log                # Warnings captured during execution
└── sessions/<session_id>/
    ├── config.json                 # SessionConfig for this task
    ├── results.json                # Session-level results
    ├── trajectory.jsonl            # One JSON line per step (action + observation)
    ├── agent/
    │   └── agent.log               # Agent execution log
    └── benchmark/
        ├── results.json            # Benchmark-specific session results
        └── session.log             # Benchmark session log
```

### run_id and session_id

Both IDs are deterministic SHA256 hashes of the configuration:

- `run_id` — 12-character prefix of the hash of `(benchmark, agent, subset, model, benchmark_kwargs, agent_kwargs)` — everything except the task list.
- `session_id` — 8-character prefix of the hash of `(benchmark, agent, subset, task_id, model, benchmark_kwargs, agent_kwargs)` — includes the task, so each task gets a stable ID.

This means re-running the same config produces the same IDs. Completed sessions are skipped unless `overwrite_sessions` is set.

---

## results.json

The top-level aggregated results for the entire run.

```json
{
  "benchmark_name": "Tau2Bench",
  "benchmark_slug_name": "tau2",
  "agent_name": "LiteLLM Tool Calling",
  "agent_slug_name": "tool_calling",
  "model_name": "gpt-4o",
  "model_names": ["gpt-4o"],
  "subset_name": "retail",
  "total_sessions": 10,
  "planned_sessions": 10,
  "successful_sessions": 7,

  "benchmark_score": 0.72,
  "average_score": 0.65,

  "average_agent_cost": 0.043,
  "total_agent_cost": 0.43,
  "average_benchmark_cost": 0.021,
  "total_benchmark_cost": 0.21,
  "total_run_cost": 0.64,

  "average_steps": 12.4,
  "average_action_count": 15.1,
  "average_invalid_action_count": 0.8,
  "average_invalid_action_percent": 5.3,

  "percent_finished": 80.0,
  "percent_successful": 70.0,
  "percent_finished_successful": 70.0,
  "percent_finished_unsuccessful": 10.0,
  "percent_unfinished": 10.0,
  "percent_error": 10.0,

  "aggregation_mode": "completed_only",
  "completed_sessions": 10,
  "incomplete_sessions": 0,
  "missing_sessions": 0,
  "aggregated_session_ids": ["a1b2c3d4", "..."],
  "skipped_session_ids": [],
  "skipped_session_reasons": {},

  "exgentic_version": "0.3.0",

  "session_results": [ ... ]
}
```

### Field reference

#### Identity

| Field | Type | Description |
|-------|------|-------------|
| `benchmark_name` | string | Benchmark display name |
| `benchmark_slug_name` | string | Benchmark CLI identifier |
| `agent_name` | string | Agent display name |
| `agent_slug_name` | string | Agent CLI identifier |
| `model_name` | string \| null | Primary model used |
| `model_names` | list[string] \| null | All models used (if multiple) |
| `subset_name` | string \| null | Benchmark subset |
| `exgentic_version` | string \| null | Exgentic version that produced these results |

#### Session counts

| Field | Type | Description |
|-------|------|-------------|
| `total_sessions` | int | Sessions that were executed |
| `planned_sessions` | int \| null | Sessions originally planned (from `num_tasks` or `task_ids`) |
| `successful_sessions` | int | Sessions where `success=true` |

#### Scores

| Field | Type | Description |
|-------|------|-------------|
| `benchmark_score` | float \| null | Primary score from `Benchmark.aggregate_sessions()` — benchmark-specific |
| `average_score` | float \| null | Mean of per-session `score` values |

#### Costs

| Field | Type | Description |
|-------|------|-------------|
| `average_agent_cost` | float \| null | Mean agent API cost per session (USD) |
| `total_agent_cost` | float \| null | Total agent API cost across all sessions |
| `average_benchmark_cost` | float \| null | Mean benchmark API cost per session (e.g. simulator LLM) |
| `total_benchmark_cost` | float \| null | Total benchmark API cost |
| `total_run_cost` | float \| null | Total cost (agent + benchmark) |

#### Performance statistics

| Field | Type | Description |
|-------|------|-------------|
| `average_steps` | float \| null | Mean number of steps per session |
| `average_action_count` | float \| null | Mean number of actions per session |
| `average_invalid_action_count` | float \| null | Mean invalid actions per session |
| `average_invalid_action_percent` | float \| null | Invalid actions as percentage of total |

#### Outcome breakdown

| Field | Type | Description |
|-------|------|-------------|
| `percent_finished` | float \| null | Sessions that reached a terminal state (success or failure) |
| `percent_successful` | float \| null | Sessions with `success=true` |
| `percent_finished_successful` | float \| null | Sessions that finished successfully |
| `percent_finished_unsuccessful` | float \| null | Sessions that finished unsuccessfully |
| `percent_unfinished` | float \| null | Sessions that ran out of steps |
| `percent_error` | float \| null | Sessions that raised an exception |

#### Aggregation provenance

| Field | Type | Description |
|-------|------|-------------|
| `aggregation_mode` | string \| null | Always `"completed_only"` — only completed sessions are aggregated |
| `completed_sessions` | int \| null | Sessions with a `results.json` on disk |
| `incomplete_sessions` | int \| null | Sessions with a directory but no `results.json` |
| `missing_sessions` | int \| null | Planned sessions with no directory at all |
| `aggregated_session_ids` | list[string] \| null | Sessions included in score aggregation |
| `skipped_session_ids` | list[string] \| null | Sessions excluded from aggregation |
| `skipped_session_reasons` | dict \| null | Reason per skipped session ID |

---

## sessions/<session_id>/results.json

Per-session results.

```json
{
  "session_id": "a1b2c3d4",
  "task_id": "retail_1",
  "success": true,
  "score": 1.0,
  "is_finished": true,
  "status": "success",
  "steps": 14,
  "action_count": 17,
  "invalid_action_count": 1,
  "agent_cost": 0.038,
  "benchmark_cost": 0.019,
  "execution_time": 42.3,
  "details": { ... },
  "cost_reports": {
    "agent": { "model_name": "gpt-4o", "input_tokens": 9400, "output_tokens": 820, ... },
    "benchmark": { ... }
  }
}
```

### Field reference

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | 8-char deterministic session ID |
| `task_id` | string \| null | Task identifier from the benchmark |
| `success` | bool | Whether the session ended successfully |
| `score` | float \| null | Benchmark-assigned score for this session (0–1 unless benchmark uses a different scale) |
| `is_finished` | bool \| null | Whether the agent signalled completion (as opposed to hitting a step limit) |
| `status` | string | Session outcome status (see below) |
| `steps` | int | Number of (action → observation) steps executed |
| `action_count` | int | Total individual actions taken |
| `invalid_action_count` | int | Actions that failed schema or contract validation |
| `agent_cost` | float | Estimated agent API cost (USD) |
| `benchmark_cost` | float | Estimated benchmark API cost (USD) |
| `execution_time` | float | Wall-clock time in seconds |
| `details` | object | Full `SessionScore` dump — benchmark-specific |
| `cost_reports` | object | Detailed cost breakdown keyed by `"agent"` and `"benchmark"` |

### Session outcome status

| Value | Meaning |
|-------|---------|
| `success` | Session finished and benchmark scored it as successful |
| `unsuccessful` | Session finished but benchmark scored it as unsuccessful |
| `unfinished` | Agent never returned `None` — ran out of steps |
| `limit_reached` | Hit `max_steps` or `max_actions` |
| `error` | An exception occurred during execution |
| `cancelled` | Run was cancelled before this session completed |
| `unknown` | Status could not be determined |

---

## sessions/<session_id>/trajectory.jsonl

A newline-delimited JSON file with one entry per step. Use this to replay or audit what the agent did.

```json
{"event": "observation", "step": 0, "initial": true, "session_id": "a1b2c3d4", "task_id": "retail_1", "observation": {...}, "action": null}
{"event": "action",      "step": 1, "initial": false, "session_id": "a1b2c3d4", "task_id": "retail_1", "observation": null, "action": {"name": "search_products", "arguments": {...}}}
{"event": "observation", "step": 1, "initial": false, "session_id": "a1b2c3d4", "task_id": "retail_1", "observation": {"content": [...]}, "action": null}
...
```

---

## benchmark_results.json

Benchmark-specific aggregated results produced by `Benchmark.aggregate_sessions()`. Schema is benchmark-defined, but always includes at minimum:

```json
{
  "benchmark_name": "Tau2Bench",
  "total_tasks": 10,
  "score": 0.72,
  "metrics": { ... }
}
```

---

## cost_reports schema

Within session `results.json`, the `cost_reports` dict contains detailed token and cost breakdowns.

```json
{
  "agent": {
    "model_name": "gpt-4o",
    "input_tokens": 9400,
    "output_tokens": 820,
    "input_cost": 0.0235,
    "output_cost": 0.0164,
    "total_cost": 0.0399
  },
  "benchmark": {
    "model_name": "gpt-4o",
    "input_tokens": 4200,
    "output_tokens": 340,
    "input_cost": 0.0105,
    "output_cost": 0.0068,
    "total_cost": 0.0173
  }
}
```

Cost estimates come from LiteLLM's pricing database. For providers or deployments not in the database, costs show as `0`.

---

## Reading results programmatically

```python
import json
from pathlib import Path

run_dir = Path("outputs/abc123def456")

# Load run-level results
results = json.loads((run_dir / "results.json").read_text())
print(f"Score: {results['benchmark_score']}")
print(f"Sessions: {results['total_sessions']}")

# Load a specific session trajectory
session_dir = run_dir / "sessions" / "a1b2c3d4"
trajectory = [
    json.loads(line)
    for line in (session_dir / "trajectory.jsonl").read_text().splitlines()
]
```

Or use the Python API to load and validate:

```python
from exgentic import results
from exgentic.batch import RunConfig

config = RunConfig(benchmark="tau2", agent="tool_calling", subset="retail")
run_results = results(config)
print(run_results.benchmark_score)
```

See [Python API](./python-api.md) for the full API reference.

---

## See also

- [Python API](./python-api.md) — `results()`, `status()`, `aggregate()` functions
- [Batch Runs](./batch.md) — `batch extract` to export results to CSV
- [CLI Reference](./cli-reference.md) — `exgentic results` command
- [docs/](./README.md) — documentation index
