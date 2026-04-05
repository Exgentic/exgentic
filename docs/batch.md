# Batch Runs

The `batch` commands let you manage large evaluations across multiple configurations — parameter sweeps, multi-benchmark comparisons, re-runs of failed sessions — without writing orchestration scripts.

**Related docs:**
[docs/](./README.md) · [CLI Reference](./cli-reference.md) · [Python API](./python-api.md) · [Output Format](./output-format.md)

---

## When to use batch vs evaluate

| Scenario | Command |
|----------|---------|
| Single benchmark run | `exgentic evaluate` |
| Multiple models on one benchmark | `batch evaluate` with config files |
| Re-run only failed sessions | `batch evaluate` (skips completed by default) |
| Sweep temperature/model grids | `batch evaluate` + `batch patch` |
| Publish results to HuggingFace | `batch publish` |
| Export results to CSV | `batch extract` |

---

## Config files

Every batch command operates on **config files** — JSON files that describe a run. There are two kinds.

### RunConfig

Describes a full multi-task run.

```json
{
  "benchmark": "tau2",
  "agent": "tool_calling",
  "subset": "retail",
  "num_tasks": 10,
  "model": "gpt-4o",
  "benchmark_kwargs": {
    "user_simulator_model": "gpt-4o"
  },
  "agent_kwargs": {
    "model_settings": {
      "temperature": 0.2
    }
  },
  "output_dir": "./outputs",
  "max_steps": 100,
  "max_actions": 100
}
```

### SessionConfig

Describes a single task. Used when you need per-task control or when replaying individual sessions.

```json
{
  "benchmark": "tau2",
  "agent": "tool_calling",
  "task_id": "retail_1",
  "subset": "retail",
  "model": "gpt-4o",
  "output_dir": "./outputs"
}
```

### Full field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `benchmark` | string | required | Benchmark slug (e.g. `tau2`) |
| `agent` | string | required | Agent slug (e.g. `tool_calling`) |
| `subset` | string | null | Benchmark subset |
| `task_ids` | list[string] | null | Explicit task IDs to run |
| `num_tasks` | int | null | Number of tasks (randomly sampled if task_ids not set) |
| `task_id` | string | required (SessionConfig) | Single task ID |
| `model` | string | null | Model override |
| `output_dir` | string | `./outputs` | Where to write results |
| `cache_dir` | string | null | Cache directory |
| `run_id` | string | auto | Deterministic ID derived from config |
| `max_steps` | int | 100 | Steps per session |
| `max_actions` | int | 100 | Actions per session |
| `max_workers` | int | null | Parallel session workers |
| `overwrite_sessions` | bool | false | Re-run already-completed sessions |
| `benchmark_kwargs` | object | null | Extra kwargs passed to the benchmark |
| `agent_kwargs` | object | null | Extra kwargs passed to the agent |

---

## Commands

All batch commands accept one or more `--config` flags, each taking a file path or a glob pattern.

```bash
exgentic batch <subcommand> --config path/to/config.json
exgentic batch <subcommand> --config "configs/*.json"
exgentic batch <subcommand> --config configs/run1.json --config configs/run2.json
```

---

### batch evaluate

Run all configs sequentially, executing sessions and aggregating results.

```bash
exgentic batch evaluate --config "configs/*.json"
```

Already-completed sessions are skipped unless `overwrite_sessions` is true in the config. This makes it safe to re-run after partial failures — only missing or failed sessions are executed.

---

### batch execute

Same as `batch evaluate` but skips the aggregation step. Use this when you want to run sessions and aggregate later.

```bash
exgentic batch execute --config "configs/*.json"
exgentic batch aggregate --config "configs/*.json"   # aggregate afterwards
```

---

### batch aggregate

Aggregate results from already-completed sessions without running anything.

```bash
exgentic batch aggregate --config "configs/*.json"
```

Useful when you have sessions from a previous run and want to recompute scores.

---

### batch status

Print a status table showing completion state for each config.

```bash
exgentic batch status --config "configs/*.json"
```

---

### batch prepare

Write session config files to disk without executing. Creates the session directory structure so you can inspect or modify configs before running.

```bash
exgentic batch prepare --config run.json
exgentic batch prepare --config run.json --overwrite   # overwrite existing session configs
```

---

### batch patch

Modify existing run or session config files in bulk using dotted-key notation.

```bash
# Preview what would change
exgentic batch patch --config "configs/*.json" \
  --set model=gpt-4o \
  --dry-run

# Apply changes
exgentic batch patch --config "configs/*.json" \
  --set model=gpt-4o \
  --set agent_kwargs.model_settings.temperature=0.2 \
  --apply
```

Dotted paths are resolved into nested dicts. Values are parsed as JSON first; if that fails, treated as strings. This lets you do sweeps:

```bash
# Change model across a whole grid of configs
exgentic batch patch --config "sweep_*.json" --set model=claude-3-5-sonnet-20241022 --apply
```

---

### batch extract

Export results from multiple runs into a single CSV file.

```bash
exgentic batch extract --config "configs/*.json" --output results.csv
exgentic batch extract --config "configs/*.json" --output -   # print to stdout
```

Each row is one run. Columns include all `RunResults` fields (see [Output Format](./output-format.md)).

---

### batch publish

Push results to a [HuggingFace dataset](https://huggingface.co/docs/datasets/).

```bash
exgentic batch publish \
  --config "configs/*.json" \
  --repo Exgentic/open-agent-leaderboard-results \
  --append
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | required | HuggingFace dataset repo ID |
| `--append` / `--overwrite` | `--append` | Append to or replace the existing dataset |
| `--private` / `--public` | `--private` | Dataset visibility |

Deduplication: when appending, existing rows with the same `(benchmark, agent, model)` triple are replaced. New combinations are appended.

Requires the `datasets` package (`pip install datasets`) and a HuggingFace token with write access:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

---

## Typical workflows

### Parameter sweep

```bash
# Create one config per model
for model in gpt-4o claude-3-5-sonnet-20241022 gemini-2.0-flash; do
  cp base_config.json "configs/${model}.json"
  exgentic batch patch --config "configs/${model}.json" --set model=${model} --apply
done

# Run all
exgentic batch evaluate --config "configs/*.json"

# Export to CSV
exgentic batch extract --config "configs/*.json" --output sweep_results.csv
```

### Resume after partial failure

```bash
# Just re-run — completed sessions are skipped automatically
exgentic batch evaluate --config "configs/*.json"
```

### Separate execute from aggregate

```bash
# Run sessions in parallel across machines, then aggregate centrally
exgentic batch execute --config "configs/*.json"
# ... copy outputs to aggregation machine ...
exgentic batch aggregate --config "configs/*.json"
```

---

## See also

- [CLI Reference](./cli-reference.md) — full flag reference for all commands
- [Output Format](./output-format.md) — RunResults schema, what batch extract produces
- [Python API](./python-api.md) — programmatic equivalents: `evaluate()`, `execute()`, `aggregate()`
- [docs/](./README.md) — documentation index
