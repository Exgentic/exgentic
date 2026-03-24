# CLI Reference

Complete reference for all `exgentic` CLI commands.

**Related docs:**
[docs/](./README.md) · [Python API](./python-api.md) · [Batch Runs](./batch.md) · [Custom Models](./custom-models.md) · [Output Format](./output-format.md)

---

## Global flags

| Flag | Description |
|------|-------------|
| `--debug` | Enable debug logging |
| `--help` | Show help for any command |

---

## Discovery

### list benchmarks

List all available benchmarks.

```bash
exgentic list benchmarks
```

### list agents

List all available agents.

```bash
exgentic list agents
```

### list subsets

List subsets for a benchmark.

```bash
exgentic list subsets --benchmark tau2
```

### list tasks

List task IDs for a benchmark (or subset).

```bash
exgentic list tasks --benchmark tau2 --subset retail
exgentic list tasks --benchmark tau2 --subset retail --limit 20
```

| Flag | Description |
|------|-------------|
| `--benchmark` | Benchmark slug (required) |
| `--subset` | Subset name |
| `--limit` | Maximum tasks to show |

---

## Setup

Install a benchmark's or agent's dependencies into the default runner environment.

```bash
exgentic setup --benchmark tau2
exgentic setup --agent tool_calling
exgentic setup --benchmark tau2 --force         # reinstall even if already set up
exgentic setup --benchmark tau2 --runner venv   # install into an isolated venv
```

| Flag | Description |
|------|-------------|
| `--benchmark` | Benchmark slug |
| `--agent` | Agent slug |
| `--force` | Force reinstall |
| `--runner` | Runner type: `direct`, `venv`, `docker` |

See [Runners](./runners.md) for details on runner types.

---

## evaluate

Run an evaluation end-to-end: execute sessions and aggregate results.

```bash
exgentic evaluate \
  --benchmark tau2 \
  --agent tool_calling \
  --subset retail \
  --num-tasks 10 \
  --model gpt-4o \
  --set benchmark.user_simulator_model="gpt-4o"
```

| Flag | Description |
|------|-------------|
| `--benchmark` | Benchmark slug (required) |
| `--agent` | Agent slug (required) |
| `--subset` | Benchmark subset |
| `--task` | One or more specific task IDs (repeatable) |
| `--num-tasks` | Number of tasks to run |
| `--model` | Model override |
| `--max-steps` | Steps per session (default: 100) |
| `--max-actions` | Actions per session (default: 100) |
| `--max-workers` | Parallel session workers |
| `--overwrite` | Re-run already-completed sessions |
| `--output-dir` | Results output directory (default: `./outputs`) |
| `--run-id` | Override the auto-generated run ID |
| `--set KEY=VALUE` | Override any config field (repeatable) |
| `--debug` | Enable debug logging |

### --set syntax

`--set` accepts dotted key paths and JSON-compatible values:

```bash
# Benchmark kwargs
--set benchmark.user_simulator_model="gpt-4o"
--set benchmark.runner=venv

# Agent kwargs
--set agent.max_steps=200

# Model settings
--set agent.model.temperature=0.2
--set agent.model.max_tokens=4096
--set agent.model.top_p=0.9
--set agent.model.reasoning_effort=high
--set agent.model.num_retries=3
--set agent.model.retry_after=1.0
--set agent.model.retry_strategy=constant
```

---

## status

Show the execution status of a run (how many sessions are done, running, missing).

```bash
exgentic status --benchmark tau2 --agent tool_calling --subset retail --num-tasks 10
```

Accepts the same flags as `evaluate`.

---

## preview

Show which tasks would run without executing anything.

```bash
exgentic preview --benchmark tau2 --agent tool_calling --subset retail --num-tasks 10
```

Prints a plan showing which sessions would be new, which already exist, and which are currently running.

---

## results

Load and display results from a completed run.

```bash
exgentic results --benchmark tau2 --agent tool_calling --subset retail --num-tasks 10
```

Reads `results.json` from the run directory. Accepts the same config flags as `evaluate`.

See [Output Format](./output-format.md) for the full results schema.

---

## compare

Statistical comparison between two run configurations.

```bash
exgentic compare \
  --agents tool_calling openai_solo \
  --benchmark tau2 \
  --subset retail \
  --num-tasks 50
```

Runs a Breslow-Day homogeneity test across subsets and reports whether the difference between agents is statistically significant.

Requires the `analysis` extra:

```bash
pip install "exgentic[analysis]"
```

---

## analyze

Generate comparison plots for multiple benchmarks or agents.

```bash
exgentic analyze \
  --agents tool_calling openai_solo \
  --benchmarks tau2 gsm8k \
  --output report.png
```

Requires the `analysis` extra:

```bash
pip install "exgentic[analysis]"
```

---

## dashboard

Launch the interactive web dashboard.

```bash
exgentic dashboard
```

Opens a NiceGUI interface for exploring runs, browsing session trajectories, and monitoring live evaluations.

---

## batch

All batch subcommands. See [Batch Runs](./batch.md) for full documentation.

```bash
exgentic batch evaluate  --config "configs/*.json"
exgentic batch execute   --config "configs/*.json"
exgentic batch aggregate --config "configs/*.json"
exgentic batch status    --config "configs/*.json"
exgentic batch prepare   --config run.json [--overwrite]
exgentic batch patch     --config "configs/*.json" --set key=value [--apply | --dry-run]
exgentic batch extract   --config "configs/*.json" --output results.csv
exgentic batch publish   --config "configs/*.json" --repo org/dataset [--append | --overwrite] [--private | --public]
```

---

## Environment variables

Exgentic reads the following environment variables.

### Exgentic settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EXGENTIC_LOG_LEVEL` | `WARNING` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `EXGENTIC_CACHE_DIR` | `.exgentic` | Cache directory for venvs and setup state |
| `EXGENTIC_DOTENV_PATH` | `.env` | Path to `.env` file loaded automatically |
| `EXGENTIC_OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `EXGENTIC_OTEL_RECORD_CONTENT` | `false` | Include prompts/responses in traces (opt-in) |
| `EXGENTIC_LITELLM_CACHING` | `true` | Enable LiteLLM response caching |
| `EXGENTIC_LITELLM_CACHE_DIR` | `.litellm_cache` | LiteLLM cache directory |
| `EXGENTIC_LITELLM_LOG_LEVEL` | `WARNING` | LiteLLM internal log level |

### LLM provider credentials

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `AZURE_API_KEY` | Azure OpenAI |
| `AZURE_API_BASE` | Azure OpenAI endpoint |
| `AZURE_API_VERSION` | Azure OpenAI API version |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS Bedrock |
| `AWS_REGION_NAME` | AWS Bedrock region |
| `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` | Google Vertex AI |
| `OPENAI_API_BASE` | Custom OpenAI-compatible endpoint |

See [Custom Models](./custom-models.md) for full provider setup instructions.

### OpenTelemetry

| Variable | Description |
|----------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` or `grpc` |

See [Observability Quick Start](./observability/quickstart.md) for tracing setup.

---

## See also

- [Python API](./python-api.md) — programmatic equivalents of all CLI commands
- [Batch Runs](./batch.md) — detailed guide for batch commands
- [Custom Models](./custom-models.md) — LLM provider and `--set agent.model.*` reference
- [Runners](./runners.md) — `--set benchmark.runner=*` options
- [Output Format](./output-format.md) — what `results` and `extract` produce
- [Observability Quick Start](./observability/quickstart.md) — tracing setup
- [docs/](./README.md) — documentation index
