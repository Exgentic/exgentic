# CLI Reference

Complete reference for all `framework` CLI commands.

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
framework list benchmarks
```

### list agents

List all available agents.

```bash
framework list agents
```

### list subsets

List subsets for a benchmark.

```bash
framework list subsets --benchmark tau2
```

### list tasks

List task IDs for a benchmark (or subset).

```bash
framework list tasks --benchmark tau2 --subset retail
framework list tasks --benchmark tau2 --subset retail --limit 20
```

| Flag | Description |
|------|-------------|
| `--benchmark` | Benchmark slug (required) |
| `--subset` | Subset name |
| `--limit` | Maximum tasks to show |

---

## install

Install a benchmark's or agent's dependencies (default: isolated venv).

```bash
framework install --benchmark tau2              # install deps + data (default: venv)
framework install --agent tool_calling
framework install --benchmark tau2 --force       # reinstall even if already set up
framework install --benchmark tau2 --docker      # build Docker image
framework install --benchmark tau2 --local       # install into local environment
```

| Flag | Description |
|------|-------------|
| `--benchmark` | Benchmark slug |
| `--agent` | Agent slug |
| `--force` | Force reinstall |
| `--docker` | Build a Docker image |
| `--local` | Install into the local environment instead of an isolated venv |

See [Runners](./runners.md) for details on runner types.

---

## uninstall

Remove an installed benchmark's or agent's environment.

```bash
framework uninstall --benchmark tau2
framework uninstall --agent tool_calling
```

| Flag | Description |
|------|-------------|
| `--benchmark` | Benchmark slug |
| `--agent` | Agent slug |

---

## setup (deprecated)

> **Deprecated:** `framework setup` is an alias for `framework install` and will be removed in a future release. Use `install`/`uninstall` instead.

---

## evaluate

Run an evaluation end-to-end: execute sessions and aggregate results.

```bash
framework evaluate \
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

# Non-standard backend params (api_base, custom auth headers, etc.)
# See docs/custom-models.md → "Non-standard backends"
--set agent.litellm_params_extra='{"api_base":"https://gw.example/v1","extra_headers":{"X-Backend-Auth":"$KEY"}}'
```

---

## status

Show the execution status of a run (how many sessions are done, running, missing).

```bash
framework status --benchmark tau2 --agent tool_calling --subset retail --num-tasks 10
```

Accepts the same flags as `evaluate`.

---

## preview

Show which tasks would run without executing anything.

```bash
framework preview --benchmark tau2 --agent tool_calling --subset retail --num-tasks 10
```

Prints a plan showing which sessions would be new, which already exist, and which are currently running.

---

## results

Load and display results from a completed run.

```bash
framework results --benchmark tau2 --agent tool_calling --subset retail --num-tasks 10
```

Reads `results.json` from the run directory. Accepts the same config flags as `evaluate`.

See [Output Format](./output-format.md) for the full results schema.

---

## compare

Statistical comparison between two run configurations.

```bash
framework compare \
  --agents tool_calling openai_solo \
  --benchmark tau2 \
  --subset retail \
  --num-tasks 50
```

Runs a Breslow-Day homogeneity test across subsets and reports whether the difference between agents is statistically significant.

Requires the `analysis` extra:

```bash
pip install "framework[analysis]"
```

---

## analyze

Generate comparison plots for multiple benchmarks or agents.

```bash
framework analyze \
  --agents tool_calling openai_solo \
  --benchmarks tau2 gsm8k \
  --output report.png
```

Requires the `analysis` extra:

```bash
pip install "framework[analysis]"
```

---

## dashboard

Launch the interactive web dashboard.

```bash
framework dashboard
```

Opens a NiceGUI interface for exploring runs, browsing session trajectories, and monitoring live evaluations.

---

## batch

All batch subcommands. See [Batch Runs](./batch.md) for full documentation.

```bash
framework batch evaluate  --config "configs/*.json"
framework batch execute   --config "configs/*.json"
framework batch aggregate --config "configs/*.json"
framework batch status    --config "configs/*.json"
framework batch prepare   --config run.json [--overwrite]
framework batch patch     --config "configs/*.json" --set key=value [--apply | --dry-run]
framework batch extract   --config "configs/*.json" --output results.csv
framework batch publish   --config "configs/*.json" --repo org/dataset [--append | --overwrite] [--private | --public]
```

---

## Environment variables

Framework reads the following environment variables.

### Framework settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEWORK_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `FRAMEWORK_CACHE_DIR` | `~/.framework` | Cache directory for venvs and setup state |
| `FRAMEWORK_DOTENV_PATH` | `.env` | Path to `.env` file loaded automatically |
| `FRAMEWORK_OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `FRAMEWORK_OTEL_RECORD_CONTENT` | `false` | Include prompts/responses in traces (opt-in) |
| `FRAMEWORK_LITELLM_CACHING` | `true` | Enable LiteLLM response caching |
| `FRAMEWORK_LITELLM_CACHE_DIR` | `~/.cache/framework/litellm` | LiteLLM cache directory |
| `FRAMEWORK_LITELLM_LOG_LEVEL` | `WARNING` | LiteLLM internal log level |

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
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint (optional — omit for local-only file export) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` or `grpc` (only needed with endpoint) |

When `FRAMEWORK_OTEL_ENABLED=true`, each session writes `otel_spans.jsonl` to its output directory regardless of whether a collector is configured. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to also send traces to Jaeger or another collector.

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
