# Development Guide

This guide covers setting up exgentic for local development, editing, and debugging.

## Setup

```bash
git clone https://github.com/Exgentic/exgentic.git
cd exgentic
uv sync
```

## Setup Benchmarks & Agents

Benchmarks and agents declare their dependencies through two mechanisms:

- **`requirements.txt`** — pip packages installed automatically via `uv pip install`
- **`setup.sh`** — shell script for non-pip setup (apt packages, git clones, data downloads)

Both are auto-discovered next to the benchmark/agent module directory. The `exgentic install` command runs both:

```bash
# Benchmarks
uv run exgentic install --benchmark tau2
uv run exgentic install --benchmark appworld
uv run exgentic install --benchmark gsm8k
uv run exgentic install --benchmark hotpotqa
uv run exgentic install --benchmark swebench
uv run exgentic install --benchmark browsecompplus

# Agents
uv run exgentic install --agent litellm_tool_calling
uv run exgentic install --agent smolagents
uv run exgentic install --agent openai
uv run exgentic install --agent claude
uv run exgentic install --agent codex
uv run exgentic install --agent gemini
```

> **Note:** `exgentic setup` still works but is deprecated. Use `install`/`uninstall` instead.

### Isolated Runners (venv / docker)

By default, benchmarks run with the `venv` runner, which creates an isolated `uv` virtual environment per benchmark under `.exgentic/<slug>/venv/`. This means **no local setup is needed** — dependencies are installed automatically in the venv on first run.

You can also use the `docker` runner for full container isolation:

```bash
uv run exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o \
  --set benchmark.runner=docker \
  --set benchmark.user_simulator_model="gpt-4o"
```

Both isolated runners follow the same pattern:

1. Install `requirements.txt` and run `setup.sh` in the isolated environment
2. Start `exgentic serve --cls <module:Class> --kwargs <json>` inside the venv/container
3. Communicate over HTTP via the runner transport layer

Setup scripts can check the `EXGENTIC_DOCKER_BUILD` environment variable to distinguish a Docker build from a local setup (e.g., to skip interactive prompts or large downloads that are handled differently in containers).

## API Credentials

```bash
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
```

Or create a `.env` file in the project root — Exgentic loads it automatically.

## Running Evaluations

```bash
uv run exgentic list benchmarks
uv run exgentic list agents

uv run exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o \
  --set benchmark.user_simulator_model="gpt-4o"
```

## Tests

```bash
# Core tests (no Docker or external services required)
uv run pytest tests/ --ignore=tests/integrations --ignore=tests/adapters/runners

# Runner/transport tests (includes Docker tests on matching Python version)
uv run pytest tests/adapters/runners -v -p no:faulthandler

# API-level tests only
uv run pytest tests/api

# Skip tests requiring external services
uv run pytest tests/ -k "not litellm and not mcp"
```

The test suite includes **replay tests** that re-run recorded benchmark sessions without any external dependencies. Recordings are stored under `tests/benchmarks/recordings/` and use `ReplayBenchmark` + `ReplayAgent` to verify the execution loop end-to-end.

## Linting

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## OpenTelemetry Tracing

```bash
uv sync --extra otel

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export EXGENTIC_OTEL_ENABLED=true
```

See [`OTEL_SEMANTIC_CONVENTIONS.md`](./OTEL_SEMANTIC_CONVENTIONS.md) for details.

## Releases

- Release process guide: `docs/releasing.md`
- Benchmark adapter design guide: `docs/adding-benchmarks.md`
- Create and push a release tag: `scripts/release.sh 0.2.0 --push`
- After PyPI publish succeeds, create the GitHub Release manually: `gh release create v0.2.0 --generate-notes --title "v0.2.0"`
- Release versions come from Git tags via `hatch-vcs`
- PyPI publishing uses GitHub Actions Trusted Publishing
