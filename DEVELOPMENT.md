# Development Guide

This guide covers setting up exgentic for local development, editing, and debugging.

## Setup

```bash
git clone https://github.com/Exgentic/exgentic.git
cd exgentic
uv sync
```

## Setup Benchmarks & Agents

Benchmarks and agents have external dependencies installed via setup scripts:

```bash
# Benchmarks
uv run exgentic setup --benchmark tau2
uv run exgentic setup --benchmark appworld
uv run exgentic setup --benchmark gsm8k
uv run exgentic setup --benchmark hotpotqa
uv run exgentic setup --benchmark swebench
uv run exgentic setup --benchmark browsecompplus

# Agents
uv run exgentic setup --agent litellm_tool_calling
uv run exgentic setup --agent smolagents
uv run exgentic setup --agent openai
uv run exgentic setup --agent claude
uv run exgentic setup --agent codex
uv run exgentic setup --agent gemini
```

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
# Full test suite
uv run pytest tests/

# API-level tests only
uv run pytest tests/api

# Skip tests requiring external services
uv run pytest tests/ -k "not litellm and not mcp"
```

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
- Create and push a release tag: `scripts/release.sh 0.2.0 --push`
- After PyPI publish succeeds, create the GitHub Release manually: `gh release create v0.2.0 --generate-notes --title "v0.2.0"`
- Release versions come from Git tags via `hatch-vcs`
- PyPI publishing uses GitHub Actions Trusted Publishing
