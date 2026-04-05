# Exgentic Documentation

Welcome to the Exgentic docs. Use the table below to find what you need.

---

## Using Exgentic

| Document | Description |
|----------|-------------|
| [CLI Reference](./cli-reference.md) | Every command, flag, and environment variable |
| [Python API](./python-api.md) | `evaluate()`, `execute()`, `aggregate()`, `status()`, `list_*()`, and all other library functions |
| [Custom Models](./custom-models.md) | Use any LLM provider (OpenAI, Anthropic, Azure, Bedrock, Ollama, and more) via LiteLLM |
| [Batch Runs](./batch.md) | Run parameter sweeps, manage large evaluations, export to CSV, publish to HuggingFace |
| [Runners](./runners.md) | `direct`, `venv`, `docker` — isolation levels, configuration, Docker-in-Docker |
| [Output Format](./output-format.md) | Schema for `results.json`, `trajectory.jsonl`, session results, and cost reports |
| [Observers](./observers.md) | Hook into the evaluation lifecycle for custom logging, monitoring, and early stopping |

## Extending Exgentic

| Document | Description |
|----------|-------------|
| [Adding Agents](./adding-agents.md) | Write a new agent adapter — design principles, required methods, file layout, and validation checklist |
| [Adding Benchmarks](./adding-benchmarks.md) | Write a new benchmark adapter — design principles, contract rules, and validation checklist |
| [Replay Testing](./replay-testing.md) | Test benchmark and agent adapters end-to-end without API calls, using recorded sessions |

## Observability

| Document | Description |
|----------|-------------|
| [Quick Start](./observability/quickstart.md) | Set up OpenTelemetry tracing with Jaeger in five minutes |
| [Semantic Conventions](./observability/semantic-conventions.md) | Full reference of every span and attribute Exgentic emits |

## Maintainers

| Document | Description |
|----------|-------------|
| [Releasing](./releasing.md) | Cut a release, publish to PyPI, and create a GitHub Release |

---

## Other resources

- [README.md](../README.md) — project overview, quick start, CLI reference, and available benchmarks/agents
- [DEVELOPMENT.md](../DEVELOPMENT.md) — local setup, running tests, linting, and the release process
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contribution workflow, legal requirements, and PR guidelines
