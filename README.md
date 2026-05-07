# Framework

<p align="center">
  <strong>Evaluate any agent on any benchmark in the simplest way possible</strong>
</p>

---

## What is Framework?

Framework is a universal evaluation framework that enables standardized testing of AI agents across diverse benchmarks and domains. It provides a consistent interface for evaluating any agent on any benchmark, making it easy to compare performance, reproduce results, and ensure your agent works reliably across different tasks and environments.

## Who is it for?

1. **General Audience** - Visit [www.framework.ai](https://www.framework.ai) to explore the first general agent leaderboard comparing leading agents and frontier models across varied tasks.
2. **Agent Builders** - Evaluate your agents comprehensively across multiple domains and benchmarks.
3. **Researchers & Component Developers** - Test agentic components (memory, context compression, planning) across different agents and domains.
4. **Benchmark Builders** - Evaluate your benchmark across multiple agents to ensure meaningful differentiation.

---

## Quick Start

### Installation

```bash
uv tool install framework
```

### API Credentials

```bash
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
```

### Run an Evaluation

```bash
# List available benchmarks and agents
framework list benchmarks
framework list agents

# Evaluate an agent on a benchmark
framework evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o \
  --set benchmark.user_simulator_model="gpt-4o"
```

Benchmarks are automatically installed on first run — no manual installation needed. You can also install them explicitly:

```bash
framework install --benchmark tau2              # install deps + data (default)
framework install --agent tool_calling
framework install --benchmark tau2 --docker     # build Docker image
framework install --benchmark tau2 --local      # install into local environment
framework uninstall --benchmark tau2            # remove installed environment
```

> **Note:** `framework setup` still works but is deprecated in favor of `install`/`uninstall`.

For full container isolation, use the Docker runner (`--set benchmark.runner=docker`). You only need Docker installed and running:

```bash
framework evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o \
  --set benchmark.runner=docker \
  --set benchmark.user_simulator_model="gpt-4o"
```

### Python API

To use framework as a library, install it first:

```bash
uv add framework   # or: pip install framework
```

```python
from framework import evaluate

results = evaluate(
    benchmark="tau2",
    agent="tool_calling",
    subset="retail",
    num_tasks=2,
    model="gpt-4o",
    benchmark_kwargs={"user_simulator_model": "gpt-4o"},
)
```

For more examples, see the [`examples/`](./examples/) directory.

---

## Available Benchmarks

```bash
framework list benchmarks
```

| Benchmark | Description |
|-----------|-------------|
| **tau2** | Simulated customer support tasks across multiple domains (mock, retail, airline, telecom) |
| **appworld** | Multi-app API environment testing agents' ability to interact with application interfaces |
| **browsecompplus** | Web search and browsing benchmark for information retrieval and navigation |
| **swebench** | Software engineering benchmark for resolving real-world GitHub issues |
| **hotpotqa** | Multi-hop question answering over Wikipedia |
| **gsm8k** | Grade school math word problems with optional calculator tool |
| **bfcl** | Berkeley Function Calling Leaderboard for evaluating tool-use capabilities |

## Available Agents

| Agent | Description |
|-------|-------------|
| **LiteLLM Tool Calling** | Generic tool-calling agent via LiteLLM |
| **SmolAgents** | HuggingFace SmolAgents framework |
| **OpenAI MCP** | OpenAI Responses API with MCP tools |
| **Claude Code** | Anthropic Claude Code agent |
| **Codex CLI** | OpenAI Codex CLI agent |
| **Gemini CLI** | Google Gemini CLI agent |

---

## Dashboard

```bash
framework dashboard
```

---

## Output Structure

Each run creates its own directory under `outputs/<run_id>/`:

```text
outputs/<run_id>/
├── results.json                    # Overall scores, costs, per-session statistics
├── benchmark_results.json          # Benchmark-specific aggregated results
├── aggregator/
│   └── runtime.json               # Run-level evaluator context (list_tasks, aggregation)
├── run/
│   ├── config.json                # Snapshot of benchmark and agent configuration
│   ├── run.log                    # Main execution log
│   └── warnings.log               # Warnings during execution
└── sessions/<session_id>/
    ├── config.json                # Session configuration
    ├── results.json               # Session results
    ├── trajectory.jsonl           # One JSON line per step (action + observation)
    ├── otel.log                   # OpenTelemetry span log (when OTEL is enabled)
    ├── otel_spans.jsonl           # Full OTEL spans as JSONL (when OTEL is enabled)
    ├── agent/
    │   ├── runtime.json          # Per-service context (run_id, session_id, OTEL trace, settings)
    │   └── agent.log             # Agent execution log
    └── benchmark/
        ├── runtime.json          # Per-service context (run_id, session_id, OTEL trace, settings)
        ├── results.json          # Benchmark-specific results
        └── session.log           # Benchmark session log
```

Each service (agent, benchmark) reads its own `runtime.json` on startup to
bootstrap context, settings, and OTEL trace propagation — so subprocesses
launched via `venv`/`docker` runners can attach to the parent's trace
without sharing process memory.

---

## CLI Reference

```bash
# Discover
framework list benchmarks
framework list subsets --benchmark tau2
framework list tasks --benchmark tau2 --subset retail --limit 5
framework list agents
framework install --benchmark tau2
framework install --benchmark tau2 --docker
framework install --benchmark tau2 --local
framework uninstall --benchmark tau2

# Run
framework evaluate --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10
framework batch run --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10

# Inspect
framework status --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10
framework preview --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10
framework results --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10

# Analyze
framework compare --agents tool_calling openai --benchmark tau2

# Explore
framework dashboard
```

---

## Advanced

### Model Configuration

```bash
framework evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --set agent.model.temperature=0.2
```

Supported fields: `temperature`, `top_p`, `max_tokens`, `reasoning_effort`, `num_retries`, `retry_after`, `retry_strategy`

### Run Limits

```bash
framework evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --max-steps 100 --max-actions 100
```

Sessions stop at either limit and record `limit_reached` status. Default: 100 for both.

### HuggingFace

Use HuggingFace models or run evaluations on HuggingFace Jobs. See [docs/huggingface.md](./docs/huggingface.md).

---

## How It Works

To learn more about Framework's architecture and design, see the accompanying paper (anonymized for review).

## Development

For local development, editing, and contributing, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## Contributing

We welcome issues and pull requests! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citing Framework

Citation withheld for double-blind review.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
