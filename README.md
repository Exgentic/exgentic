<img src="misc/assets/exgentic_banner_black.png" alt="Exgentic Banner" width="100%"/>

<p align="center">
  <strong>Evaluate any agent on any benchmark in the simplest way possible</strong>
</p>

---

## What is Exgentic?

Exgentic is a universal evaluation framework that enables standardized testing of AI agents across diverse benchmarks and domains. It provides a consistent interface for evaluating any agent on any benchmark, making it easy to compare performance, reproduce results, and ensure your agent works reliably across different tasks and environments.

## Who is it for?

1. **General Audience** - Visit [www.exgentic.ai](https://www.exgentic.ai) to explore the first general agent leaderboard comparing leading agents and frontier models across varied tasks.
2. **Agent Builders** - Evaluate your agents comprehensively across multiple domains and benchmarks.
3. **Researchers & Component Developers** - Test agentic components (memory, context compression, planning) across different agents and domains.
4. **Benchmark Builders** - Evaluate your benchmark across multiple agents to ensure meaningful differentiation.

---

## Quick Start

### Installation

```bash
uv tool install exgentic
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
exgentic list benchmarks
exgentic list agents

# Evaluate an agent on a benchmark
exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o \
  --set benchmark.user_simulator_model="gpt-4o"
```

Benchmarks are automatically installed on first run — no manual installation needed. You can also install them explicitly:

```bash
exgentic install --benchmark tau2              # install deps + data (default)
exgentic install --agent tool_calling
exgentic install --benchmark tau2 --docker     # build Docker image
exgentic install --benchmark tau2 --local      # install into local environment
exgentic uninstall --benchmark tau2            # remove installed environment
```

> **Note:** `exgentic setup` still works but is deprecated in favor of `install`/`uninstall`.

For full container isolation, use the Docker runner (`--set benchmark.runner=docker`). You only need Docker installed and running:

```bash
exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --model gpt-4o \
  --set benchmark.runner=docker \
  --set benchmark.user_simulator_model="gpt-4o"
```

### Python API

To use exgentic as a library, install it first:

```bash
uv add exgentic   # or: pip install exgentic
```

```python
from exgentic import evaluate

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
exgentic list benchmarks
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

<img src="misc/assets/gui.png" alt="Dashboard" width="100%"/>

```bash
exgentic dashboard
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

<img src="misc/assets/cli.png" alt="CLI" width="100%"/>

```bash
# Discover
exgentic list benchmarks
exgentic list subsets --benchmark tau2
exgentic list tasks --benchmark tau2 --subset retail --limit 5
exgentic list agents
exgentic install --benchmark tau2
exgentic install --benchmark tau2 --docker
exgentic install --benchmark tau2 --local
exgentic uninstall --benchmark tau2

# Run
exgentic evaluate --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10
exgentic batch run --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10

# Inspect
exgentic status --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10
exgentic preview --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10
exgentic results --benchmark tau2 --agent tool_calling --subset airline --num-tasks 10

# Analyze
exgentic compare --agents tool_calling openai --benchmark tau2

# Explore
exgentic dashboard
```

---

## Advanced

### Model Configuration

```bash
exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --set agent.model.temperature=0.2
```

Supported fields: `temperature`, `top_p`, `max_tokens`, `reasoning_effort`, `num_retries`, `retry_after`, `retry_strategy`

### Run Limits

```bash
exgentic evaluate --benchmark tau2 --agent tool_calling --subset retail --num-tasks 2 \
  --max-steps 100 --max-actions 100
```

Sessions stop at either limit and record `limit_reached` status. Default: 100 for both.

### HuggingFace

Use HuggingFace models or run evaluations on HuggingFace Jobs. See [docs/huggingface.md](./docs/huggingface.md).

---

## How It Works

To learn more about Exgentic's architecture and design, see our [arXiv paper](https://arxiv.org/abs/2602.22953).

## Development

For local development, editing, and contributing, see [DEVELOPMENT.md](./DEVELOPMENT.md).

## Contributing

We welcome issues and pull requests! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citing Exgentic

```bibtex
@misc{bandel2026generalagentevaluation,
      title={General Agent Evaluation},
      author={Elron Bandel and Asaf Yehudai and Lilach Eden and Yehoshua Sagron and Yotam Perlitz and Elad Venezian and Natalia Razinkov and Natan Ergas and Shlomit Shachor Ifergan and Segev Shlomov and Michal Jacovi and Leshem Choshen and Liat Ein-Dor and Yoav Katz and Michal Shmueli-Scheuer},
      year={2026},
      url={https://arxiv.org/abs/2602.22953},
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Support

For questions and support, [open an issue](https://github.com/Exgentic/exgentic/issues) on GitHub.
