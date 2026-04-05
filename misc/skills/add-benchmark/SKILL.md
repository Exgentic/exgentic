---
name: add-benchmark
description: Use when adding or updating a benchmark adapter in the Exgentic repository. Follow the repository benchmark principles, keep the benchmark contract protocol-agnostic, prefer the thinnest possible wrapper that makes the benchmark accessible to any Exgentic agent, reuse external harness assets and scoring where possible, and validate the adapter with representative smoke tests before finishing.
---

# Add Benchmark

Use this skill when working on benchmark adapters in the Exgentic repository.

## First read

Start with:
- `docs/adding-benchmarks.md`
- `src/exgentic/core/benchmark.py`
- `src/exgentic/interfaces/registry.py`

Then inspect the most relevant existing adapters:
- `src/exgentic/benchmarks/tau2/tau2_benchmark.py`
- `src/exgentic/benchmarks/bfcl/bfcl_benchmark.py`

## Workflow

1. Define the benchmark contract before writing code.
   Decide the real `task`, the agent-relevant `context`, the semantic `actions`, the finish condition, and the scoring boundary.

2. Keep the agent-facing contract protocol-agnostic.
   Do not define the benchmark in terms of one provider's chat or tool-calling format.

3. Prefer the thinnest possible wrapper.
   Make the benchmark accessible to any Exgentic agent with the minimum translation surface necessary. Do not add extra abstraction, copied logic, or runtime machinery unless it is needed to preserve benchmark meaning.

4. Decide the source-of-truth boundary.
   Reuse external benchmark assets, setup, and scoring where possible, but do not let an external harness dictate the wrong runtime contract for Exgentic.

5. Implement runtime, setup, and registration separately.
   Prefer a benchmark module, a `setup.sh`, and a registry entry with clear responsibilities.

6. Validate the adapter as a benchmark, not just as code.
   Check task listing, subset listing, happy-path scoring, failure-path scoring, and error semantics.

## Non-negotiable rules

- `task` must be the actual task.
- `context` must include only what the agent should know.
- subset names and internal metadata stay out of `context`.
- Prefer the thinnest wrapper that preserves the benchmark's meaning.
- Actions should describe semantic operations, not protocol artifacts.
- Use `finish` only when it is part of the benchmark contract.
- If outputs are not real execution results, say so plainly in the benchmark contract.
- Keep success, unsuccessful completion, unfinished runs, and errors distinct.

## Validation

Before finishing, run at least:
- `python -m py_compile` on changed benchmark files
- `pre-commit run --files ...`
- `git diff --check`

Also run benchmark-specific smoke tests that prove:
- one passing case works
- one failing case is represented correctly
- one real error is surfaced as an error
