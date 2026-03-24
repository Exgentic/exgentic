# Adding Benchmarks

This document defines the benchmark design principles for Exgentic.

It is intentionally opinionated. A benchmark adapter should not just "work"; it should preserve the benchmark's meaning while still fitting Exgentic's agent abstraction cleanly.

Use these existing adapters as reference points:
- `src/exgentic/benchmarks/tau2/tau2_benchmark.py`
- `src/exgentic/benchmarks/bfcl/bfcl_benchmark.py`

**Related docs:**
[docs/](./README.md) · [Adding Agents](./adding-agents.md) · [Runners](./runners.md) · [Replay Testing](./replay-testing.md) · [Output Format](./output-format.md) · [CONTRIBUTING.md](../CONTRIBUTING.md)

## Core Principle

The benchmark owns the contract.

That means the benchmark decides:
- what the task is
- what context the agent receives
- what actions exist
- how steps progress
- when a session is finished
- how scoring works

The agent should adapt to the benchmark contract through Exgentic's normal interfaces. The benchmark should not be shaped around one specific model protocol.

The default goal should be the thinnest possible benchmark wrapper.

That means:
- reuse the source benchmark wherever possible
- add only the translation layers that are actually necessary
- avoid reimplementing benchmark logic unless there is a clear reason
- avoid introducing runtime behavior that exists only to satisfy one agent or one model protocol

The target is simple:
- make the benchmark accessible to any Exgentic agent
- while adding the minimum adapter surface necessary
- and without clashing with agent-specific assumptions

## Principles

### 1. Keep the agent-facing contract protocol-agnostic

Do not define a benchmark in terms of OpenAI tool calls, raw assistant messages, or any other provider-specific response format.

Define it in terms of:
- task semantics
- available actions
- observations
- finish conditions
- score

Protocol-specific translation belongs in adapters, not in the benchmark contract.

Bad:
- "The model must return all tool calls in one assistant message."

Good:
- "The task is complete when the required actions have been taken and the benchmark-specific finish condition is met."

### 2. The task should be the real task

`task` should contain the actual task the agent is meant to solve.

Do not wrap the task in fake chat scaffolding unless that scaffolding is genuinely part of the benchmark.

If the benchmark is not about user interaction, do not invent a chat conversation just to make it look conversational.

Bad:
- synthetic "user" messages when the benchmark is not actually testing user interaction
- generic wrapper prompts replacing the real benchmark task

Good:
- the benchmark prompt itself is the `task`

### 3. Context should contain only what the agent should know

`context` is not a metadata dump.

It should contain only information that is necessary for the agent to behave correctly on the task.

Keep internal benchmark metadata out of `context`, including:
- subset names
- source dataset ids
- registry information
- adapter implementation details

Good context:
- policy text
- execution constraints
- information the agent genuinely needs to act correctly

Bad context:
- `"subset": "live_parallel_multiple"`
- `"benchmark": "bfcl"`

### 4. Actions should represent semantic operations

Actions are the benchmark's action space.

Name and describe them in terms of what they do, not in terms of a transport protocol.

Prefer "actions" over protocol-specific terms like "tool calls" in benchmark-facing language, because not all agents consume or produce actions through the same protocol.

If the source benchmark exposes functions, commands, or tools, translate those into Exgentic actions at the boundary.

### 5. Use `finish` only as part of the benchmark contract

`finish` is valid when the benchmark needs an explicit end-of-step or end-of-task signal.

It should exist because the benchmark contract needs it, not because a specific model API needs it.

Use it when:
- the benchmark has multiple steps or turns and needs an explicit transition point
- the benchmark needs a clear "done with this step" signal
- the benchmark should allow completion without another normal action

Do not force `finish` into a benchmark if the source benchmark's semantics are cleaner without it.

### 6. Distinguish execution modes by contract, not by protocol

If a benchmark has single-turn, live, or multi-turn variants, define those as execution contracts.

The important differences are things like:
- whether more steps may follow
- whether the task ends after the current finish
- whether action outputs affect later state
- whether the benchmark continues after a step completes

Do not define the mode in terms of how many assistant messages or tool-call payloads a model is allowed to emit.

Important:
- single-turn does not necessarily mean a single action
- multi-action single-turn tasks are valid
- the distinction is about step structure, not about one specific model protocol

### 7. Action outputs must be honest

If the benchmark can produce real execution outputs, use them.

If it cannot, do not fabricate realistic outputs that imply more runtime semantics than actually exist.

Be explicit in the contract when actions are only being recorded rather than executed.

Good:
- real execution results when the source benchmark exposes an official executor
- `Action recorded.` when there is no real runtime execution for that task family

Bad:
- made-up outputs that look like real environment state changes when none were actually computed

### 8. Reuse external harnesses as the source of truth where possible

When adapting an external benchmark, prefer to reuse:
- dataset loading
- official assets
- ground-truth files
- official checkers or scorers
- official execution helpers

Avoid copying large chunks of benchmark logic into Exgentic if the source repository already provides them.

But there is an important boundary:
- external harnesses should be the source of truth for benchmark assets and scoring
- they should not automatically own the Exgentic runtime contract

If the external harness assumes a model-specific interaction pattern, Exgentic should usually keep its own runtime and bridge to the harness at load/score time instead.

When choosing between two valid integrations, prefer the thinner one.

Use the more complex approach only when the thinner one would:
- distort benchmark meaning
- hard-code one agent's assumptions
- or force Exgentic to own logic that should stay with the source benchmark

### 9. Be explicit about what is official and what is adapted

If the adapter preserves official scoring but changes runtime behavior, document that clearly.

If some subsets use official execution while others only use official scoring, document that too.

Do not imply full equivalence when the integration is intentionally more abstract than the source benchmark.

For each benchmark adapter, it should be easy to answer:
- What comes directly from the source benchmark?
- What is adapted by Exgentic?
- What is exact?
- What is approximate?

### 10. Success, failure, and error must stay distinct

Finished benchmark failures are not the same as runtime errors.

The adapter should keep these states separate:
- success: benchmark completed and passed
- unsuccessful: benchmark completed and failed
- unfinished: benchmark did not complete
- error: adapter or runtime failure prevented a proper benchmark result

Do not swallow real errors and report them as ordinary failures.

If an exception happens, record it explicitly in session metadata.

### 11. The benchmark should work for many agents, not just one

A benchmark adapter should not depend on modifying one particular agent implementation.

Prefer to build benchmark logic around Exgentic's shared abstractions:
- `task`
- `context`
- `actions`
- observations
- `Session.start()`
- `Session.step()`
- `Session.done()`
- `Session.score()`

If the adapter only works because one agent has special behavior, the adapter is too coupled.

### 12. Keep setup, runtime, and registration separate

A well-structured benchmark adapter usually has three separate concerns:

1. Setup
- external checkout or installation
- pinned dependencies
- benchmark-specific environment preparation

2. Runtime
- session logic
- task loading
- action translation
- scoring

3. Registration
- registry entry
- subset listing
- CLI discoverability

Do not mix setup logic directly into the runtime path when it can be handled once in `setup.sh`.

## Recommended Structure

For a benchmark package under `src/exgentic/benchmarks/<name>/`, prefer:

- `<name>_benchmark.py`
  - benchmark and session runtime
- `setup.sh`
  - benchmark installation/bootstrap
- optional shim module
  - thin import boundary around an external harness
- optional helper modules
  - action translation, scoring helpers, data parsing

Then register it in:
- `src/exgentic/interfaces/registry.py`

## Validation Checklist

Before opening a PR for a new benchmark, validate all of the following.

### Contract validation

- `task` is the actual task, not fake wrapper chat
- `context` contains only agent-relevant information
- actions are semantically named
- `finish` exists only if the benchmark contract needs it
- success/failure/error semantics are distinct

### Functional validation

- benchmark is discoverable through the registry
- subsets list correctly
- tasks list correctly
- setup script works from a clean environment
- at least one happy-path task works end to end
- at least one failure-path task is represented correctly
- adapter errors surface as errors, not silent failures

### Source-of-truth validation

- official assets are reused where possible
- official scoring is reused where possible
- any remaining deviations from the source benchmark are documented explicitly

### Quality validation

- `py_compile` passes for changed Python files
- `pre-commit` passes for changed files
- `git diff --check` passes

## Practical Rule Of Thumb

When in doubt, ask:

1. Is this benchmark contract describing the task, or just mirroring one model API?
2. Is this information something the agent should truly know?
3. Is this the thinnest adapter that still preserves the benchmark's meaning?
4. Am I reusing the source benchmark where it helps, without letting it dictate the wrong runtime shape?
5. Are the benchmark outputs honest about what was actually executed?
6. Will this adapter still make sense for a very different kind of Exgentic agent?

If the answer to any of those is no, the adapter is probably too coupled or too misleading.

---

## See also

- [Adding Agents](./adding-agents.md) — the other side of the contract
- [Runners](./runners.md) — how setup.sh and requirements.txt are discovered and executed
- [Replay Testing](./replay-testing.md) — write end-to-end tests for your benchmark without API calls
- [Output Format](./output-format.md) — trajectory.jsonl and results.json schemas
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR workflow and legal requirements
- [docs/](./README.md) — documentation index
