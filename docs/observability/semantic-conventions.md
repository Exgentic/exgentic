# Semantic Conventions

This document maps Exgentic's core types to [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). The integration lives in `src/exgentic/observers/handlers/otel.py` and `src/exgentic/integrations/litellm/trace_logger.py`.

For setup instructions, see [Quick Start](./quickstart.md).

---

## Span hierarchy

```
{benchmark_name} {subset} session       ← session ROOT (operation: invoke_agent)
├── execute_tool initial_observation
├── chat {model}                        ← LLM inference
├── execute_tool {tool_name}
├── chat {model}                        ← LLM inference
├── execute_tool {tool_name}
└── ...                                 ← continues until session ends
```

The session span is the trace root. Its operation is identified by the `gen_ai.operation.name = "invoke_agent"` attribute (per spec); the span *name* keeps the exgentic-native `{benchmark_name} {subset} session` form so existing dashboards keyed on it continue to work and the benchmark/subset stay visible in trace lists.

Run-level grouping is done by attribute (`exgentic.run.id`, `exgentic.benchmark.slug_name`) rather than by a parent span — sessions in batch mode run in independent processes and don't share an in-process parent. The spec's `invoke_workflow` and `create_agent` spans are optional and not emitted.

---

## Tool observability lifecycle

An agent's tool usage is observable at four distinct stages, each emitted on a different span:

| Stage | Span | Attribute | Source |
|-------|------|-----------|--------|
| 1. Configured capability | invoke_agent (session) | `exgentic.session.tools` | `Session.actions` (framework-level) |
| 2. Offered to LLM | LLM inference | `gen_ai.tool.definitions` | `LitellmKwargs.tools` (per-call) |
| 3. Chosen by LLM | LLM inference | `gen_ai.output.messages[].tool_calls` | provider response |
| 4. Execution result | execute_tool | `gen_ai.tool.result` | `Observation` |

Stage 1 reflects what the agent is configured with — stable over the session. Stage 2 is what was actually sent to the model for a specific inference — may be filtered, reformatted per provider, or vary between calls (e.g., `is_finish` withheld until late). Stages 2–4 are opt-in (`EXGENTIC_OTEL_RECORD_CONTENT=true`).

All LLM calls in exgentic are routed through LiteLLM, so stages 2–3 are guaranteed to emit on every inference.

---

## Attribute reference

The table below documents every attribute defined by the integration, organised by span type.

| Span type | OTel attribute | Exgentic source | Type | Requirement | Content-filtered | Notes |
|-----------|---------------|-----------------|------|-------------|-----------------|-------|
| **invoke_agent (session ROOT)** | `gen_ai.operation.name` | `"invoke_agent"` | string | Required | No | Constant value |
| **invoke_agent (session ROOT)** | `gen_ai.agent.id` | `Agent.slug_name` | string | Conditionally required | No | Stable agent template identifier; one of `agent.id`/`agent.name` is required |
| **invoke_agent (session ROOT)** | `gen_ai.agent.name` | `Agent.display_name` | string | Conditionally required | No | Human-readable agent name |
| **invoke_agent (session ROOT)** | `gen_ai.agent.description` | `Agent.__doc__` | string | Recommended | No | Sourced from the agent class docstring; omitted if absent |
| **invoke_agent (session ROOT)** | `gen_ai.conversation.id` | `Session.session_id` | string | Recommended | No | Heritable; the LLM message thread identifier (distinct from the session entity — see `exgentic.session.id`) |
| **invoke_agent (session ROOT)** | `exgentic.session.id` | `Session.session_id` | string | Custom | No | Heritable; identifies the Session entity (task, score, cost, lifecycle). Used as the routing key by `PerSessionFileExporter`. Same value as `gen_ai.conversation.id` today, but a different concept — kept so consumers can address the session entity directly. |
| **invoke_agent (session ROOT)** | `gen_ai.request.model` | `RunConfig.model` | string | Recommended | No | Heritable; set when model is known at run start |
| **invoke_agent (session ROOT)** | `exgentic.benchmark.slug_name` | `BenchmarkEntry.slug_name` | string | Custom | No | Heritable |
| **invoke_agent (session ROOT)** | `exgentic.benchmark.subset` | `RunConfig.subset` | string | Custom | No | Heritable |
| **invoke_agent (session ROOT)** | `exgentic.benchmark.agent.name` | `agent_entry.slug_name` (falls back to `RunConfig.agent`) | string | Custom | No | Heritable; the resolved agent registry slug (distinct from `gen_ai.agent.name`, which is the human-readable display name) |
| **invoke_agent (session ROOT)** | `exgentic.agent.slug` | `RunConfig.agent` | string | Custom | No | Heritable; the requested agent slug (typically equal to `exgentic.benchmark.agent.name`, but may differ if registry aliasing is in play) |
| **invoke_agent (session ROOT)** | `exgentic.run.id` | `Context.run_id` | string | Custom | No | Heritable; primary key for grouping sessions of one benchmark run |
| **invoke_agent (session ROOT)** | `exgentic.session.task_id` | `Session.task_id` | string | Custom | No | |
| **invoke_agent (session ROOT)** | `exgentic.session.task` | `Session.task` | string | Opt-in | **Yes** | Task prompt; requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **invoke_agent (session ROOT)** | `exgentic.session.tools` | `Session.actions` | string (JSON) | Custom | No | JSON list of all tools available at session start (name, description, is_message, is_finish) |
| **invoke_agent (session ROOT)** | `exgentic.context.{key}` | `Session.context[key]` | string | Custom | No | One entry per context key |
| **invoke_agent (session ROOT)** | `exgentic.session.agent.id` | `AgentInstance.agent_id` | string | Custom | No | Per-instance runtime UUID for the AgentInstance (distinct from `gen_ai.agent.id`, which is the stable template identifier) |
| **invoke_agent (session ROOT)** | `exgentic.session.agent.path` | `AgentInstance.paths.agent_dir` | string | Custom | No | |
| **invoke_agent (session ROOT)** | `exgentic.score.success` | `SessionScore.success` | bool | Custom | No | Set on session close; also emitted as `gen_ai.evaluation.result` event |
| **invoke_agent (session ROOT)** | `exgentic.score` | `SessionScore.score` | float | Custom | No | Set on session close; also emitted as `gen_ai.evaluation.result` event |
| **invoke_agent (session ROOT)** | `exgentic.score.is_finished` | `SessionScore.is_finished` | bool | Custom | No | Set on session close |
| **invoke_agent (session ROOT)** | `exgentic.score.metrics.{key}` | `SessionScore.session_metrics[key]` | primitive or string (JSON) | Custom | No | One entry per metric; nested values JSON-encoded |
| **invoke_agent (session ROOT)** | `exgentic.score.metadata.{key}` | `SessionScore.session_metadata[key]` | primitive or string (JSON) | Custom | No | One entry per metadata field; nested values JSON-encoded |
| **invoke_agent (session ROOT)** | `exgentic.session.steps` | step counter | int | Custom | No | Set on session close |
| **invoke_agent (session ROOT)** | `exgentic.agent.agent_cost` | `AgentInstance.get_cost()` | string (JSON) | Custom | No | Set on session close |
| **invoke_agent (session ROOT)** | `exgentic.session.cost` | `Session.get_cost()` | string (JSON) | Custom | No | Set on session close |
| **execute_tool** | `gen_ai.operation.name` | `"execute_tool"` | string | Required | No | Constant value |
| **execute_tool** | `gen_ai.tool.name` | `Action.name` | string | Required | No | |
| **execute_tool** | `gen_ai.tool.id` | `Action.id` | string | Recommended | No | Matches the `tool_calls[].id` from the model response. The spec has renamed this to `gen_ai.tool.call.id`; tracked separately for the rename. |
| **execute_tool** | `gen_ai.tool.description` | `ActionType.description` | string | Recommended | No | Looked up from `Session.actions` |
| **execute_tool** | `gen_ai.tool.type` | `"function"` | string | Recommended | No | Constant — exgentic actions are always function-style tools |
| **execute_tool** | `gen_ai.tool.parameters` | `Action.arguments` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true`. The spec has renamed this to `gen_ai.tool.call.arguments`; tracked separately for the rename. |
| **execute_tool** | `gen_ai.tool.result` | `Observation` | string | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true`. The spec has renamed this to `gen_ai.tool.call.result`; tracked separately for the rename. |
| **execute_tool** | `gen_ai.conversation.id` | `Session.session_id` | string | Recommended | No | Inherited from invoke_agent span |
| **LLM inference** | `gen_ai.operation.name` | `"chat"` or `"text_completion"` | string | Required | No | |
| **LLM inference** | `gen_ai.provider.name` | `litellm_params.custom_llm_provider` | string | Required | No | Mapped to standard provider names |
| **LLM inference** | `gen_ai.request.model` | `LitellmKwargs.model` | string | Required | No | |
| **LLM inference** | `error.type` | exception class name | string | Required | No | Set on failure |
| **LLM inference** | `gen_ai.conversation.id` | `Context.session_id` | string | Recommended | No | |
| **LLM inference** | `gen_ai.request.max_tokens` | `optional_params.max_tokens` | int | Recommended | No | |
| **LLM inference** | `gen_ai.request.temperature` | `optional_params.temperature` | float | Recommended | No | |
| **LLM inference** | `gen_ai.request.top_p` | `optional_params.top_p` | float | Recommended | No | |
| **LLM inference** | `gen_ai.request.top_k` | `optional_params.top_k` | float | Recommended | No | |
| **LLM inference** | `gen_ai.request.frequency_penalty` | `optional_params.frequency_penalty` | float | Recommended | No | |
| **LLM inference** | `gen_ai.request.presence_penalty` | `optional_params.presence_penalty` | float | Recommended | No | |
| **LLM inference** | `gen_ai.request.stop_sequences` | `optional_params.stop` | string[] | Recommended | No | |
| **LLM inference** | `gen_ai.request.choice.count` | `optional_params.n` | int | Required | No | Only when `n != 1` |
| **LLM inference** | `gen_ai.request.seed` | `optional_params.seed` | int | Required | No | |
| **LLM inference** | `gen_ai.request.stream` | `kwargs.stream` | bool | Recommended | No | Whether the request used streaming mode |
| **LLM inference** | `gen_ai.response.id` | `ResponseObject.id` | string | Recommended | No | |
| **LLM inference** | `gen_ai.response.model` | `ResponseObject.model` | string | Recommended | No | Actual model resolved by the provider |
| **LLM inference** | `gen_ai.usage.input_tokens` | `usage.prompt_tokens` | int | Recommended | No | Includes cached tokens per spec |
| **LLM inference** | `gen_ai.usage.output_tokens` | `usage.completion_tokens` | int | Recommended | No | |
| **LLM inference** | `gen_ai.usage.cache_creation.input_tokens` | `usage.cache_creation_input_tokens` | int | Recommended | No | When prompt caching is used (e.g. Anthropic) |
| **LLM inference** | `gen_ai.usage.cache_read.input_tokens` | `usage.cache_read_input_tokens` | int | Recommended | No | When prompt caching is used |
| **LLM inference** | `gen_ai.usage.reasoning.output_tokens` | `usage.completion_tokens_details.reasoning_tokens` | int | Recommended | No | When the model emits reasoning/thinking tokens |
| **LLM inference** | `gen_ai.response.finish_reasons` | `choices[*].finish_reason` | string[] | Recommended | No | |
| **LLM inference** | `gen_ai.tool.definitions` | `LitellmKwargs.tools` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **LLM inference** | `gen_ai.input.messages` | `LitellmKwargs.messages` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true`. Includes system messages; the spec recommends splitting them into `gen_ai.system_instructions` — tracked separately. |
| **LLM inference** | `gen_ai.output.messages` | `choices[*].message` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |

---

## Span details

### invoke_agent span (session ROOT)

- **Name**: `{benchmark_name} {subset} session` (the spec recommends `invoke_agent {agent_name}` but exgentic keeps the existing name; the operation is identified by the `gen_ai.operation.name` attribute)
- **Kind**: `INTERNAL`
- **Opened**: `OtelTracingObserver.on_session_creation`
- **Closed**: `OtelTracingObserver.on_session_success` or `on_session_error`
- **Reference**: [OTel GenAI invoke_agent span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#invoke-agent-span)

### execute_tool span

- **Name**: `execute_tool {tool_name}` or `execute_tool initial_observation`
- **Kind**: `CLIENT`
- **Opened**: `OtelTracingObserver.on_session_start` (initial), `on_react_success`, or `on_react_error`
- **Closed**: `OtelTracingObserver.on_step_success` or `on_step_error`
- **Reference**: [OTel GenAI execute_tool span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span)

### LLM inference span

- **Name**: `{operation} {model}` (e.g., `chat gpt-4o`)
- **Kind**: `CLIENT`
- **Opened/Closed**: `TraceLogger._write_otel` (LiteLLM callback)
- **Parent**: invoke_agent span (via OTEL context propagation)
- **Reference**: [OTel GenAI inference span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference)

---

## Events

### `gen_ai.evaluation.result`

Emitted on session close as a span event on the `invoke_agent` span. One event for the primary score, plus one event per metric in `SessionScore.session_metrics`.

| Field | Source | Notes |
|-------|--------|-------|
| `gen_ai.evaluation.name` | metric key (`"score"` for the primary event) | Required |
| `gen_ai.evaluation.score.value` | `SessionScore.score` (primary event) or numeric metric value | Set when the value is numeric |
| `gen_ai.evaluation.score.label` | `"success"`/`"failure"` from `SessionScore.success` (primary event), or stringified non-numeric metric value | Set when no numeric value |

---

## Attribute inheritance

The following attributes are set on the invoke_agent span and automatically propagated to all child spans via `SessionSpanManager.set_heritable_attribute()`:

| Attribute | Source |
|-----------|--------|
| `gen_ai.conversation.id` | `Session.session_id` — LLM thread identifier |
| `exgentic.session.id` | `Session.session_id` — Session entity identifier (and exporter routing key) |
| `gen_ai.request.model` | `RunConfig.model` (when available) |
| `exgentic.run.id` | `Context.run_id` |
| `exgentic.benchmark.slug_name` | `BenchmarkEntry.slug_name` |
| `exgentic.benchmark.subset` | `RunConfig.subset` |
| `exgentic.benchmark.agent.name` | `agent_entry.slug_name` |
| `exgentic.agent.slug` | `RunConfig.agent` |

---

## Content filtering

Attributes marked **Yes** in the content-filtered column contain user data (prompts, tool arguments, model responses). They are **not recorded by default** and must be explicitly enabled:

```bash
export EXGENTIC_OTEL_RECORD_CONTENT=true
```

Attributes that are never filtered include all IDs, names, counters, scores, and static schemas — only runtime user content requires opt-in.

---

## Implementation notes

### Spans not emitted

The spec defines two additional agent spans that are intentionally **not** emitted:

- **`invoke_workflow`** — a single parent span over an entire benchmark run does not fit exgentic's runtime: sessions in batch mode run in independent processes that do not share an in-process OTEL context. Run-level grouping is done by attribute (`exgentic.run.id`, `exgentic.benchmark.slug_name`) and computed from session-span aggregates instead.
- **`create_agent`** — would require a new framework lifecycle hook for agent instantiation. May be added in the future to capture agent setup overhead (venv install, container start, model warm-up) separately from session execution.

### Model name resolution

Because `AgentInstance` does not expose model settings, the model name is extracted from `RunConfig` at run start:

```python
model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")
```

### Cost attributes

`LiteLLMCostReport` and `UpdatableCostReport` are serialized to JSON strings for OTEL compatibility:

- `exgentic.agent.agent_cost` — agent-level cost report
- `exgentic.session.cost` — full session cost report

### Agent description and version

`gen_ai.agent.description` is read from the agent's class docstring (`Agent.__doc__`). Subclasses are expected to document their behaviour in the class docstring; if absent, the attribute is omitted.

`gen_ai.agent.version` is not currently emitted — `Agent` has no version field. Implementations may map it to the package version of the agent module if a stable source is desirable.

### LLM span parent context

LLM inference spans are created inside the LiteLLM callback and attached to the invoke_agent span via OTEL context propagation:

1. The session span manager writes the current OTEL context into the `Context` ContextVar via `update_tracing_context()`.
2. The LiteLLM trace logger reads the OTEL context from that ContextVar.
3. LLM spans are created with the invoke_agent span as their parent using `_get_parent_context()`.

---

## References

- [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [GenAI attribute registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)
- [Inference span spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference)
- [execute_tool span spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span)
- [Agent spans spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)
- [GenAI metrics spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)
- [GenAI events spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/)
- [Quick Start](./quickstart.md)
