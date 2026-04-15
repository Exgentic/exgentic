# Semantic Conventions

This document maps Exgentic's core types to [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). It reflects the actual implementation in `src/exgentic/observers/handlers/otel.py` and `src/exgentic/integrations/litellm/trace_logger.py`.

For setup instructions, see [Quick Start](./quickstart.md).

---

## Span hierarchy

```
Session Span (ROOT)
├── execute_tool initial_observation
├── chat {model}                        ← LLM inference
├── execute_tool {tool_name}
├── chat {model}                        ← LLM inference
├── execute_tool {tool_name}
└── ...                                 ← continues until session ends
```

---

## Attribute reference

The table below documents every attribute actually emitted by the implementation, organised by span type.

| Span type | OTel attribute | Exgentic source | Type | Requirement | Content-filtered | Notes |
|-----------|---------------|-----------------|------|-------------|-----------------|-------|
| **Session (ROOT)** | `exgentic.benchmark.slug_name` | `BenchmarkEntry.slug_name` | string | Custom | No | Heritable |
| **Session (ROOT)** | `exgentic.benchmark.subset` | `RunConfig.subset` | string | Custom | No | Heritable |
| **Session (ROOT)** | `exgentic.benchmark.agent.name` | `AgentEntry.display_name` | string | Custom | No | Heritable |
| **Session (ROOT)** | `exgentic.agent.slug` | `RunConfig.agent` | string | Custom | No | Heritable |
| **Session (ROOT)** | `exgentic.run.id` | `Context.run_id` | string | Custom | No | Heritable |
| **Session (ROOT)** | `gen_ai.request.model` | `RunConfig.model` | string | Recommended | No | Heritable; set when model is known at run start |
| **Session (ROOT)** | `gen_ai.conversation.id` | `Session.session_id` | string | Recommended | No | Heritable; primary correlation attribute |
| **Session (ROOT)** | `exgentic.session.id` | `Session.session_id` | string | Custom | No | Heritable; kept for backwards compatibility |
| **Session (ROOT)** | `exgentic.session.task_id` | `Session.task_id` | string | Custom | No | |
| **Session (ROOT)** | `exgentic.session.task` | `Session.task` | string | Opt-in | **Yes** | Task prompt; requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **Session (ROOT)** | `exgentic.session.action.{name}.name` | `ActionType.name` | string | Custom | No | One entry per action in `Session.actions` |
| **Session (ROOT)** | `exgentic.session.action.{name}.description` | `ActionType.description` | string | Custom | No | |
| **Session (ROOT)** | `exgentic.session.action.{name}.is_message` | `ActionType.is_message` | bool | Custom | No | |
| **Session (ROOT)** | `exgentic.session.action.{name}.is_finish` | `ActionType.is_finish` | bool | Custom | No | |
| **Session (ROOT)** | `exgentic.context.{key}` | `Session.context[key]` | string | Custom | No | One entry per context key |
| **Session (ROOT)** | `exgentic.session.agent.id` | `AgentInstance.agent_id` | string | Custom | No | |
| **Session (ROOT)** | `exgentic.session.agent.path` | `AgentInstance.paths.agent_dir` | string | Custom | No | |
| **Session (ROOT)** | `exgentic.score.success` | `SessionScore.success` | bool | Custom | No | Set on session close |
| **Session (ROOT)** | `exgentic.score` | `SessionScore.score` | float | Custom | No | Set on session close |
| **Session (ROOT)** | `exgentic.score.is_finished` | `SessionScore.is_finished` | bool | Custom | No | Set on session close |
| **Session (ROOT)** | `exgentic.score.metrics.{key}` | `SessionScore.session_metrics[key]` | primitive or string (JSON) | Custom | No | One entry per metric; nested values JSON-encoded |
| **Session (ROOT)** | `exgentic.score.metadata.{key}` | `SessionScore.session_metadata[key]` | primitive or string (JSON) | Custom | No | One entry per metadata field; nested values JSON-encoded |
| **Session (ROOT)** | `exgentic.session.steps` | step counter | int | Custom | No | Set on session close |
| **Session (ROOT)** | `exgentic.agent.agent_cost` | `AgentInstance.get_cost()` | string (JSON) | Custom | No | Set on session close |
| **Session (ROOT)** | `exgentic.session.cost` | `Session.get_cost()` | string (JSON) | Custom | No | Set on session close |
| **execute_tool** | `gen_ai.operation.name` | `"execute_tool"` | string | Required | No | Constant value |
| **execute_tool** | `gen_ai.tool.name` | `Action.name` | string | Required | No | |
| **execute_tool** | `gen_ai.tool.id` | `Action.id` | string | Recommended | No | |
| **execute_tool** | `gen_ai.tool.description` | `ActionType.description` | string | Recommended | No | Looked up from `Session.actions` |
| **execute_tool** | `gen_ai.tool.parameters` | `Action.arguments` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **execute_tool** | `gen_ai.tool.result` | `Observation` | string | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **execute_tool** | `gen_ai.conversation.id` | `Session.session_id` | string | Recommended | No | Inherited from session span |
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
| **LLM inference** | `gen_ai.response.id` | `ResponseObject.id` | string | Recommended | No | |
| **LLM inference** | `gen_ai.response.model` | `ResponseObject.model` | string | Recommended | No | Actual model resolved by the provider |
| **LLM inference** | `gen_ai.usage.input_tokens` | `usage.prompt_tokens` | int | Recommended | No | |
| **LLM inference** | `gen_ai.usage.output_tokens` | `usage.completion_tokens` | int | Recommended | No | |
| **LLM inference** | `gen_ai.response.finish_reasons` | `choices[*].finish_reason` | string[] | Recommended | No | |
| **LLM inference** | `gen_ai.tool.definitions` | `LitellmKwargs.tools` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **LLM inference** | `gen_ai.input.messages` | `LitellmKwargs.messages` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |
| **LLM inference** | `gen_ai.output.messages` | `choices[*].message` | string (JSON) | Opt-in | **Yes** | Requires `EXGENTIC_OTEL_RECORD_CONTENT=true` |

---

## Span details

### Session span (ROOT)

- **Name**: `{benchmark_name} {subset} session`
- **Kind**: `INTERNAL`
- **Opened**: `OtelTracingObserver.on_session_creation`
- **Closed**: `OtelTracingObserver.on_session_success` or `on_session_error`

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
- **Parent**: session span (via OTEL context propagation)
- **Reference**: [OTel GenAI inference span](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference)

---

## Attribute inheritance

The following attributes are set on the session span and automatically propagated to all child spans via `SessionSpanManager.set_heritable_attribute()`:

| Attribute | Source |
|-----------|--------|
| `gen_ai.conversation.id` | `Session.session_id` — primary correlation key |
| `exgentic.session.id` | `Session.session_id` — backwards compatibility |
| `gen_ai.request.model` | `RunConfig.model` (when available) |
| `exgentic.run.id` | `Context.run_id` |
| `exgentic.benchmark.slug_name` | `BenchmarkEntry.slug_name` |
| `exgentic.benchmark.subset` | `RunConfig.subset` |
| `exgentic.benchmark.agent.name` | `AgentEntry.display_name` |
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

### Model name resolution

Because `AgentInstance` does not expose model settings, the model name is extracted from `RunConfig` at run start:

```python
model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")
```

### Cost attributes

`LiteLLMCostReport` and `UpdatableCostReport` are serialized to JSON strings for OTEL compatibility:

- `exgentic.agent.agent_cost` — agent-level cost report
- `exgentic.session.cost` — full session cost report

### LLM span parent context

LLM inference spans are created inside the LiteLLM callback and attached to the session span via OTEL context propagation:

1. The session span manager writes the current OTEL context into the `Context` ContextVar via `update_tracing_context()`.
2. The LiteLLM trace logger reads the OTEL context from that ContextVar.
3. LLM spans are created with the session span as their parent using `_get_parent_context()`.

---

## References

- [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [execute_tool span spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span)
- [Inference span spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference)
- [Quick Start](./quickstart.md)
