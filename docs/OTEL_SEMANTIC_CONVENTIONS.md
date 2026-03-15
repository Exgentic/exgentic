# Exgentic → OpenTelemetry GenAI Mapping

This document maps Exgentic core types and members to OpenTelemetry (OTel) GenAI semantic conventions. It reflects the **actual implementation** in `src/exgentic/observers/handlers/otel.py` and `src/exgentic/integrations/litellm/trace_logger.py`.

## Implemented Trace Model

### Span Hierarchy

```
Session Span (ROOT)
├── execute_tool initial_observation
├── chat {model} (LLM inference)
├── execute_tool {tool_name}
├── chat {model} (LLM inference)
├── execute_tool {tool_name}
└── ... (continues until session ends)
```

## Semantic Mapping Table

This table documents **all attributes actually set in the code**, organized by span type and code scope.

| Span Type | OTel Attribute | Exgentic Source | Type | Requirement | Content Filtered | Code Scope | Notes |
|-----------|----------------|-----------------|------|-------------|------------------|------------|-------|
| **Session (ROOT)** | `exgentic.benchmark.slug_name` | BenchmarkEntry.slug_name | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.benchmark.subset` | RunConfig.subset | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.benchmark.agent.name` | AgentEntry.display_name | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.agent.slug` | RunConfig.agent | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `exgentic.run.id` | Context.run_id | string | Custom | No | OtelTracingObserver.on_run_start | Heritable, from run attributes |
| **Session (ROOT)** | `gen_ai.request.model` | RunConfig.model | string | Recommended | No | OtelTracingObserver.on_run_start | Heritable, from run attributes if available |
| **Session (ROOT)** | `gen_ai.conversation.id` | Session.session_id | string | Recommended | No | OtelTracingObserver.on_session_creation | Heritable, primary correlation attribute |
| **Session (ROOT)** | `exgentic.session.id` | Session.session_id | string | Custom | No | OtelTracingObserver.on_session_creation | Heritable, backwards compatibility |
| **Session (ROOT)** | `exgentic.session.task_id` | Session.task_id | string | Custom | No | OtelTracingObserver.on_session_creation | Task identifier |
| **Session (ROOT)** | `exgentic.session.task` | Session.task | string | Opt-In | **Yes** | OtelTracingObserver.on_session_creation | Task prompt/description (user content) |
| **Session (ROOT)** | `exgentic.session.action.{name}.name` | ActionType.name | string | Custom | No | OtelTracingObserver.on_session_creation | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.session.action.{name}.description` | ActionType.description | string | Custom | No | OtelTracingObserver.on_session_creation | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.session.action.{name}.is_message` | ActionType.is_message | bool | Custom | No | OtelTracingObserver.on_session_creation | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.session.action.{name}.is_finish` | ActionType.is_finish | bool | Custom | No | OtelTracingObserver.on_session_creation | For each action in Session.actions |
| **Session (ROOT)** | `exgentic.context.{key}` | Session.context[key] | string | Custom | No | OtelTracingObserver.on_session_creation | For each context key/value |
| **Session (ROOT)** | `exgentic.session.agent.id` | AgentInstance.agent_id | string | Custom | No | OtelTracingObserver.on_session_start | Agent instance ID |
| **Session (ROOT)** | `exgentic.session.agent.path` | AgentInstance.paths.agent_dir | string | Custom | No | OtelTracingObserver.on_session_start | Agent directory path |
| **Session (ROOT)** | `exgentic.score.success` | SessionScore.success | bool | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.score` | SessionScore.score | float | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.score.is_finished` | SessionScore.is_finished | bool | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.session.steps` | Step counter | int | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.agent.agent_cost` | AgentInstance.get_cost() | string (JSON) | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **Session (ROOT)** | `exgentic.session.cost` | Session.get_cost() | string (JSON) | Custom | No | OtelTracingObserver.on_session_success | Set at completion |
| **execute_tool** | `gen_ai.operation.name` | "execute_tool" | string | Required | No | OtelTracingObserver.on_session_start, on_react_success, on_react_error | Constant value |
| **execute_tool** | `gen_ai.tool.name` | Action.name | string | Required | No | OtelTracingObserver.on_session_start, on_react_success, on_react_error | From SingleAction.name |
| **execute_tool** | `gen_ai.tool.id` | Action.id | string | Recommended | No | OtelTracingObserver.on_react_success | From SingleAction.id |
| **execute_tool** | `gen_ai.tool.description` | ActionType.description | string | Recommended | No | OtelTracingObserver.on_session_start, on_react_success | Looked up from Session.actions by tool name |
| **execute_tool** | `gen_ai.tool.parameters` | Action.arguments | string (JSON) | Opt-In | **Yes** | OtelTracingObserver.on_react_success | From SingleAction.arguments (user content) |
| **execute_tool** | `gen_ai.tool.result` | Observation | string | Opt-In | **Yes** | OtelTracingObserver._record_observation | From Observation.to_observation_list() (user content) |
| **execute_tool** | `gen_ai.conversation.id` | Session.session_id | string | Recommended | No | Inherited | Inherited from session span |
| **LLM Inference** | `gen_ai.operation.name` | "chat" or "text_completion" | string | Required | No | TraceLogger._write_otel | Based on presence of messages in kwargs |
| **LLM Inference** | `gen_ai.provider.name` | LitellmKwargs.litellm_params.custom_llm_provider | string | Required | No | TraceLogger._write_otel | Mapped to standard provider names |
| **LLM Inference** | `gen_ai.request.model` | LitellmKwargs.model | string | Required | No | TraceLogger._write_otel | Model identifier from request |
| **LLM Inference** | `error.type` | Exception.class| string | Required | No | TraceLogger._write_otel | If operation failed |
| **LLM Inference** | `gen_ai.conversation.id` | Context.session_id | string | Recommended | No | TraceLogger._write_otel | From Context.session_id |
| **LLM Inference** | `gen_ai.request.max_tokens` | LitellmKwargs.optional_params.max_tokens | int | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.temperature` | LitellmKwargs.optional_params.temperature | float | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.top_p` | LitellmKwargs.optional_params.top_p | float | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.top_k` | LitellmKwargs.optional_params.top_k | float | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.frequency_penalty` | LitellmKwargs.optional_params.frequency_penalty | float | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.presence_penalty` | LitellmKwargs.optional_params.presence_penalty | float | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.stop_sequences` | LitellmKwargs.optional_params.stop | string[] | Recommended | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.request.choice.count` | LitellmKwargs.optional_params.n | int | Required | No | TraceLogger._write_otel | If != 1 |
| **LLM Inference** | `gen_ai.request.seed` | LitellmKwargs.optional_params.seed | int | Required | No | TraceLogger._write_otel | If present in request |
| **LLM Inference** | `gen_ai.response.id` | ResponseObject.id | string | Recommended | No | TraceLogger._write_otel | Response ID from LLM |
| **LLM Inference** | `gen_ai.response.model` | ResponseObject.model | string | Recommended | No | TraceLogger._write_otel | Actual model used by LLM |
| **LLM Inference** | `gen_ai.usage.input_tokens` | ResponseObject.usage.prompt_tokens | int | Recommended | No | TraceLogger._write_otel | Input token count from response |
| **LLM Inference** | `gen_ai.usage.output_tokens` | ResponseObject.usage.completion_tokens | int | Recommended | No | TraceLogger._write_otel | Output token count from response |
| **LLM Inference** | `gen_ai.response.finish_reasons` | ResponseObject.choices[*].finish_reason | string[] | Recommended | No | TraceLogger._write_otel | Array of finish reasons from all choices |
| **LLM Inference** | `gen_ai.tool.definitions` | LitellmKwargs.tools | string (JSON) | Opt-In | **Yes** | TraceLogger._write_otel | Tool definitions from request (user content) |
| **LLM Inference** | `gen_ai.input.messages` | LitellmKwargs.messages | string (JSON) | Opt-In | **Yes** | TraceLogger._write_otel | LLM input messages (user content) |
| **LLM Inference** | `gen_ai.output.messages` | ResponseObject.choices[*].message | string (JSON) | Opt-In | **Yes** | TraceLogger._write_otel | LLM output messages (user content) |

## Span Details

### Session Span (ROOT)
- **Name**: `{benchmark_name} {subset} session`
- **Kind**: INTERNAL (default)
- **Created**: `OtelTracingObserver.on_session_creation`
- **Closed**: `OtelTracingObserver.on_session_success` or `on_session_error`

### execute_tool Span
- **Name**: `execute_tool {tool_name}` or `execute_tool initial_observation`
- **Kind**: CLIENT
- **Created**: `OtelTracingObserver.on_session_start` (initial), `on_react_success`, or `on_react_error`
- **Closed**: `OtelTracingObserver.on_step_success` or `on_step_error`
- **OTel Reference**: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span

### LLM Inference Span
- **Name**: `{operation} {model}` (e.g., "chat gpt-4")
- **Kind**: CLIENT
- **Created**: `TraceLogger._write_otel` (in LiteLLM callback)
- **Closed**: `TraceLogger._write_otel` (with end_time)
- **Parent**: Session span (via OTEL context propagation)
- **OTel Reference**: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference

## Attribute Inheritance

Heritable attributes are set on the session span and automatically propagate to all child spans via `SessionSpanManager.set_heritable_attribute()`:

1. **`gen_ai.conversation.id`** - Primary correlation attribute (Session.session_id)
2. **`exgentic.session.id`** - Backwards compatibility (Session.session_id)
3. **`gen_ai.request.model`** - Model name (RunConfig.model, if available)
4. **`exgentic.run.id`** - Run identifier (Context.run_id)
5. **`exgentic.benchmark.slug_name`** - Benchmark identifier (BenchmarkEntry.slug_name)
6. **`exgentic.benchmark.subset`** - Benchmark subset (RunConfig.subset)
7. **`exgentic.benchmark.agent.name`** - Agent display name (AgentEntry.display_name)
8. **`exgentic.agent.slug`** - Agent identifier (RunConfig.agent)

## Implementation Notes

### Model Name Resolution
Since `AgentInstance` doesn't expose model settings, the model name is extracted from `RunConfig` in `on_run_start`:
```python
model_name = run_config.model or (run_config.agent_kwargs or {}).get("model")
```

### Cost Attributes
Cost objects (`LiteLLMCostReport` and `UpdatableCostReport`) are serialized to JSON strings for OTEL compatibility:
- `exgentic.agent.agent_cost` - JSON string of agent cost report
- `exgentic.session.cost` - JSON string of session cost report

### LLM Span Parent Context
LLM inference spans are created by the LiteLLM trace logger and attached as children of the session span via OTEL context propagation:
1. Session span manager updates the global Context with current OTEL span context (`update_tracing_context()`)
2. LiteLLM trace logger reads the OTEL context from the Context ContextVar
3. LLM spans are created with the session span as parent using `_get_parent_context()`

### Content Recording (Opt-In)

Sensitive content attributes are only recorded when `EXGENTIC_OTEL_RECORD_CONTENT=true`. This follows OpenTelemetry GenAI semantic conventions for opt-in content recording.

**Content Filtered Attributes (marked with "Yes" in table):**

**In otel.py observer:**
- `exgentic.session.task` - Task description (user prompt)
- `gen_ai.tool.result` - Tool execution results (runtime user data)
- `gen_ai.tool.parameters` - Tool call parameters (runtime user data)

**In trace_logger.py (LLM inference):**
- `gen_ai.tool.definitions` - Tool definitions passed to LLM (runtime user data)
- `gen_ai.input.messages` - LLM input messages (user prompts)
- `gen_ai.output.messages` - LLM output messages (model responses)

**NOT Content Filtered (metadata/schemas):**
- All IDs, names, counts, scores, and other metadata
- Action type definitions on session span (static schemas, not runtime data)

### Custom Namespace
All Exgentic-specific attributes use the `exgentic.` prefix to distinguish them from standard OTel semantic conventions.

### Backwards Compatibility
- `exgentic.session.id` maintained alongside `gen_ai.conversation.id`
- All custom attributes preserved with `exgentic.` prefix

## References

- [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Execute Tool Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span)
- [Inference Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#inference)