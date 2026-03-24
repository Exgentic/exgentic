# HuggingFace

## Using HuggingFace Models

Set your HF token and use the `huggingface/<provider>/<org>/<model>` model string format:

```bash
export HF_TOKEN=hf_...
```

```bash
exgentic evaluate \
  --benchmark gsm8k \
  --agent tool_calling \
  --model huggingface/together/meta-llama/Llama-3.1-70B-Instruct
```

LiteLLM routes the call through HuggingFace's inference providers (billed to your HF account). Supported providers include `together`, `sambanova`, and others. Tool calling support depends on the provider and model.

## Running on HuggingFace Jobs

HuggingFace Jobs run containerized workloads on HF infrastructure (requires Pro/Team/Enterprise).

```bash
hf jobs run astral-sh/uv:python3.12-bookworm sh -c "
  uvx exgentic evaluate \
    --benchmark gsm8k \
    --agent tool_calling \
    --model huggingface/together/meta-llama/Llama-3.1-70B-Instruct \
    --output-dir /tmp/outputs &&
  uvx exgentic batch publish --repo-id your-org/eval-results /tmp/outputs
" --env HF_TOKEN=hf_...
```

Results are published to `https://huggingface.co/datasets/your-org/eval-results`.
