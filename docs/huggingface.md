# Running Exgentic on HuggingFace

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
uv add huggingface_hub
```

```python
from huggingface_hub import run_job

job = run_job(
    command=["sh", "-c", """
        uv tool install exgentic &&
        exgentic evaluate \
          --benchmark gsm8k \
          --agent tool_calling \
          --model huggingface/together/meta-llama/Llama-3.1-70B-Instruct \
          --output-dir /tmp/outputs &&
        exgentic batch publish --repo-id your-org/eval-results /tmp/outputs
    """],
    environment={
        "HF_TOKEN": "hf_...",
    },
    hardware="cpu-basic",
)
```

Results are published to `https://huggingface.co/datasets/your-org/eval-results`.
