# Running Exgentic on HuggingFace

## Using HuggingFace Models

Set two environment variables to route LLM calls through HuggingFace's OpenAI-compatible inference router:

```bash
export OPENAI_API_BASE=https://router.huggingface.co/v1
export OPENAI_API_KEY=hf_...
```

Then use any HF model ID prefixed with `openai/`:

```bash
exgentic evaluate \
  --benchmark gsm8k \
  --agent tool_calling \
  --model openai/meta-llama/Llama-3.1-70B-Instruct
```

## Running on HuggingFace Jobs

HuggingFace Jobs run containerized workloads on HF infrastructure (requires Pro/Team/Enterprise).

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import run_job

job = run_job(
    command=["sh", "-c", """
        uv tool install exgentic &&
        exgentic evaluate \
          --benchmark gsm8k \
          --agent tool_calling \
          --model openai/meta-llama/Llama-3.1-70B-Instruct \
          --output-dir /tmp/outputs &&
        exgentic batch publish --repo-id your-org/eval-results /tmp/outputs
    """],
    environment={
        "OPENAI_API_BASE": "https://router.huggingface.co/v1",
        "OPENAI_API_KEY": "hf_...",
        "HF_TOKEN": "hf_...",
    },
    hardware="cpu-basic",
)
```

Results are published to `https://huggingface.co/datasets/your-org/eval-results`.
