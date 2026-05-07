# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Anonymous Authors.

from __future__ import annotations

import json
from pathlib import Path

from ...utils.cost import litellm_tokens_cost


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def load_trace_cost(log_path: Path | str, model_name: str) -> float:
    """Sum cost from a trace JSONL log.

    Uses the explicit ``cost`` field when present; otherwise falls back to
    computing cost from prompt/completion token counts.
    """
    path = Path(log_path)
    if not path.exists():
        return 0.0

    total = 0.0
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                explicit = _coerce_float(record.get("cost"))
                if explicit is not None:
                    total += explicit
                    continue

                prompt_tokens = record.get("prompt_tokens") or 0
                completion_tokens = record.get("completion_tokens") or 0
                computed = litellm_tokens_cost(
                    model_name=model_name,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                ).total_cost
                total += float(computed or 0.0)
    except FileNotFoundError:
        return 0.0

    return total
