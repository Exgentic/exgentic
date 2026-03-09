# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import math
from typing import Any


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        abs_val = abs(value)
        if abs_val >= 100 or value.is_integer():
            return f"{value:.2f}".rstrip("0").rstrip(".")
        if abs_val >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_payload(value: Any, limit: int = 4000) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        payload = str(value)
    if len(payload) > limit:
        return payload[:limit] + "..."
    return payload


def _format_error_message(value: Any, limit: int = 4000) -> str:
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + "..."
        return value
    return _format_payload(value, limit=limit)


def _normalize_action_payload(payload: Any) -> dict:
    if payload is None:
        return {"name": None, "arguments": None}
    if isinstance(payload, dict):
        action_type = payload.get("type")
        if action_type in {"parallel", "sequential"}:
            actions = payload.get("actions") or []
            cleaned_actions = []
            if isinstance(actions, list):
                for item in actions:
                    if not isinstance(item, dict):
                        cleaned_actions.append({"name": str(item), "arguments": None})
                        continue
                    cleaned_actions.append(
                        {
                            "name": item.get("name"),
                            "arguments": item.get("arguments"),
                        }
                    )
            return {"mode": action_type, "actions": cleaned_actions}
        return {
            "name": payload.get("name"),
            "arguments": payload.get("arguments"),
        }
    return {"raw": payload}


def _normalize_observation_item(item: Any) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        if "result" in item:
            result = item.get("result")
            if isinstance(result, dict):
                if "sender" in result and "message" in result:
                    return {
                        "sender": result.get("sender"),
                        "message": result.get("message"),
                    }
            return result
        if "sender" in item and "message" in item:
            return {"sender": item.get("sender"), "message": item.get("message")}
        return item
    return item


def _normalize_observation_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if isinstance(payload.get("observations"), list):
            return [_normalize_observation_item(item) for item in payload["observations"]]
        return _normalize_observation_item(payload)
    if isinstance(payload, list):
        return [_normalize_observation_item(item) for item in payload]
    return payload
