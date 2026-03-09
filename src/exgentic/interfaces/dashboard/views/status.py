# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any

from nicegui import ui

_STATUS_ORDER = [
    "success",
    "unsuccessful",
    "unfinished",
    "agent error",
    "benchmark error",
    "cancelled",
    "error",
    "running",
]

_STATUS_COLORS = {
    "success": "#22c55e",
    "unsuccessful": "#f59e0b",
    "unfinished": "#facc15",
    "agent error": "#ef4444",
    "benchmark error": "#f97316",
    "cancelled": "#94a3b8",
    "error": "#dc2626",
    "running": "#38bdf8",
}


def _status_from_outcome(success: Any, is_finished: Any, error_source: Any = None) -> str:
    if success is True:
        return "success"
    if is_finished is True:
        return "unsuccessful"
    if is_finished is False:
        return "unfinished"
    if error_source == "agent":
        return "agent error"
    if error_source == "benchmark":
        return "benchmark error"
    if error_source == "cancelled":
        return "cancelled"
    return "error"


def _status_counts_from_sessions(sessions: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for data in sessions.values():
        status = data.get("status") or "error"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _render_status_pie(status_counts: dict[str, int]) -> None:
    if not status_counts:
        ui.label("No session data available.")
        return
    data = []
    colors = []
    for status in _STATUS_ORDER:
        count = status_counts.get(status, 0)
        if count:
            data.append({"value": count, "name": status})
            colors.append(_STATUS_COLORS.get(status, "#94a3b8"))
    if not data:
        ui.label("No session data available.")
        return
    ui.echart(
        {
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "color": colors,
            "series": [
                {
                    "name": "Sessions",
                    "type": "pie",
                    "radius": ["35%", "70%"],
                    "center": ["60%", "55%"],
                    "label": {"formatter": "{b}: {c} ({d}%)"},
                    "data": data,
                }
            ],
        }
    ).classes("w-full").style("height: 260px;")
