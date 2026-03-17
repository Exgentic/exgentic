# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json

from ...utils.paths import RunPaths

FUNNEL = [
    ("edit", lambda s: (s.get("num_edit_commands") or 0) > 0),
    ("call_submit", lambda s: s.get("agent_call_submit") == 1),
    ("non_empty_patch", lambda s: s.get("patch_non_empty") == 1),
    ("container", lambda s: s.get("container_status") == 1),
    ("fail_to_pass", lambda s: s.get("fail_to_pass_rate") == 100.0),
    ("pass_to_pass", lambda s: s.get("pass_to_pass_rate") == 100.0),
    ("score", lambda s: s.get("score") == 1),
]


def collect_metrics(run_id, session_ids: list[str]) -> dict:
    from ...core.context import get_context

    run_paths = RunPaths(run_id=run_id, output_dir=get_context().output_dir)
    summaries = []
    for sid in session_ids:
        path = run_paths.session(sid).benchmark_results
        if path.exists():
            summaries.append(json.loads(path.read_text()).get("summary", {}))

    return {
        "funnel": {"total": len(summaries)} | {name: sum(1 for s in summaries if check(s)) for name, check in FUNNEL}
    }
