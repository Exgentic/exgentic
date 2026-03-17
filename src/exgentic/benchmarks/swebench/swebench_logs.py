# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from ...core.types import SessionScore as BaseSessionScore
from ...utils.paths import SessionPaths


class SessionScore(BaseSessionScore):
    instance_id: str = ""
    agent: dict[str, Any] = Field(default_factory=dict)
    patch: dict[str, Any] = Field(default_factory=dict)
    container: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


FILE_OPS_RE = re.compile(r"\b(cp|mv|rm|mkdir|touch|tee|patch)\b")
BUILD_PHASES = {"base": "build_base", "env": "build_env", "instances": "build_instance"}


def build_score(
    paths: SessionPaths,
    num_actions: int,
    max_interactions: int,
    instance_id: str,
    harness_data: Any | None = None,
) -> SessionScore:
    score = SessionScore(score=0.0, success=False, instance_id=instance_id)
    score.agent = _parse_agent(paths)
    score.patch = _parse_patch(paths)
    score.container, score.evaluation = _parse_harness_files(paths, score.patch)
    if harness_data:
        _apply_harness_data(score, harness_data, instance_id)
    score.summary = _build_summary(score, num_actions, max_interactions)
    score.score = score.summary["score"]
    return score


def _parse_agent(paths: SessionPaths) -> dict[str, Any]:
    log_path = paths.benchmark_dir / "session.log"
    if not log_path.exists():
        return {"commands": None, "edit_commands": None, "call_submit": False}
    content = log_path.read_text()
    commands = re.findall(r"\| command: (.+?)(?=\n(?:INFO|ERROR|\[LiteLM\])|$)", content, re.DOTALL)
    edit_cmds = [c for c in commands if _is_edit_cmd(c)]
    return {
        "commands": commands or None,
        "edit_commands": edit_cmds or None,
        "call_submit": bool(re.search(r"submit_patch\s*\| summary:", content)),
    }


def _parse_patch(paths: SessionPaths) -> dict[str, Any]:
    pred_path = paths.benchmark_dir / "predictions.jsonl"
    _none = {"generated": None, "length": None, "structurally_valid": None}
    if not pred_path.exists():
        return _none
    try:
        lines = pred_path.read_text().splitlines()
        if not lines:
            return _none
        pred = json.loads(lines[0])
        patch_len = len(pred.get("model_patch", ""))
        return {
            "generated": patch_len > 0,
            "length": patch_len,
            "structurally_valid": pred.get("patch_valid", False),
        }
    except (json.JSONDecodeError, IndexError):
        return _none


def _parse_harness_files(paths: SessionPaths, patch: dict) -> tuple:
    container = {
        "required": None,
        "build_base": None,
        "build_env": None,
        "build_instance": None,
        "started": None,
        "patch_exists": None,
        "applying_patch": None,
        "patch_applied": None,
        "removed": None,
    }
    evaluation = {"grading": None, "resolved": None, "test_results": {}}

    container["required"] = bool(patch.get("generated") and patch.get("structurally_valid"))
    if not container["required"]:
        return container, evaluation

    for path in paths.benchmark_dir.rglob("*"):
        if path.name == "build_image.log" and "build_images" in str(path):
            try:
                phase = path.parts[-3]
                if phase in BUILD_PHASES:
                    container[BUILD_PHASES[phase]] = path.read_text().strip().endswith("Image built successfully!")
            except IndexError:
                pass

        elif path.name == "run_instance.log" and "run_evaluation" in str(path):
            content = path.read_text()
            if re.search(r"Container .* started", content):
                container["started"] = True
            if re.search(r"Intermediate patch for .* written to logs", content):
                container["patch_exists"] = True
            if re.search(r"now applying to container", content):
                container["applying_patch"] = True
            if re.search(r"Patch Apply Failed", content):
                container["patch_applied"] = False
            if re.search(r"Grading answer for .*", content):
                evaluation["grading"] = True
            if re.search(r"Container .* removed\.", content):
                container["removed"] = True
            if container.get("applying_patch") and container.get("patch_applied") is None:
                container["patch_applied"] = True

        elif path.name == "report.json":
            try:
                report = json.loads(path.read_text())
                if len(report) == 1:
                    data = next(iter(report.values()))
                    container["patch_exists"] = data["patch_exists"]
                    container["patch_applied"] = data["patch_successfully_applied"]
                    evaluation["resolved"] = data["resolved"]
                    for case, cd in data.get("tests_status", {}).items():
                        s, f = len(cd.get("success", [])), len(cd.get("failure", []))
                        if s + f > 0:
                            rate = round(100 * s / (s + f), 2)
                            evaluation["test_results"][case] = {
                                "expected": s + f,
                                "success": s,
                                "failure": f,
                                "rate": rate,
                                "display": f"expected {s + f}: success {s}, failure {f}. success rate: {rate}%",
                            }
            except json.JSONDecodeError:
                pass

    return container, evaluation


def _apply_harness_data(score: SessionScore, hd: Any, instance_id: str):
    if hd.harness_report:
        submitted = set(hd.harness_report.get("submitted_ids", []))
        if submitted != {instance_id}:
            raise ValueError(f"Instance ID mismatch in harness report: expected {{'{instance_id}'}}, got {submitted}")
        if hd.harness_report.get("resolved_instances", 0) > 0:
            score.evaluation["resolved"] = True
    score.success = hd.error is None


def _build_summary(score: SessionScore, num_actions: int, max_interactions: int) -> dict[str, Any]:
    c, e, a, p = score.container, score.evaluation, score.agent, score.patch
    f2p = e.get("test_results", {}).get("FAIL_TO_PASS", {})
    p2p = e.get("test_results", {}).get("PASS_TO_PASS", {})
    final = int(e.get("resolved") or False)
    return {
        "num_actions": num_actions,
        "actions_limit": max_interactions,
        "num_edit_commands": len(a.get("edit_commands") or []),
        "agent_call_submit": 1 if a.get("call_submit") else 0,
        "patch_non_empty": 1 if p.get("generated") else 0,
        "container_status": -1 if not c.get("required") else (1 if c.get("started") else 0),
        "fail_to_pass_rate": f2p.get("rate") if isinstance(f2p, dict) else None,
        "pass_to_pass_rate": p2p.get("rate") if isinstance(p2p, dict) else None,
        "score": final,
    }


def _is_edit_cmd(cmd: str) -> bool:
    c = cmd.lower()
    return (
        ("python" in c and any(p in c for p in ["write_text", "write_bytes", ".write(", "replace(", "open("]))
        or "git apply" in c
        or "sed -i" in c
        or bool(FILE_OPS_RE.search(c))
        or ">>" in c
    )


def write_results(results_path: Path, score: SessionScore):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(score.model_dump_json(indent=2))
