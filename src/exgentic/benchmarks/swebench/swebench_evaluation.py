# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

from ...utils.logging import capture_stdio_to_session
from ...utils.paths import SessionPaths


@dataclass
class HarnessResult:
    harness_report: dict[str, Any]
    patch: str
    patch_valid: bool
    harness_ran: bool
    error: str | None = None


def is_patch_valid(patch: str) -> bool:
    if not patch or not patch.strip():
        return False
    return any(m in patch for m in ["---", "+++", "@@", "diff --git", "*** Begin Patch"])


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_harness(
    patch: str,
    instance_id: str,
    subset: str,
    paths: SessionPaths,
    eval_config: dict[str, Any],
    logger: Logger,
) -> HarnessResult:
    patch = patch or ""
    valid = is_patch_valid(patch)

    pred_path = paths.benchmark_dir / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_patch": patch,
                "patch_valid": valid,
                "model_name_or_path": "exgentic",
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    logger.info(f"EVAL | Writing patch | valid: {valid} | size: {len(patch)}")

    if not valid:
        logger.warning("EVAL | Skipping harness - invalid patch structure")
        return HarnessResult({}, patch, False, False)

    logger.info("EVAL | Running SWE-bench harness evaluation")
    try:
        from swebench.harness import run_evaluation

        with _pushd(paths.benchmark_dir), capture_stdio_to_session(logger):
            report_path = run_evaluation.main(
                dataset_name=subset,
                split="test",
                instance_ids=[instance_id],
                predictions_path=pred_path.name,
                max_workers=eval_config["max_workers"],
                run_id="exgentic1",
                namespace="swebench",
                force_rebuild=False,
                cache_level=eval_config["cache_level"],
                clean=False,
                open_file_limit=eval_config["open_file_limit"],
                timeout=eval_config["harness_timeout"],
                rewrite_reports=False,
                modal=False,
            )
            report = json.loads(Path(report_path).read_text()) if report_path and Path(report_path).is_file() else {}
        logger.info("EVAL | Harness evaluation completed")
        return HarnessResult(report, patch, True, True)
    except Exception as e:
        logger.exception(f"EVAL | Harness evaluation failed: {e}")
        return HarnessResult({}, patch, True, True, str(e))
