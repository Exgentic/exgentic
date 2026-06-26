# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Pod-native SWE-bench sandbox (no docker / no privileged).

The default SWE-bench path runs each task's environment as a Docker container
(minisweagent's ``DockerEnvironment`` + the docker-based ``run_evaluation``
harness), which needs a container runtime in the pod (privileged DinD or
rootless podman).

This module is the cloud-native alternative: each task's environment becomes
its **own sibling Kubernetes Pod**, created from the SWE-bench instance image
(``swebench/sweb.eval...``). The agent's bash actions run via ``kubectl exec``,
and grading runs the *same* SWE-bench eval script (from ``make_test_spec``)
inside that pod, parsed by SWE-bench's own ``get_eval_report``. So there is no
docker, no DinD, and no privileged container - the instance image runs as root
under the ``anyuid`` SCC, nothing more.

``KubernetesEnvironment.execute`` is shape-compatible with minisweagent's
environments (``{"output": str, "returncode": int}``), so the Session's
``run_bash`` / ``generate_patch`` use it unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

_DEFAULT_CWD = "/testbed"


def _kubectl_bin() -> str:
    exe = shutil.which("kubectl") or shutil.which("oc")
    if exe is None:
        raise RuntimeError("kubectl (or oc) not found on PATH")
    return exe


class KubernetesEnvironment:
    """Run one SWE-bench task in a dedicated Pod, exec'ing via kubectl.

    Mirrors minisweagent's ``DockerEnvironment`` (``docker run`` -> Pod,
    ``docker exec`` -> ``kubectl exec``) so the agent loop is unchanged.
    """

    def __init__(
        self,
        *,
        image: str,
        namespace: str,
        service_account: str | None = None,
        image_pull_secrets: list[str] | None = None,
        cwd: str = _DEFAULT_CWD,
        timeout: int = 1800,
        pull_timeout: int = 900,
        env: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("exgentic.swebench.kube")
        self.image = image
        self.namespace = namespace
        self.service_account = service_account
        self.image_pull_secrets = image_pull_secrets or []
        self.cwd = cwd
        self.timeout = timeout
        # Pod lifetime is decoupled from the per-command timeout: a session can run
        # for far longer than any single bash command (esp. middleware modes whose
        # per-step latency is higher), and the pod must outlive the WHOLE session so
        # grade_in_pod can still kubectl-exec into it at the end. If the pod's sleep
        # expired mid-session the task silently went ungraded. Generous, env-tunable.
        self.pod_ttl = int(os.environ.get("SWEBENCH_POD_TTL", "14400"))
        self.pull_timeout = pull_timeout
        self.env = env or {}
        self.pod_name = f"swebench-{uuid4().hex[:8]}"
        self._deleted = False
        self._start_pod()

    # ── pod lifecycle ────────────────────────────────────────────────
    def _manifest(self) -> dict[str, Any]:
        container: dict[str, Any] = {
            "name": "task",
            "image": self.image,
            # The instance image's filesystem (repo at /testbed, conda env) is
            # what we exec into; keep it alive for the whole session (pod_ttl),
            # NOT just one command timeout, so end-of-session grading can exec in.
            "command": ["sleep", str(self.pod_ttl)],
            "workingDir": self.cwd,
            "imagePullPolicy": "IfNotPresent",
        }
        # Run as root: SWE-bench instance images own /testbed (repo + .git) as root.
        # Without this, OpenShift's SCC assigns a random uid that cannot write the
        # root-owned source files AND trips git's "dubious ownership" guard, so
        # `git add -A && git diff` returns empty -> correct fixes are captured as an
        # empty patch and scored 0. (exgentic-task SA carries anyuid, so root schedules.)
        container["securityContext"] = {"runAsUser": 0}
        spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [container],
            "securityContext": {"runAsUser": 0},
        }
        if self.service_account:
            spec["serviceAccountName"] = self.service_account
        if self.image_pull_secrets:
            spec["imagePullSecrets"] = [{"name": n} for n in self.image_pull_secrets]
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod_name,
                "namespace": self.namespace,
                "labels": {"app": self.pod_name, "exgentic.swebench": "task"},
            },
            "spec": spec,
        }

    def _start_pod(self) -> None:
        self.logger.info(f"KUBE | creating task pod {self.pod_name} (image {self.image})")
        self._run(["apply", "-f", "-"], stdin=json.dumps(self._manifest()))
        # Image pull of a SWE-bench instance image can be slow on a cold node.
        r = self._run(
            ["wait", f"pod/{self.pod_name}", "--for=condition=Ready",
             f"--timeout={self.pull_timeout}s"],
            check=False,
        )
        if r.returncode != 0:
            desc = self._run(["describe", "pod", self.pod_name], check=False).stdout
            self.cleanup()
            raise RuntimeError(
                f"Task pod {self.pod_name} not Ready within {self.pull_timeout}s.\n{desc}"
            )
        self.logger.info(f"KUBE | task pod {self.pod_name} ready")

    def _run(self, args: list[str], *, stdin: str | None = None,
             check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [_kubectl_bin(), "-n", self.namespace, *args],
            input=stdin, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=check,
        )

    # ── agent contract (minisweagent-compatible) ─────────────────────
    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        cwd = cwd or self.cwd
        exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in self.env.items())
        script = f"{exports}cd {shlex.quote(cwd)} && {command}"
        try:
            r = self._run(
                ["exec", self.pod_name, "-c", "task", "--", "bash", "-lc", script],
                check=False, timeout=timeout or self.timeout,
            )
            return {"output": r.stdout, "returncode": r.returncode}
        except subprocess.TimeoutExpired as e:
            return {"output": (e.stdout or "") + "\n<timeout>", "returncode": 124}

    def write_file(self, path: str, content: str) -> None:
        """Stream *content* into *path* inside the pod (kubectl exec stdin)."""
        self._run(
            ["exec", "-i", self.pod_name, "-c", "task", "--",
             "sh", "-c", f"cat > {shlex.quote(path)}"],
            stdin=content,
        )

    # ── teardown ─────────────────────────────────────────────────────
    def cleanup(self) -> None:
        if self._deleted:
            return
        self._deleted = True
        try:
            self._run(["delete", "pod", self.pod_name, "--ignore-not-found", "--wait=false"],
                      check=False)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    def __del__(self) -> None:
        self.cleanup()


def grade_in_pod(env: KubernetesEnvironment, instance: dict, model_patch: str,
                 test_output_path: Path, logger: logging.Logger):
    """Grade a SWE-bench task inside its own pod, reusing SWE-bench's own eval
    script + report parser (no docker harness).

    Returns a ``swebench_evaluation.HarnessResult``.
    """
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_PREDICTION
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    from .swebench_evaluation import HarnessResult, is_patch_valid

    import os

    instance_id = instance["instance_id"]
    valid = is_patch_valid(model_patch)
    ts = make_test_spec(instance, namespace=os.environ.get("SWEBENCH_IMAGE_NAMESPACE", "swebench"))
    base = instance["base_commit"]

    # Clean checkout at base, then apply exactly the submitted patch (matches
    # swebench's run_instance semantics: container starts clean, model patch
    # applied, then the eval script applies the test patch and runs the tests).
    env.execute(f"git reset --hard {shlex.quote(base)} && git clean -fdxq")
    if valid:
        env.write_file("/tmp/model.patch", model_patch)
        ap = env.execute(
            "git apply -v /tmp/model.patch || "
            "patch --batch --fuzz=5 -p1 -i /tmp/model.patch"
        )
        if ap["returncode"] != 0:
            logger.warning(f"KUBE | model patch did not apply cleanly:\n{ap['output'][-2000:]}")

    # Old SWE-bench repos (astropy era) build their editable install via
    # setuptools.dep_util, removed in setuptools>=70. If the grading env's setuptools
    # is too new, the rebuild + conftest import fail silently and pytest can't collect
    # -> every FAIL_TO_PASS is marked failed even when the fix is correct. Pin <70 in
    # the runtime env (legacy setup.py uses the *runtime* setuptools, so PIP_CONSTRAINT
    # alone is not enough) only when dep_util is actually missing, and constrain later
    # pip so the eval script can't re-upgrade past it. No-op for envs already <70.
    setuptools_guard = (
        "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true\n"
        "printf 'setuptools<70\\n' > /tmp/exg_setuptools_constraint.txt\n"
        "export PIP_CONSTRAINT=/tmp/exg_setuptools_constraint.txt\n"
        "python -c 'import setuptools.dep_util' 2>/dev/null || "
        "pip install 'setuptools<70' -q 2>/dev/null || true\n"
    )
    env.write_file("/eval.sh", setuptools_guard + ts.eval_script)
    result = env.execute("chmod +x /eval.sh && /eval.sh 2>&1")
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.write_text(result["output"])

    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_PREDICTION: model_patch or "",
        "model_name_or_path": "exgentic",
    }
    per_instance = get_eval_report(ts, prediction, str(test_output_path), include_tests_status=True)
    inst = per_instance.get(instance_id, {})
    resolved = bool(inst.get("resolved", False))
    logger.info(f"KUBE | grade {instance_id}: resolved={resolved}")

    # The score is assembled from two shapes (matching the docker harness):
    #  (1) a per-instance report.json in benchmark_dir, scanned for resolved +
    #      per-test (FAIL_TO_PASS / PASS_TO_PASS) results;
    #  (2) the run-summary shape consumed by _apply_harness_data.
    try:
        (test_output_path.parent / "report.json").write_text(json.dumps(per_instance))
    except OSError as e:  # pragma: no cover - best effort
        logger.warning(f"KUBE | could not write report.json: {e}")
    summary = {
        "submitted_ids": [instance_id],
        "completed_ids": [instance_id],
        "resolved_instances": 1 if resolved else 0,
    }
    return HarnessResult(
        harness_report=summary,
        patch=model_patch or "",
        patch_valid=valid,
        harness_ran=True,
    )
