# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""KubernetesRunner — runs the HTTP service inside a Kubernetes Pod.

The cloud-native sibling of :class:`DockerRunner`: instead of ``docker run``
on the laptop it creates a ConfigMap + Pod + ClusterIP Service running
``exgentic serve``, waits for ``/health``, and returns the same
``ObjectProxy`` over the same ``HTTPTransport``.  vLLM (or any model
endpoint) is reached as an ordinary in-cluster Service — the eval lifecycle
no longer lives on the laptop.

Manifests are applied with ``kubectl apply`` as JSON (no PyYAML / k8s-client
dependency).  Reach the service either over in-cluster DNS
(``<svc>.<ns>.svc.cluster.local``) or, when driving from a laptop, over a
``kubectl port-forward`` (``port_forward=True``).

The runner is deliberately a thin "run ``exgentic serve`` in a Pod" mechanism
with no container-runtime concerns. SWE-bench, which needs a per-task
environment, has its own docker-free, non-privileged backend that creates one
Pod per task from the instance image - see
``benchmarks/swebench/kube_sandbox.py`` (``SWEBENCH_SANDBOX=kubernetes``).
"""

from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from ...core.context import Role

from ._utils import (
    find_free_port,
    inject_exgentic_env,
    make_close,
    prepare_subprocess_env,
    serialize_kwargs,
)
from .service import HTTPTransport, _wait_for_health
from .transport import ObjectProxy

_RUNTIME_MOUNT = "/etc/exgentic/runtime.json"


def _kubectl(*args: str, check: bool = True, stdin: str | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    kubectl = shutil.which("kubectl") or shutil.which("oc")
    if kubectl is None:
        raise RuntimeError("kubectl (or oc) not found on PATH")
    return subprocess.run([kubectl, *args], check=check, input=stdin, **kwargs)


class KubernetesRunner:
    """Start an ``exgentic serve`` Pod and return an ObjectProxy.

    Parameters
    ----------
    target_cls:      Class to instantiate inside the pod (or ``"module:Class"``).
    env_name/module_path: EnvironmentManager image lookup (mirrors DockerRunner).
    image:           Container image (registry-pullable by the cluster). Required
                     in practice — in-cluster builds are out of scope.
    namespace:       Target namespace (default: current kube-context namespace).
    service_account: ServiceAccount for the pod (e.g. one bound to anyuid/privileged).
    resources:       Pod resource requests/limits dict.
    env_secrets:     Secret names surfaced via ``envFrom: secretRef``.
    env:             Extra literal env (merged over forwarded provider creds).
    volumes:         ``{pvc_claim_name: mount_path}`` PVC mounts (e.g. shared outputs).
    security_context: Container ``securityContext`` dict (runAsUser, privileged…).
    node_selector:   Optional node pinning.
    labels:          Extra pod/service labels.
    use_job:         Wrap the pod in a Job (``ttlSecondsAfterFinished`` cleanup).
    health_timeout:  Seconds to wait for ``/health`` (k8s scheduling is slow).
    port_forward:    Reach the service via ``kubectl port-forward`` (laptop) vs
                     in-cluster DNS (when the orchestrator itself runs in-cluster).
    """

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        env_name: str = "",
        module_path: str = "",
        image: str | None = None,
        namespace: str | None = None,
        service_account: str | None = None,
        image_pull_secrets: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        env_secrets: list[str] | None = None,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        security_context: dict[str, Any] | None = None,
        node_selector: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        use_job: bool = False,
        health_timeout: float = 180.0,
        port_forward: bool = True,
        role: Role | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "KubernetesRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._env_name = env_name
        self._module_path = module_path
        self._image = image
        self._namespace = namespace or self._current_namespace()
        self._service_account = service_account
        self._image_pull_secrets = image_pull_secrets or []
        self._resources = resources
        self._env_secrets = env_secrets or []
        self._extra_env = env or {}
        self._volumes = volumes or {}
        self._security_context = security_context
        self._node_selector = node_selector
        self._extra_labels = labels or {}
        self._use_job = use_job
        self._health_timeout = health_timeout
        self._port_forward = port_forward
        self._role = role

        self._name = f"exgentic-{uuid4().hex[:8]}"
        self._local_port = find_free_port() if port_forward else 8080
        self._pf_proc: subprocess.Popen | None = None
        self._deleted = False

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _current_namespace() -> str:
        try:
            r = _kubectl(
                "config", "view", "--minify", "-o", "jsonpath={..namespace}",
                check=False, capture_output=True, text=True,
            )
            return r.stdout.strip() or "default"
        except Exception:
            return "default"

    def _ensure_image(self) -> str:
        if self._image:
            return self._image
        if not self._env_name:
            raise RuntimeError(
                "KubernetesRunner requires an 'image' (registry-pullable by the "
                "cluster) or an 'env_name' with a pre-built EnvironmentManager image."
            )
        from ...environment.instance import get_manager

        image = get_manager().docker_image(self._env_name)
        if not image:
            raise RuntimeError(
                f"No pre-built image for env '{self._env_name}'. Build and push it to "
                "a cluster-pullable registry, then pass image=<ref> (in-cluster builds "
                "are out of scope for KubernetesRunner)."
            )
        return image

    def _owner_labels(self) -> dict[str, str]:
        from ...utils.container_reaper import LABEL_OWNER_PID, LABEL_OWNER_TOKEN, OWN_TOKEN
        import os

        return {
            "app": self._name,
            LABEL_OWNER_PID.replace(".", "_"): str(os.getpid()),
            LABEL_OWNER_TOKEN.replace(".", "_"): OWN_TOKEN,
            **self._extra_labels,
        }

    # ── manifest builders ────────────────────────────────────────────

    def _pod_spec(self, image: str, cls_ref: str, kwargs_flag: str, kwargs_value: str,
                  env: dict[str, str], runtime_json: str | None) -> dict[str, Any]:
        serve_cmd = [
            "exgentic", "serve", "--cls", cls_ref, kwargs_flag, kwargs_value,
            "--host", "0.0.0.0", "--port", "8080",
        ]
        container_env = [{"name": k, "value": v} for k, v in {**env, **self._extra_env}.items()]
        volumes: list[dict[str, Any]] = []
        mounts: list[dict[str, Any]] = []
        if runtime_json is not None:
            volumes.append({"name": "runtime", "configMap": {"name": f"{self._name}-cfg"}})
            mounts.append({"name": "runtime", "mountPath": "/etc/exgentic", "readOnly": True})
        for claim, path in self._volumes.items():
            vname = f"pvc-{claim}"
            volumes.append({"name": vname, "persistentVolumeClaim": {"claimName": claim}})
            mounts.append({"name": vname, "mountPath": path})

        runner_container: dict[str, Any] = {
            "name": "runner",
            "image": image,
            "command": serve_cmd,
            "ports": [{"containerPort": 8080}],
            "env": container_env,
            "volumeMounts": mounts,
        }
        if self._env_secrets:
            runner_container["envFrom"] = [{"secretRef": {"name": s}} for s in self._env_secrets]
        if self._resources:
            runner_container["resources"] = self._resources
        if self._security_context:
            runner_container["securityContext"] = self._security_context

        spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "containers": [runner_container],
            "volumes": volumes,
        }
        if self._service_account:
            spec["serviceAccountName"] = self._service_account
        if self._image_pull_secrets:
            spec["imagePullSecrets"] = [{"name": n} for n in self._image_pull_secrets]
        if self._node_selector:
            spec["nodeSelector"] = self._node_selector
        return spec

    def _manifests(self, image: str, cls_ref: str, kwargs_flag: str, kwargs_value: str,
                   env: dict[str, str], runtime_json: str | None) -> dict[str, Any]:
        labels = self._owner_labels()
        items: list[dict[str, Any]] = []

        if runtime_json is not None:
            items.append({
                "apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": f"{self._name}-cfg", "namespace": self._namespace, "labels": labels},
                "data": {"runtime.json": runtime_json},
            })

        pod_spec = self._pod_spec(image, cls_ref, kwargs_flag, kwargs_value, env, runtime_json)
        pod_meta = {"labels": labels}
        if self._use_job:
            items.append({
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"name": self._name, "namespace": self._namespace, "labels": labels},
                "spec": {
                    "ttlSecondsAfterFinished": 3600, "backoffLimit": 0,
                    "template": {"metadata": pod_meta, "spec": pod_spec},
                },
            })
        else:
            items.append({
                "apiVersion": "v1", "kind": "Pod",
                "metadata": {"name": self._name, "namespace": self._namespace, **pod_meta},
                "spec": pod_spec,
            })

        items.append({
            "apiVersion": "v1", "kind": "Service",
            "metadata": {"name": f"{self._name}-svc", "namespace": self._namespace, "labels": labels},
            "spec": {
                "selector": {"app": self._name},
                "ports": [{"port": 8080, "targetPort": 8080}],
            },
        })
        return {"apiVersion": "v1", "kind": "List", "items": items}

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> ObjectProxy:
        image = self._ensure_image()
        cls_ref = (
            self._target_cls if isinstance(self._target_cls, str)
            else f"{self._target_cls.__module__}:{self._target_cls.__qualname__}"
        )
        kwargs_flag, kwargs_value = serialize_kwargs(self._kwargs)

        env = prepare_subprocess_env()
        inject_exgentic_env(env, role=self._role)
        # runtime.json travels via a ConfigMap (no host bind mount in k8s).
        runtime_json: str | None = None
        runtime_file = env.pop("EXGENTIC_RUNTIME_FILE", None)
        if runtime_file and Path(runtime_file).is_file():
            runtime_json = Path(runtime_file).read_text()
            env["EXGENTIC_RUNTIME_FILE"] = _RUNTIME_MOUNT

        manifests = self._manifests(image, cls_ref, kwargs_flag, kwargs_value, env, runtime_json)
        _kubectl("apply", "-f", "-", stdin=json.dumps(manifests), capture_output=True, text=True)
        atexit.register(self._delete)

        url = self._connect()
        try:
            _wait_for_health(url, timeout=self._health_timeout)
        except TimeoutError:
            logs = _kubectl("logs", f"pod/{self._name}", "-n", self._namespace, "-c", "runner",
                            check=False, capture_output=True, text=True)
            desc = _kubectl("describe", "pod", self._name, "-n", self._namespace,
                            check=False, capture_output=True, text=True)
            self._delete()
            raise TimeoutError(
                f"Pod {self._name} did not become healthy within {self._health_timeout}s.\n"
                f"--- describe ---\n{desc.stdout}\n--- logs ---\n{logs.stdout}\n{logs.stderr}"
            ) from None

        transport = HTTPTransport(url, timeout=600.0)
        proxy = ObjectProxy(transport)
        object.__setattr__(proxy, "close", make_close(transport, self._delete))
        return proxy

    def _connect(self) -> str:
        """Return the base URL, starting a port-forward when off-cluster."""
        if not self._port_forward:
            return f"http://{self._name}-svc.{self._namespace}.svc.cluster.local:8080"
        # Wait for the pod to be Ready before port-forwarding (the svc has no
        # endpoints until then, and port-forward to a Pod needs it scheduled).
        _kubectl("wait", f"pod/{self._name}", "-n", self._namespace,
                 "--for=condition=Ready", f"--timeout={int(self._health_timeout)}s",
                 check=False, capture_output=True, text=True)
        kubectl = shutil.which("kubectl") or shutil.which("oc")
        self._pf_proc = subprocess.Popen(
            [kubectl, "port-forward", "-n", self._namespace,
             f"svc/{self._name}-svc", f"{self._local_port}:8080"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        return f"http://127.0.0.1:{self._local_port}"

    def _delete(self) -> None:
        if self._deleted:
            return
        self._deleted = True
        if self._pf_proc is not None:
            self._pf_proc.terminate()
            self._pf_proc = None
        # Scope teardown to THIS instance (app=<name>): concurrent per-task
        # sessions share one owner token, so deleting by token would reap
        # siblings. The owner-token label stays on the resources for the
        # process-wide reaper's best-effort sweep on abnormal exit.
        _kubectl(
            "delete", "pod,job,service,configmap", "-n", self._namespace,
            "-l", f"app={self._name}", "--ignore-not-found", "--wait=false",
            check=False, capture_output=True, text=True,
        )
