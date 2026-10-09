# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""kubernetes-sigs/agent-sandbox as a cage driver: a ``Sandbox`` object.

agent-sandbox (Apache-2.0, a Kubernetes SIG) is a fleet API: a ``Sandbox``
custom resource (``agents.x-k8s.io/v1beta1``) whose ``podTemplate`` names
a ``runtimeClassName``, gVisor or Kata, and whose controller keeps one
backing pod running. This driver applies the object with ``kubectl``, reads
its ``Ready`` condition, and deletes it to stop.

The policy is the Sandbox manifest itself, JSON or YAML (YAML needs the
``vaara[yaml]`` extra). The driver sets ``metadata.name``, puts the agent
argv on the first container as ``command`` and ``args``, and adds the cage
declaration to that container's ``env``. The record's ``config_digest`` is
``sha256:`` over the manifest as given, before those edits. Inside, the
deciding process confirms the cage by whichever the runtime class gives it:
gVisor's kernel log, or the hypervisor a Kata VM shows.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_CONTROL, BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import Tool, digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "agent-sandbox"
API_VERSION = "agents.x-k8s.io/v1beta1"
KIND = "Sandbox"
RESOURCE = "sandboxes.agents.x-k8s.io"


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise CageError(f"cannot read {path}: {exc}") from None
    try:
        data = json.loads(text)
    except ValueError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            raise CageError(f"{path} is not JSON, and reading YAML needs the vaara[yaml] "
                            "extra (pip install 'vaara[yaml]')") from None
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise CageError(f"{path}: {exc}") from None
    if not isinstance(data, dict):
        raise CageError(f"{path}: the manifest must be one object")
    if data.get("kind") != KIND:
        raise CageError(f"{path}: kind is {data.get('kind')!r}, not {KIND}")
    return data


class AgentSandboxDriver:
    name = NAME

    def __init__(self, kubectl: Optional[str] = None, namespace: Optional[str] = None,
                 timeout: float = 120.0) -> None:
        self._kubectl = Tool("kubectl", kubectl, "KUBECTL_BIN", "install kubectl", timeout)
        self._namespace = namespace or os.environ.get("VAARA_CAGE_NAMESPACE") or ""

    def _ns(self) -> list[str]:
        return ["-n", self._namespace] if self._namespace else []

    def upstream_version(self) -> str:
        try:
            crd = self._kubectl.json("get", "crd", RESOURCE, "-o", "json", timeout=30)
        except CageError:
            return "agent-sandbox"
        labels = ((crd.get("metadata") or {}).get("labels") or {}) if isinstance(crd, dict) else {}
        version = labels.get("app.kubernetes.io/version")
        if version:
            return f"agent-sandbox {version}"
        versions = [v.get("name") for v in ((crd.get("spec") or {}).get("versions") or [])
                    if isinstance(v, dict)] if isinstance(crd, dict) else []
        return f"agent-sandbox {','.join(str(v) for v in versions)}" if versions else "agent-sandbox"

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        if policy is None:
            raise CageError("start needs the Sandbox manifest (--policy sandbox.yaml)")
        manifest = load_manifest(Path(policy))
        digest = digest_file(Path(policy))
        launch_name = name or str((manifest.get("metadata") or {}).get("name") or
                                  re.sub(r"[^a-z0-9-]", "-", os.path.basename(agent[0]).lower()))
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest, confirmed=False, basis=BASIS_DECLARED,
                          name=launch_name)
        obj = copy.deepcopy(manifest)
        obj.setdefault("apiVersion", API_VERSION)
        obj.setdefault("metadata", {})["name"] = launch_name
        if self._namespace:
            obj["metadata"]["namespace"] = self._namespace
        try:
            containers = obj["spec"]["podTemplate"]["spec"]["containers"]
            first = containers[0]
        except (KeyError, IndexError, TypeError):
            raise CageError(f"{policy}: spec.podTemplate.spec.containers[0] is missing") from None
        first["command"] = [agent[0]]
        first["args"] = list(agent[1:])
        env = [e for e in (first.get("env") or [])
               if not str(e.get("name", "")).startswith("VAARA_CAGE")]
        env += [{"name": k, "value": v} for k, v in environ_for(state).items()]
        first["env"] = env
        self._kubectl.run("apply", *self._ns(), "-f", "-", stdin=json.dumps(obj))
        return CageLaunch(driver=NAME, name=launch_name,
                          state=self.enforcement_state(launch_name),
                          detail={"declared": state.to_record()})

    def stop(self, name: str) -> None:
        self._kubectl.run("delete", *self._ns(), RESOURCE, name, "--wait=false")

    def status(self, name: str) -> dict[str, Any]:
        obj = self._kubectl.json("get", *self._ns(), RESOURCE, name, "-o", "json")
        if not isinstance(obj, dict):
            raise CageError("kubectl get did not return an object")
        status = obj.get("status") or {}
        conditions = {c.get("type"): c for c in status.get("conditions") or []
                      if isinstance(c, dict)}
        ready = conditions.get("Ready") or {}
        pod_spec = (((obj.get("spec") or {}).get("podTemplate") or {}).get("spec") or {})
        return {"name": name, "ready": ready.get("status") == "True",
                "ready_reason": ready.get("reason", ""), "conditions": conditions,
                "runtime_class": pod_spec.get("runtimeClassName", ""),
                "operating_mode": (obj.get("spec") or {}).get("operatingMode", ""),
                "pod_ips": status.get("podIPs"), "service": status.get("service"),
                "resource_version": (obj.get("metadata") or {}).get("resourceVersion"),
                "generation": (obj.get("metadata") or {}).get("generation")}

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("agent-sandbox reports per Sandbox; give the launch name")
        status = self.status(name)
        return CageState(
            driver=NAME, upstream=self.upstream_version(), config_digest="",
            confirmed=bool(status["ready"]), basis=BASIS_CONTROL, name=name,
            detail={"ready_reason": status["ready_reason"],
                    "runtime_class": status["runtime_class"],
                    "operating_mode": status["operating_mode"],
                    "resource_version": status["resource_version"]},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        data = self._kubectl.json("get", *self._ns(), "events", "-o", "json",
                                  "--field-selector", f"involvedObject.name={name}")
        items = data.get("items") if isinstance(data, dict) else []

        def _gen() -> Iterator[dict[str, Any]]:
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                ts = item.get("lastTimestamp") or item.get("eventTime") or \
                    (item.get("metadata") or {}).get("creationTimestamp") or time.time()
                yield {"ts": ts, "source": "kubernetes", "reason": item.get("reason"),
                       "type": item.get("type"), "message": str(item.get("message", ""))}

        return _gen()
