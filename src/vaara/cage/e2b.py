# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""E2B's self-hosted infrastructure as a cage driver, over its HTTP API.

E2B infra (Apache-2.0) runs each sandbox as a Firecracker microVM from a
template, behind an API server (``POST /sandboxes``, ``GET`` and ``DELETE
/sandboxes/{id}``, ``GET /sandboxes/{id}/logs``, ``X-API-Key``) and a
guest daemon, envd, that starts processes inside over Connect RPC
(``process.Process/Start``) at ``49983-<sandboxID>-<clientID>.<domain>``
with the sandbox's ``envdAccessToken``.

This driver creates the sandbox with the cage declaration in ``envVars``
and the agent argv in ``metadata``, then asks envd to start the agent with
the same variables. The policy is a JSON file with the ``NewSandbox`` body
(``templateID`` required; ``network``, ``timeout``, ``allow_internet_access``
as the operator wants); the record's ``config_digest`` is ``sha256:`` over
it. Inside, the deciding process confirms the cage by the hypervisor the
CPU reports underneath it. The API's own view, ``state`` ``running`` or
``paused``, is the operator side.

Self-hosted by default: set ``E2B_API_URL`` to the API server and
``E2B_API_KEY`` to a team key. ``E2B_DOMAIN`` names the sandbox domain
when the API does not return one.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_API, BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "e2b"
ENVD_PORT = 49983


class E2BDriver:
    name = NAME

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None,
                 domain: Optional[str] = None, timeout: float = 60.0,
                 envd_url: Optional[str] = None) -> None:
        self._api = (api_url or os.environ.get("E2B_API_URL") or "").rstrip("/")
        self._key = api_key or os.environ.get("E2B_API_KEY") or ""
        self._domain = domain or os.environ.get("E2B_DOMAIN") or ""
        self._timeout = timeout
        # "{scheme}://{port}-{sandbox_id}-{client_id}.{domain}" unless overridden
        self._envd_url = envd_url or os.environ.get("E2B_ENVD_URL") or ""
        self._sandboxes: dict[str, dict[str, Any]] = {}

    # ── HTTP ─────────────────────────────────────────────────────

    def _request(self, method: str, url: str, body: Optional[Any] = None,
                 headers: Optional[dict[str, str]] = None, raw: Optional[bytes] = None) -> Any:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if not url.startswith(("http://", "https://")):
            raise CageError(f"E2B {method} {url}: not an http(s) URL")
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if raw is None and body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            raise CageError(f"E2B {method} {url}: {exc.code} {detail}") from None
        except urllib.error.URLError as exc:
            raise CageError(f"E2B {method} {url}: {exc.reason}") from None
        if not payload:
            return None
        if raw is not None:
            return payload
        try:
            return json.loads(payload)
        except ValueError:
            return payload.decode("utf-8", "replace")

    def _api_call(self, method: str, path: str, body: Optional[Any] = None) -> Any:
        if not self._api:
            raise CageError("E2B_API_URL is not set; point it at the self-hosted API server")
        if not self._key:
            raise CageError("E2B_API_KEY is not set")
        return self._request(method, f"{self._api}{path}", body, {"X-API-Key": self._key})

    def upstream_version(self) -> str:
        return "e2b-infra"

    # ── Driver interface ─────────────────────────────────────────

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, template: Optional[str] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        body: dict[str, Any] = {}
        digest = ""
        if policy is not None:
            try:
                body = json.loads(Path(policy).read_text())
            except OSError as exc:
                raise CageError(f"cannot read {policy}: {exc}") from None
            except ValueError:
                raise CageError(f"{policy}: the E2B sandbox request must be JSON "
                                "(the NewSandbox body)") from None
            if not isinstance(body, dict):
                raise CageError(f"{policy}: the request must be one object")
            digest = digest_file(Path(policy))
        if template:
            body["templateID"] = template
        if not body.get("templateID"):
            raise CageError("start needs a templateID (--policy request.json or --template)")
        launch_name = name or re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(agent[0]))
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest, confirmed=False, basis=BASIS_DECLARED,
                          name=launch_name)
        env = environ_for(state)
        body.setdefault("envVars", {}).update(env)
        body.setdefault("metadata", {}).update({
            "vaara.cage.name": launch_name,
            "vaara.agent": json.dumps(agent, separators=(",", ":")),
        })
        created = self._api_call("POST", "/sandboxes", body)
        if not isinstance(created, dict) or not created.get("sandboxID"):
            raise CageError("the E2B API created no sandbox")
        created["_name"] = launch_name
        self._sandboxes[launch_name] = created
        started = self._start_process(created, agent, env)
        return CageLaunch(driver=NAME, name=launch_name,
                          state=self.enforcement_state(launch_name),
                          detail={"sandbox_id": created["sandboxID"], "process": started,
                                  "declared": state.to_record()})

    def _sandbox_id(self, name: str) -> str:
        if name in self._sandboxes:
            return str(self._sandboxes[name]["sandboxID"])
        return name  # a sandbox id given directly

    def _envd_base(self, sandbox: dict[str, Any]) -> str:
        if self._envd_url:
            return self._envd_url.format(port=ENVD_PORT, sandbox_id=sandbox["sandboxID"],
                                         client_id=sandbox.get("clientID", ""),
                                         domain=sandbox.get("domain") or self._domain)
        domain = sandbox.get("domain") or self._domain
        if not domain:
            raise CageError("no sandbox domain: the API returned none and E2B_DOMAIN is unset")
        return f"https://{ENVD_PORT}-{sandbox['sandboxID']}-{sandbox.get('clientID', '')}.{domain}"

    def _start_process(self, sandbox: dict[str, Any], agent: list[str],
                       env: dict[str, str]) -> dict[str, Any]:
        """``process.Process/Start`` over Connect, enveloped JSON; the first
        data frame carries the start event with the pid."""
        url = self._envd_base(sandbox) + "/process.Process/Start"
        message = json.dumps({
            "process": {"cmd": agent[0], "args": list(agent[1:]), "envs": env},
            "tag": "vaara-agent", "stdin": False,
        }).encode()
        frame = struct.pack(">BI", 0, len(message)) + message
        headers = {"Content-Type": "application/connect+json",
                   "Connect-Protocol-Version": "1"}
        token = sandbox.get("envdAccessToken")
        if token:
            headers["X-Access-Token"] = str(token)
        payload = self._request("POST", url, headers=headers, raw=frame)
        return _first_frame(payload or b"")

    def stop(self, name: str) -> None:
        self._api_call("DELETE", f"/sandboxes/{self._sandbox_id(name)}")
        self._sandboxes.pop(name, None)

    def status(self, name: str) -> dict[str, Any]:
        detail = self._api_call("GET", f"/sandboxes/{self._sandbox_id(name)}")
        if not isinstance(detail, dict):
            raise CageError("the E2B API did not describe the sandbox")
        return detail

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("E2B reports per sandbox; give the launch name or sandbox id")
        detail = self.status(name)
        state = str(detail.get("state") or "")
        return CageState(
            driver=NAME, upstream=f"e2b-infra envd {detail.get('envdVersion') or ''}".strip(),
            config_digest="", confirmed=state == "running", basis=BASIS_API, name=name,
            detail={"sandbox_id": detail.get("sandboxID"), "state": state,
                    "template": detail.get("templateID"),
                    "allow_internet_access": detail.get("allowInternetAccess"),
                    "network": detail.get("network"), "started_at": detail.get("startedAt"),
                    "end_at": detail.get("endAt")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        path = f"/sandboxes/{self._sandbox_id(name)}/logs"
        if since:
            path += f"?start={int(since * 1000)}"
        data = self._api_call("GET", path)
        logs = (data.get("logs") or data.get("logEntries") or []) if isinstance(data, dict) else []

        def _gen() -> Iterator[dict[str, Any]]:
            for entry in logs:
                if not isinstance(entry, dict):
                    continue
                yield {"ts": entry.get("timestamp") or time.time(), "source": NAME,
                       "message": str(entry.get("line") or entry.get("message") or "")}

        return _gen()


def _first_frame(payload: bytes) -> dict[str, Any]:
    """The first data frame of a Connect streaming response, as JSON."""
    offset = 0
    while offset + 5 <= len(payload):
        flags, length = struct.unpack(">BI", payload[offset:offset + 5])
        body = payload[offset + 5:offset + 5 + length]
        offset += 5 + length
        try:
            data = json.loads(body) if body else {}
        except ValueError:
            data = {"raw": body.decode("utf-8", "replace")}
        if flags & 0x02:  # end of stream: an error, or a clean end with nothing
            if isinstance(data, dict) and data.get("error"):
                raise CageError(f"envd refused to start the agent: {data['error']}")
            continue
        if isinstance(data, dict):
            return data
    return {}
