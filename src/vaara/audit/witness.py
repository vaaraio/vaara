# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Outside witnesses for the trail head, switched on by configuration.

Two cadences live on :class:`vaara.audit.trail.AuditTrail`: publishing the
head digest to a public transparency log (``enable_auto_publish``) and
timestamping it at an RFC 3161 authority (``enable_auto_anchor``). Both were
library calls with no door. This module is the door: every surface that
opens a trail through ``SQLiteAuditBackend.load_trail`` calls
:func:`configure_from_env`, which reads

- ``VAARA_AUTO_PUBLISH_EVERY``: publish the head every N records, signed
  with the key at ``VAARA_PUBLISH_KEY`` (generated there when absent) to the
  log at ``VAARA_PUBLISH_LOG``;
- ``VAARA_AUTO_ANCHOR_EVERY``: anchor the head every N records at the
  authority ``VAARA_ANCHOR_TSA`` names, or the one chosen with
  ``vaara anchor-providers``.

Off unless set. Publishing is permanent and world-readable, so the facts a
person needs are printed to stderr once per process, the same facts
``vaara trail publish-head`` prints before it asks, and the cadence is
written to the trail as a ``witness_configured`` record so a verifier reads
the bound on the unwitnessed window instead of guessing it. A value that
cannot be used is said on stderr and leaves the cadence off; opening the
trail never fails because of it.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

PUBLISH_EVERY_ENV = "VAARA_AUTO_PUBLISH_EVERY"
PUBLISH_KEY_ENV = "VAARA_PUBLISH_KEY"
PUBLISH_LOG_ENV = "VAARA_PUBLISH_LOG"
ANCHOR_EVERY_ENV = "VAARA_AUTO_ANCHOR_EVERY"
ANCHOR_TSA_ENV = "VAARA_ANCHOR_TSA"

DEFAULT_PUBLISH_KEY = Path.home() / ".vaara" / "resin-publishing-key.pem"
DEFAULT_PUBLISH_LOG = "https://rekor.sigstore.dev"

#: Kinds disclosed on stderr in this process already.
_disclosed: set[str] = set()


def requested(environ: Optional[dict[str, str]] = None) -> bool:
    """Whether either cadence variable is set at all; cheap, so the trail
    door can skip everything else when it is not."""
    env = os.environ if environ is None else environ
    return bool(env.get(PUBLISH_EVERY_ENV) or env.get(ANCHOR_EVERY_ENV))


def _say(message: str, stderr) -> None:
    print(message, file=stderr or sys.stderr)


def _cadence(name: str, raw: str, stderr) -> Optional[int]:
    try:
        every = int(raw.strip())
    except ValueError:
        every = 0
    if every < 1:
        _say(f"vaara: {name}={raw!r} is not a positive integer; the cadence stays off", stderr)
        return None
    return every


def _configured_tsa() -> Optional[str]:
    """The authority chosen with ``vaara anchor-providers``, if any."""
    try:
        from vaara.menu import _load_config

        url = _load_config().get("anchor_tsa_url")
    except Exception:  # noqa: BLE001 - a missing or unreadable config means no choice
        return None
    return url if isinstance(url, str) and url else None


def _anchor_client(url: str):
    from vaara.audit.timeanchor import RFC3161TimeAnchorClient

    return RFC3161TimeAnchorClient(url)


def _load_or_make_signer(path: Path, stderr):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    # One long-lived key whose public half travels with every entry, so a
    # verifier can tie a signer's published heads together. Same as the
    # publish-head command generates.
    signer = ec.generate_private_key(ec.SECP256R1())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(signer.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _say(f"vaara: generated a publishing key at {path}", stderr)
    return signer


def _fingerprint(signer) -> str:
    from cryptography.hazmat.primitives import serialization

    der = signer.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()[:32]


def _disclose_publish(every: int, log_url: str, key_path: Path, stderr) -> None:
    if "publish" in _disclosed:
        return
    _disclosed.add("publish")
    _say(f"vaara: auto-publish is on ({PUBLISH_EVERY_ENV}={every}). Every {every} records "
         f"the trail head is published to a public transparency log.", stderr)
    _say("  what goes    one digest and a signature over it", stderr)
    _say("  what stays   every record, payload, tool name, identity, and the", stderr)
    _say("               number of entries in this trail", stderr)
    _say(f"  where        {log_url}", stderr)
    _say("  who reads it everyone, forever. The whole log is public and", stderr)
    _say("               enumerable, not just visible to whoever runs it.", stderr)
    _say("  erasure      permanent and irreversible. A published head cannot be withdrawn,", stderr)
    _say("               deleted or corrected once integrated.", stderr)
    _say(f"  linkability  every head published with {key_path} is groupable by", stderr)
    _say(f"               anyone; a separate {PUBLISH_KEY_ENV} per context keeps", stderr)
    _say("               trails from sharing one pseudonym.", stderr)
    _say(f"  off          unset {PUBLISH_EVERY_ENV}", stderr)


def _disclose_anchor(every: int, tsa_url: str, stderr) -> None:
    if "anchor" in _disclosed:
        return
    _disclosed.add("anchor")
    _say(f"vaara: auto-anchor is on ({ANCHOR_EVERY_ENV}={every}). Every {every} records "
         f"the trail head digest is timestamped at {tsa_url}; the authority sees the "
         f"digest and nothing else. An unreachable authority leaves an anchor_gap "
         f"record in the chain. Off: unset {ANCHOR_EVERY_ENV}.", stderr)


def configure_from_env(trail: Any, *, environ: Optional[dict[str, str]] = None,
                       stderr=None) -> list[str]:
    """Switch on the cadences the environment asks for; return their kinds."""
    env = os.environ if environ is None else environ
    enabled: list[str] = []

    raw = env.get(ANCHOR_EVERY_ENV)
    if raw:
        every = _cadence(ANCHOR_EVERY_ENV, raw, stderr)
        tsa_url = (env.get(ANCHOR_TSA_ENV) or "").strip() or _configured_tsa()
        if every is not None and not tsa_url:
            _say(f"vaara: {ANCHOR_EVERY_ENV} is set but no authority is named; set "
                 f"{ANCHOR_TSA_ENV} or choose one with `vaara anchor-providers`. "
                 f"Anchoring stays off.", stderr)
        elif every is not None and tsa_url:
            try:
                client = _anchor_client(tsa_url)
            except Exception as exc:  # noqa: BLE001 - said, and the trail still opens
                _say(f"vaara: auto-anchor could not start ({exc}); anchoring stays off", stderr)
            else:
                trail.enable_auto_anchor(client, every_records=every)
                _disclose_anchor(every, tsa_url, stderr)
                trail.record_witness_configured(
                    "anchor", every, tsa_url, source=ANCHOR_EVERY_ENV,
                )
                enabled.append("anchor")

    raw = env.get(PUBLISH_EVERY_ENV)
    if raw:
        every = _cadence(PUBLISH_EVERY_ENV, raw, stderr)
        if every is not None:
            key_path = Path(env.get(PUBLISH_KEY_ENV) or DEFAULT_PUBLISH_KEY).expanduser()
            log_url = (env.get(PUBLISH_LOG_ENV) or "").strip() or DEFAULT_PUBLISH_LOG
            try:
                signer = _load_or_make_signer(key_path, stderr)
            except ImportError:
                _say("vaara: auto-publish needs the attestation extra (pip install "
                     "'vaara[attestation]'); publishing stays off", stderr)
                signer = None
            except (OSError, ValueError) as exc:
                _say(f"vaara: auto-publish could not load or write the key at {key_path} "
                     f"({exc}); publishing stays off", stderr)
                signer = None
            if signer is not None:
                trail.enable_auto_publish(signer, every_records=every, log_url=log_url)
                _disclose_publish(every, log_url, key_path, stderr)
                trail.record_witness_configured(
                    "publish", every, log_url,
                    key_fingerprint=_fingerprint(signer), source=PUBLISH_EVERY_ENV,
                )
                enabled.append("publish")
    return enabled
