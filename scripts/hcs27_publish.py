#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Create an HCS-27 checkpoint topic and submit a Vaara checkpoint to it.

This is the network half of `vaara.audit.hcs27`. The module itself is stdlib
only and never touches a network; everything that speaks to Hedera lives here,
behind the `hedera` extra, so a checkpoint can still be built and verified on a
machine that has no SDK and no account.

    pip install 'vaara[hedera]'

Credentials are never written, logged or echoed. They come from the
environment, or from an env file that is not in the repository:

    HEDERA_ACCOUNT_ID    operator account, e.g. 0.0.12345      (public)
    HEDERA_PRIVATE_KEY   operator key, DER or hex               (SECRET)
    HEDERA_NETWORK       testnet (default), previewnet, mainnet

The default env file is `.shared/hedera/testnet.env`, under the one gitignored
root, so a key put there cannot reach a commit. A real environment variable
always wins over the file, and the file is refused if its permissions let
anyone but the owner read it.

Two steps, deliberately separate, because they have very different costs. A
topic is created once and lives forever. Messages are submitted against it many
times.

    scripts/hcs27_publish.py create-topic
    scripts/hcs27_publish.py submit --topic-id 0.0.NNNNN

What gets submitted
-------------------
By default, the checkpoint from the committed conformance vector
`tests/vectors/hcs27_checkpoint_v0/`. That is the useful thing to publish
first: the root on the ledger can then be checked against entries any stranger
can download from the repository, so the claim "this root commits to these
seven records" is checkable by someone who trusts neither of us.

`--from-checkpoint` takes any checkpoint message JSON instead, which is what a
real trail run will pass.

On the admin key
----------------
The topic is created with the operator key as admin key, so it can be updated
or deleted later. Pass `--immutable` to create a topic with no admin key at
all, which nobody can ever change or remove. That is the right choice for a
production checkpoint stream and the wrong one for a first experiment, so it is
opt-in.

HIP-991 custom fees are not set here. The SDK supports them
(`set_custom_fees`, `set_fee_schedule_key`, `set_fee_exempt_keys`) and the fee
schedule key cannot be added after creation, so a topic intended to carry fees
later has to be created with one from the start. That is a commercial decision
rather than a technical default, so this script does not quietly make it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
VECTOR = REPO / "tests" / "vectors" / "hcs27_checkpoint_v0"
DEFAULT_ENV_FILE = REPO / ".shared" / "hedera" / "testnet.env"

NETWORKS = ("testnet", "previewnet", "mainnet")

CREDENTIAL_KEYS = (
    "HEDERA_ACCOUNT_ID",
    "HEDERA_PRIVATE_KEY",
    "HEDERA_NETWORK",
    "HEDERA_KEY_TYPE",
)

#: A raw 32-byte key is ambiguous: the SDK's own warning says there is no way
#: to tell an Ed25519 seed from an ECDSA scalar, so it tries Ed25519 first and
#: silently succeeds either way. Signing with the wrong curve produces a key
#: that parses, an account that does not match, and an error at submit time
#: that names neither. Say which curve it is.
KEY_TYPES = ("ed25519", "ecdsa", "auto")


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_env_file(path: Path) -> dict[str, str]:
    """Read KEY=value lines, without ever putting a value in a message.

    A real environment variable wins over the file, so an operator can override
    a stored key for one run without editing anything.
    """
    if not path.exists():
        return {}

    mode = path.stat().st_mode & 0o077
    if mode:
        _fail(
            f"{path} is readable or writable by others (mode {oct(path.stat().st_mode & 0o777)}). "
            f"Run: chmod 600 {path}"
        )

    values: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            _fail(f"{path}:{lineno} is not a KEY=value line")
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key not in CREDENTIAL_KEYS:
            _fail(
                f"{path}:{lineno} sets {key!r}, which is not one of "
                f"{', '.join(CREDENTIAL_KEYS)}"
            )
        values[key] = value.strip().strip("'\"")
    return values


def _client(env_file: Path | None = None) -> Any:
    """Build an operator-bound client, or explain exactly what is missing."""
    try:
        from hiero_sdk_python import AccountId, Client, Network, PrivateKey
    except ImportError:
        _fail(
            "the hiero-sdk-python package is not installed. "
            "Install the extra with: pip install 'vaara[hedera]'"
        )

    path = env_file if env_file is not None else DEFAULT_ENV_FILE
    from_file = _load_env_file(path)

    def _get(key: str, default: str = "") -> str:
        return (os.environ.get(key) or from_file.get(key) or default).strip()

    account_id = _get("HEDERA_ACCOUNT_ID")
    private_key = _get("HEDERA_PRIVATE_KEY")
    network = _get("HEDERA_NETWORK", "testnet") or "testnet"

    if not account_id:
        _fail(
            "HEDERA_ACCOUNT_ID is not set. Export it, or put it in "
            f"{path} as HEDERA_ACCOUNT_ID=0.0.12345"
        )
    if not private_key:
        _fail(f"HEDERA_PRIVATE_KEY is not set. Export it, or put it in {path}")
    if network not in NETWORKS:
        _fail(f"HEDERA_NETWORK must be one of {', '.join(NETWORKS)}, got {network!r}")

    key_type = (_get("HEDERA_KEY_TYPE", "ed25519") or "ed25519").lower()
    if key_type not in KEY_TYPES:
        _fail(f"HEDERA_KEY_TYPE must be one of {', '.join(KEY_TYPES)}, got {key_type!r}")

    parser = {
        "ed25519": PrivateKey.from_string_ed25519,
        "ecdsa": PrivateKey.from_string_ecdsa,
        "auto": PrivateKey.from_string,
    }[key_type]

    try:
        key = parser(private_key)
    except Exception:
        # Deliberately does not include the exception text, which some parsers
        # render with the offending key material in it.
        _fail(
            f"HEDERA_PRIVATE_KEY could not be parsed as a {key_type} private key. "
            f"If the account was created with the other curve, set HEDERA_KEY_TYPE."
        )

    client = Client(Network(network=network))
    client.set_operator(AccountId.from_string(account_id), key)
    return client, key, network, account_id


def cmd_create_topic(args: argparse.Namespace) -> int:
    from hiero_sdk_python import TopicCreateTransaction

    client, key, network, account_id = _client(args.env_file and Path(args.env_file))

    tx = TopicCreateTransaction().set_memo(args.memo)
    if not args.immutable:
        tx = tx.set_admin_key(key.public_key())
    if args.submit_key:
        tx = tx.set_submit_key(key.public_key())

    receipt = tx.execute(client)
    topic_id = str(receipt.topic_id)

    print(f"network   {network}")
    print(f"operator  {account_id}")
    print(f"topic     {topic_id}")
    print(f"memo      {args.memo}")
    print(f"admin key {'none, this topic is immutable' if args.immutable else 'operator'}")
    print(f"submit    {'operator key only' if args.submit_key else 'open to anyone'}")
    print()
    print("Next:")
    print(f"  scripts/hcs27_publish.py submit --topic-id {topic_id}")
    return 0


def _load_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    if args.from_checkpoint:
        return json.loads(Path(args.from_checkpoint).read_text(encoding="utf-8"))
    checkpoints = json.loads((VECTOR / "checkpoints.json").read_text(encoding="utf-8"))
    return checkpoints["current"]


def cmd_submit(args: argparse.Namespace) -> int:
    checkpoint = _load_checkpoint(args)

    # Exact submitted bytes. json.dumps with these settings is what
    # vaara.audit.hcs27.message_bytes emits, and the digest on the wire has to
    # be the digest anyone recomputing from the vector gets.
    payload = json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(payload) > 1024:
        _fail(f"message is {len(payload)} bytes, over Hedera's 1024-byte cap")

    root = checkpoint["metadata"]["root"]

    # Before the client, so a dry run needs no SDK, no key and no network. It
    # is the step you use to see the exact bytes before anything is spent.
    if args.dry_run:
        print("DRY RUN, nothing submitted")
        print(f"topic      {args.topic_id}")
        print(f"bytes      {len(payload)}")
        print(f"treeSize   {root['treeSize']}")
        print(f"root       {root['rootHashB64u']}")
        print()
        print(payload.decode("utf-8"))
        return 0

    from hiero_sdk_python import TopicId, TopicMessageSubmitTransaction

    client, _key, network, account_id = _client(args.env_file and Path(args.env_file))

    receipt = (
        TopicMessageSubmitTransaction(
            topic_id=TopicId.from_string(args.topic_id), message=payload
        )
    ).execute(client)

    sequence = getattr(receipt, "topic_sequence_number", None)

    print(f"network    {network}")
    print(f"operator   {account_id}")
    print(f"topic      {args.topic_id}")
    print(f"bytes      {len(payload)}")
    print(f"treeSize   {root['treeSize']}")
    print(f"root       {root['rootHashB64u']}")
    print(f"sequence   {sequence}")
    print()
    print("Read it back with a checker that imports neither Vaara nor the SDK:")
    print(
        f"  scripts/hcs27_mirror_check.py --topic-id {args.topic_id} "
        f"--network {network}"
    )

    if args.receipt:
        out = Path(args.receipt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "network": network,
                    "topicId": args.topic_id,
                    "sequenceNumber": sequence,
                    "messageBytes": len(payload),
                    "treeSize": root["treeSize"],
                    "rootHashB64u": root["rootHashB64u"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nreceipt written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-topic", help="create a checkpoint topic, once")
    create.add_argument(
        "--memo",
        default="vaara trail checkpoints, HCS-27",
        help="topic memo, 100 bytes max",
    )
    create.add_argument(
        "--immutable",
        action="store_true",
        help="create with no admin key, so the topic can never be changed or deleted",
    )
    create.add_argument(
        "--submit-key",
        action="store_true",
        help="restrict submission to the operator key (default: anyone may submit)",
    )
    create.set_defaults(func=cmd_create_topic)

    submit = sub.add_parser("submit", help="submit a checkpoint message")
    submit.add_argument("--topic-id", required=True, help="e.g. 0.0.12345")
    submit.add_argument(
        "--from-checkpoint",
        help="path to a checkpoint message JSON (default: the committed vector)",
    )
    submit.add_argument(
        "--receipt", help="write a JSON receipt of the submission to this path"
    )
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact bytes and stop, touching no network and no credentials",
    )
    submit.set_defaults(func=cmd_submit)

    for p_ in (create, submit):
        p_.add_argument(
            "--env-file",
            help=f"credentials file (default: {DEFAULT_ENV_FILE})",
        )

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
