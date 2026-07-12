"""Prove what your AI agent actually did, and verify it yourself.

One run, offline, no keys to trust. An AI agent makes a handful of tool calls
under Vaara governance: a safe one runs, dangerous ones are blocked. Vaara
writes every decision into a signed, hash-chained record. Then you verify that
record yourself, and watch a single forged byte get caught.

The point is the last part. Anyone can write a log. The question that matters
is whether a party who does not trust the producer can check it. Here you are
that party.

    pip install 'vaara[export]'
    python prove_it.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara import InterceptionPipeline
from vaara.audit.export import export_signed
from vaara.audit.verify import verify_signed

HERE = Path(__file__).parent
BUNDLE = HERE / "evidence.zip"
TAMPERED = HERE / "evidence_tampered.zip"

RULE = "=" * 78


def banner(title: str) -> None:
    print(f"\n{RULE}\n {title}\n{RULE}")


def main() -> None:
    banner("Prove what your AI agent actually did")

    # A signing key you generate here and now. In production this lives in a
    # KMS/HSM and never touches disk; for the demo it is ephemeral.
    signer = Ed25519PrivateKey.generate()

    # An agent proposes tool calls. Vaara decides each one before it runs.
    pipeline = InterceptionPipeline()
    calls = [
        ("read a project file", "read_file",
         {"path": "README.md"}),
        ("search the support knowledge base", "search_docs",
         {"query": "how do I reset my password"}),
        ("run a destructive shell command", "shell_exec",
         {"command": "rm -rf /"}),
        ("fetch the cloud instance-metadata endpoint (classic SSRF)", "http_get",
         {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}),
    ]

    print("\n An agent (agent_id=support-bot) proposes four tool calls.")
    print(" Vaara scores and decides each one before it can run:\n")
    for label, tool, params in calls:
        result = pipeline.intercept(agent_id="support-bot", tool_name=tool, parameters=params)
        verdict = "ALLOW" if result.allowed else result.decision.upper()
        mark = "  run " if result.allowed else "BLOCK "
        print(f"   {mark} {verdict:8} risk={result.risk_score:.3f}  {label}")
        if result.allowed:
            pipeline.report_outcome(result.action_id, 0.0)

    print("\n Every decision above, allow and block alike, is now in a")
    print(" hash-chained record. Nothing was cherry-picked.")

    # Export the record as a signed, self-contained evidence bundle.
    export_signed(pipeline.trail, BUNDLE, signer_key=signer)
    print(f"\n Signed evidence bundle written: {BUNDLE.name}")

    banner("Now you verify it, offline, without trusting us")

    ok = verify_signed(BUNDLE)
    print(f"\n verify_signed({BUNDLE.name})  ->  ok={ok.ok}")
    print(" The signature checks out and the hash chain re-derives from the")
    print(" bundle's own bytes. No network, no access to the machine that made it.")

    # Forge one byte and prove the tampering is caught.
    _forge_one_record(BUNDLE, TAMPERED)
    forged = verify_signed(TAMPERED)
    print("\n Someone edits one recorded action, then re-verifies:")
    print(f" verify_signed({TAMPERED.name})  ->  ok={forged.ok}")
    if forged.errors:
        print(f"   caught: {forged.errors[0]}")

    banner("That is the whole idea")
    print("""
 A regulator, an auditor, or a customer after an incident can take this
 bundle and check exactly what the agent did, without your logs, your
 software, or your word for it. A single forged byte fails the check.

 This is evidence, not a claim. Logs say "trust me"; this does not need to.
""")


def _forge_one_record(src: Path, dst: Path) -> None:
    """Copy the bundle, flip a character inside the recorded trail."""
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}
    trail_name = next(n for n in names if n.endswith("trail.jsonl"))
    raw = data[trail_name]
    # Change an "allow" outcome to look like something it was not.
    forged = raw.replace(b"support-bot", b"other-agent", 1)
    if forged == raw:  # fallback: flip the first digit we find
        forged = bytes(b ^ 0x01 if 48 <= b <= 57 else b for b in raw[:1]) + raw[1:]
    data[trail_name] = forged
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, data[n])
    dst.write_bytes(buf.getvalue())


if __name__ == "__main__":
    main()
