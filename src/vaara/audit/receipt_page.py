"""Render a Vaara receipt as a self-contained static HTML evidence page.

One receipt in, one HTML document out: the decision, the signed-payload
digest, every timestamp anchor, and the commitment chain from receipt to
witness, plus the exact commands a skeptic runs to check it without trusting
the page. No JavaScript, no external assets, no network; the file can be
opened from disk, attached to an email, or hosted anywhere.

The page never claims more than what is verifiable offline here: for
``opentimestamps`` anchors it re-checks the proof against the receipt with
:func:`vaara.audit.ots_anchor.verify_ots_anchor` when the ``ots`` extra is
installed, and otherwise reports the anchor's recorded status as an
unverified claim. Bitcoin block headers are never verified; the verify
section tells the reader how to do that themselves.

Branding: edit ``BRAND`` (colors, name, tagline) to rebrand the page; the
values below are the Vaara palette.
"""

from __future__ import annotations

import html
from typing import Any

from vaara.audit.receipt_anchor import _signed_payload_digest

# Edit these to rebrand the page. Colors are CSS values.
BRAND: dict[str, str] = {
    "name": "Vaara",
    "tagline": "verifiable evidence of AI agent actions",
    "bg": "#fcfcfb",
    "grid": "#e8e7e2",
    "muted": "#8a897f",
    "text": "#0b0b0b",
    "text2": "#52514e",
    "accent": "#2a78d6",
    "alert": "#e34948",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _ots_detail(receipt: dict, anchor: dict) -> tuple[str, list[int], list[str], bool]:
    """(status, bitcoin block heights, pending calendars, verified_here).

    ``verified_here`` is True only when the proof was actually re-checked
    against the receipt in this process; otherwise the status is whatever the
    anchor claims.
    """
    try:
        from vaara.audit.ots_anchor import verify_ots_anchor
        from vaara.audit.timeanchor import TimeAnchorError
    except ImportError:
        return str(anchor.get("status", "unknown")), [], [], False
    try:
        result = verify_ots_anchor(receipt, anchor)
    except TimeAnchorError as exc:
        return f"INVALID: {exc}", [], [], True
    return (result["status"], result["bitcoin_block_heights"],
            result["pending_calendars"], True)


def _chain_steps(receipt: dict, digest_hex: str) -> list[tuple[str, str]]:
    """(label, value) pairs for the commitment-chain figure, in order."""
    steps: list[tuple[str, str]] = []
    back = receipt.get("backLink") or {}
    if back.get("attestationDigest"):
        steps.append(("previous attestation (backLink)",
                      back["attestationDigest"]))
    evidence = (receipt.get("decisionDerived") or {}).get("evidenceRef") or {}
    if evidence.get("digest"):
        steps.append(("decision evidence (JCS digest)", evidence["digest"]))
    steps.append(("signed payload, sha256 (anchoredDigest)",
                  "sha256:" + digest_hex))
    for anchor in receipt.get("timestampAnchors") or []:
        if not isinstance(anchor, dict):
            continue
        method = anchor.get("method", "?")
        if method == "opentimestamps":
            steps.append(("OpenTimestamps calendars",
                          ", ".join(anchor.get("calendars") or ["?"])))
        else:
            steps.append((f"{method} anchor",
                          anchor.get("anchoredDigest", "")))
    return steps


def render_receipt_page(receipt: dict, *, title: str | None = None) -> str:
    """Render ``receipt`` (SPEC.md envelope) to a standalone HTML page."""
    digest_hex = _signed_payload_digest(receipt).hex()
    issuer = receipt.get("issuerAsserted") or {}
    decision = receipt.get("decisionDerived") or {}
    anchors = [a for a in receipt.get("timestampAnchors") or []
               if isinstance(a, dict)]

    anchor_rows = []
    bitcoin_heights: list[int] = []
    for i, anchor in enumerate(anchors):
        method = anchor.get("method", "?")
        if method == "opentimestamps":
            status, heights, pending, verified = _ots_detail(receipt, anchor)
            bitcoin_heights.extend(heights)
            if heights:
                detail = "Bitcoin block " + ", ".join(str(h) for h in heights)
            elif pending:
                detail = f"pending at {len(pending)} calendar(s)"
            else:
                detail = ""
            checked = ("verified against this receipt offline" if verified
                       else "status as recorded, not re-checked here")
        else:
            status = str(anchor.get("status", "recorded"))
            detail = _esc(anchor.get("tsa", anchor.get("anchoredDigest", "")))
            checked = "status as recorded, not re-checked here"
        cls = "ok" if ("confirmed" in status or status == "recorded") else (
            "bad" if "INVALID" in status else "wait")
        anchor_rows.append(
            f'<tr><td>{i}</td><td>{_esc(method)}</td>'
            f'<td><span class="pill {cls}">{_esc(status)}</span></td>'
            f'<td>{_esc(detail)}<div class="fine">{_esc(checked)}</div></td></tr>'
        )

    chain_html = "".join(
        f'<div class="step"><div class="lbl">{_esc(label)}</div>'
        f'<div class="val">{_esc(value)}</div></div><div class="arrow">&#8595;</div>'
        for label, value in _chain_steps(receipt, digest_hex)
    )
    final = ("Bitcoin block " + ", ".join(str(h) for h in sorted(set(bitcoin_heights)))
             if bitcoin_heights else "witness anchors above")
    chain_html += (f'<div class="step final"><div class="lbl">public witness</div>'
                   f'<div class="val">{_esc(final)}</div></div>')

    facts = [
        ("Issuer", issuer.get("iss", "")),
        ("Subject", issuer.get("sub", "")),
        ("Issued at", issuer.get("iat", "")),
        ("Decision", decision.get("decision", "")),
        ("Policy", decision.get("policyId", "")),
        ("Reason", decision.get("reason", "")),
        ("Signature alg", receipt.get("alg", "")),
    ]
    facts_html = "".join(
        f'<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>'
        for k, v in facts if v
    )

    b = BRAND
    page_title = title or f'{b["name"]} receipt evidence'
    has_ots = any(a.get("method") == "opentimestamps" for a in anchors)
    verify_cmds = _esc(
        "vaara receipt upgrade-ots receipt.json   # fold in Bitcoin finality\n"
        "ots verify payload.ots                   # reference client, against "
        "a Bitcoin node\n"
        "# or drop payload.ots on opentimestamps.org"
    ) if has_ots else _esc("vaara verify-bundle bundle.json")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(page_title)}</title>
<style>
:root {{
  --bg:{b["bg"]}; --grid:{b["grid"]}; --muted:{b["muted"]};
  --text:{b["text"]}; --text2:{b["text2"]};
  --accent:{b["accent"]}; --alert:{b["alert"]};
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
  line-height:1.5; }}
main {{ max-width:720px; margin:0 auto; padding:48px 24px 64px; }}
header {{ border-bottom:1px solid var(--grid); padding-bottom:16px;
  margin-bottom:32px; }}
h1 {{ font-size:20px; margin:0; letter-spacing:0.02em; }}
h1 small {{ color:var(--muted); font-weight:400; margin-left:8px; }}
h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:0.08em;
  color:var(--text2); margin:36px 0 12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:6px 12px 6px 0; vertical-align:top;
  border-bottom:1px solid var(--grid); }}
th {{ color:var(--text2); font-weight:500; white-space:nowrap; width:1%; }}
code, .val {{ font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
  font-size:12px; word-break:break-all; }}
.pill {{ display:inline-block; padding:1px 10px; border-radius:999px;
  font-size:12px; border:1px solid var(--grid); }}
.pill.ok {{ color:#1d7a34; border-color:#1d7a34; }}
.pill.wait {{ color:var(--accent); border-color:var(--accent); }}
.pill.bad {{ color:var(--alert); border-color:var(--alert); }}
.chain {{ margin-top:8px; }}
.step {{ border:1px solid var(--grid); border-radius:6px; padding:10px 14px;
  background:#fff; }}
.step .lbl {{ font-size:12px; color:var(--text2); }}
.step.final {{ border-color:var(--text); }}
.arrow {{ text-align:center; color:var(--muted); padding:2px 0; }}
.fine {{ font-size:12px; color:var(--muted); }}
pre {{ background:#fff; border:1px solid var(--grid); border-radius:6px;
  padding:14px; font-size:12px; overflow-x:auto; }}
footer {{ margin-top:48px; padding-top:16px; border-top:1px solid var(--grid);
  font-size:12px; color:var(--muted); }}
</style>
</head>
<body>
<main>
<header>
  <h1>{_esc(b["name"])} <small>{_esc(b["tagline"])}</small></h1>
</header>

<h2>Receipt</h2>
<table>{facts_html}</table>

<h2>Timestamp anchors</h2>
<table>
<tr><th>#</th><th>Method</th><th>Status</th><th>Detail</th></tr>
{"".join(anchor_rows)}
</table>

<h2>Commitment chain</h2>
<p class="fine">Each value commits the one above it. Changing any recorded
fact changes the signed payload's sha256, which no longer matches what the
witnesses hold.</p>
<div class="chain">{chain_html}</div>

<h2>Verify it yourself</h2>
<p class="fine">This page is a rendering, not the evidence. The receipt JSON
is the evidence; check it with tools that do not trust this page:</p>
<pre>{verify_cmds}</pre>

<footer>Rendered by {_esc(b["name"])} from the receipt JSON.
Signed payload sha256: <code>{_esc(digest_hex)}</code></footer>
</main>
</body>
</html>
"""


__all__ = ["BRAND", "render_receipt_page"]
