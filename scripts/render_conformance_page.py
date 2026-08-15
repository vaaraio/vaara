#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render Vaara Conformance Results from the runner's own report.

The page is generated rather than written, for one reason: a hand-maintained
results table drifts from the checkers within a release or two, and a stale
conformance claim is worse than none. Everything numeric here comes from
``scripts/conformance_runner.py --json``. The only hand-maintained input is
``conformance/reproductions.json``, which records parties other than the
maintainer who ran the checkers and said so in public.

The page states its own authorship limit. The vectors are Vaara's and there is
no ratification body behind them, which a reader is entitled to know before
weighing a pass. What the page offers instead is that every verdict recomputes
from committed bytes with no Vaara import, so a stranger can disagree and show
their work.

Usage:
    python scripts/conformance_runner.py --json report.json
    python scripts/render_conformance_page.py report.json webpage/conformance.html
    python scripts/render_conformance_page.py report.json --check   # CI: fail if stale
"""
from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPRODUCTIONS = REPO / "conformance" / "reproductions.json"
DEFAULT_OUT = REPO / "webpage" / "conformance.html"
BADGE_DIR = REPO / "webpage" / "badge"

TITLE = "Vaara Conformance Results"

# The terms a party agrees to when they ask for a row live in reproductions.json
# under ``terms_version``, so the desk and the page cannot disagree about which
# version is current. Bump it there when the text changes, never edit a version
# in place. Each row carries the version it agreed to, so a later change reaches
# new rows only. Somebody who said yes to one set of terms is never moved onto
# another set by an edit they never saw.
FALLBACK_TERMS_VERSION = "unversioned"

# One badge per listed party, named for that party's slug. There is no generic
# badge on purpose: a single shared URL is copyable by anyone who never ran
# anything, and the copy is indistinguishable from the real thing. A per-party
# badge points at a specific row, so a stranger pasting it is claiming to be a
# named person whose record is one click away.
BADGE_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" \
height="20" role="img" aria-label="{alt}">
  <title>{alt}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <metadata>
    <vcr xmlns="https://vaara.io/ns/vcr/v1">
      <row>{row_id}</row>
      <digest>{digest}</digest>
      <over>https://vaara.io/badge/{slug}.json</over>
      <recompute>sha256 of those bytes, which are JCS-canonical per RFC 8785</recompute>
    </vcr>
  </metadata>
  <clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)">
    <rect width="{lw}" height="20" fill="#1A2226"/>
    <rect x="{lw}" width="{mw}" height="20" fill="#78A08A"/>
    <rect width="{w}" height="20" fill="url(#s)"/>
  </g>
  <g font-family="Verdana,DejaVu Sans,Geneva,sans-serif" font-size="11">
    <text x="6" y="15" fill="#000" fill-opacity=".3" textLength="{lt}" \
lengthAdjust="spacingAndGlyphs">{label}</text>
    <text x="6" y="14" fill="#fff" textLength="{lt}" \
lengthAdjust="spacingAndGlyphs">{label}</text>
    <text x="{mx}" y="15" fill="#000" fill-opacity=".3" textLength="{mt}" \
lengthAdjust="spacingAndGlyphs">{message}</text>
    <text x="{mx}" y="14" fill="#1A2226" textLength="{mt}" \
lengthAdjust="spacingAndGlyphs">{message}</text>
  </g>
</svg>
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def row_bytes(row: dict) -> bytes:
    """The published row, JCS-canonical (RFC 8785), exactly as it is served.

    The file at ``/badge/<slug>.json`` holds these bytes and nothing else, so
    checking a badge is ``sha256sum`` on a downloaded file. No parser, no
    canonicaliser and no Vaara install stands between a reader and the answer.

    Imported here rather than at module scope because ``rfc8785`` ships in the
    ``attestation`` extra, and this page has to render from a checkout that
    installed nothing. With no rows there is no digest to compute, so the
    dependency is only reached once somebody is actually listed.
    """
    import rfc8785

    return rfc8785.dumps(row)


def row_digest(row: dict) -> str:
    return "sha256:" + hashlib.sha256(row_bytes(row)).hexdigest()


def badge_svg(row: dict) -> str:
    """A badge that dates itself, so nobody has to police staleness.

    The row number, the date and the short commit are baked into the pixels.
    A badge earned two years ago reads as two years old wherever it is pasted,
    without the badge ever calling home to say so.
    """
    label = f"VCR #{row['id']}"
    message = f"reproduced {row.get('date', '')} {str(row.get('at_commit', ''))[:7]}"
    # 6.2px per character at 11px Verdana, plus 6px padding either side.
    lw = int(len(label) * 6.2) + 12
    mw = int(len(message) * 6.2) + 12
    alt = f"Vaara Conformance Results row {row['id']}: {row.get('party', '')}"
    return BADGE_TEMPLATE.format(
        w=lw + mw,
        lw=lw,
        mw=mw,
        lt=lw - 12,
        mt=mw - 12,
        mx=lw + 6,
        label=esc(label),
        message=esc(message),
        alt=esc(alt),
        row_id=esc(row["id"]),
        slug=esc(row["slug"]),
        digest=esc(row_digest(row)),
    )


CERTIFICATE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vaara Conformance Results, row {row_id}</title>
<meta name="robots" content="noindex">
<style>
  @page {{ size: A4 landscape; margin: 18mm; }}
  :root {{ --ink:#1A2226; --green:#78A08A; --faint:#8B9792; --rule:#DEE4E1; }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:28mm 24mm; color:var(--ink); background:#fff;
    font:16px/1.55 Georgia, "Times New Roman", serif;
  }}
  .sheet {{ border:2px solid var(--ink); padding:16mm 18mm; height:100%; }}
  .kicker {{
    font:600 11px/1 ui-sans-serif, system-ui, sans-serif;
    letter-spacing:.22em; text-transform:uppercase; color:var(--faint);
  }}
  h1 {{ font-size:31px; margin:10px 0 2px; letter-spacing:.01em; }}
  .num {{ color:var(--green); font-weight:700; }}
  .party {{ font-size:26px; margin:18px 0 2px; }}
  .aff {{ color:var(--faint); font-size:15px; margin:0; }}
  .claim {{ margin:16px 0 0; max-width:58ch; }}
  .scoping {{
    margin:12px 0 0; padding-left:12px; border-left:3px solid var(--rule);
    color:#3d474b; font-style:italic; max-width:62ch;
  }}
  dl {{
    display:grid; grid-template-columns:auto 1fr; gap:3px 14px;
    margin:18px 0 0; font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  dt {{ color:var(--faint); }}
  dd {{ margin:0; word-break:break-all; }}
  footer {{
    margin-top:16px; padding-top:10px; border-top:1px solid var(--rule);
    font:11px/1.5 ui-sans-serif, system-ui, sans-serif; color:var(--faint);
    max-width:74ch;
  }}
  @media print {{ body {{ padding:0; }} .noprint {{ display:none; }} }}
</style>
</head>
<body>
<div class="sheet">
  <p class="kicker">Vaara Conformance Results</p>
  <h1>Independent reproduction <span class="num">#{row_id}</span></h1>

  <p class="party">{party}</p>
  <p class="aff">{affiliation}</p>

  <p class="claim">ran the independent checkers listed below, from a clean
  checkout at the commit recorded here, with no Vaara installed, and reported
  the outcome in public. The result they reported was {result}.</p>

  <p class="scoping">{scoping}</p>

  <dl>
    <dt>Suites</dt><dd>{suites}</dd>
    <dt>At commit</dt><dd>{at_commit}</dd>
    <dt>Date</dt><dd>{date}</dd>
    <dt>Row digest</dt><dd>{digest}</dd>
    <dt>Record</dt><dd>{record}</dd>
    <dt>Terms</dt><dd>{terms_version}</dd>
  </dl>

  <footer>
    This states what those checkers returned on that date at that commit. It is
    not a certification, it does not say Vaara is compliant with anything, and
    it does not say the party above endorses Vaara. These vectors have no
    ratification process behind them and the maintainer decides what a verdict
    means. Every case recomputes from committed bytes, so anyone may disagree
    and show their work. This row is a permanent record of a run on a date and
    is never removed or edited, by anyone, including Vaara. Check this sheet
    against
    https://vaara.io/badge/{slug}.json, whose sha256 is the row digest above.
  </footer>
</div>
</body>
</html>
"""


def certificate_html(row: dict) -> str:
    """The wall version of the badge, printable without any dependency.

    A browser turns this into a PDF, so nothing here needs a rendering library
    and the sheet keeps the property the rest of the site has: no off-origin
    load, readable with the network off. It carries the digest, so the thing
    somebody pins to a wall can still be checked against the published row.
    """
    scoping = row.get("their_scoping") or (
        "No scoping was given beyond the result recorded here."
    )
    return CERTIFICATE_TEMPLATE.format(
        row_id=esc(row["id"]),
        slug=esc(row["slug"]),
        party=esc(row.get("party", "")),
        affiliation=esc(row.get("affiliation", "")),
        result=esc(row.get("result", "")),
        scoping=esc(scoping),
        suites=esc(", ".join(row.get("suites", []))),
        at_commit=esc(row.get("at_commit", "")),
        date=esc(row.get("date", "")),
        digest=esc(row_digest(row)),
        record=esc(row.get("record", "")),
        terms_version=esc(row.get("terms_version", "unversioned")),
    )


def write_badges(repro: dict, badge_dir: Path = BADGE_DIR) -> list[Path]:
    """Regenerate the badge directory from the rows, and only from the rows.

    Stale files are deleted rather than left behind, so a badge URL stops
    resolving the moment its row comes down. Withdrawal has to reach the badge
    too, otherwise removal from the page is cosmetic.
    """
    badge_dir.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, bytes] = {}
    for row in repro.get("reproductions", []):
        wanted[f"{row['slug']}.svg"] = badge_svg(row).encode("utf-8")
        wanted[f"{row['slug']}.json"] = row_bytes(row)
        wanted[f"{row['slug']}.html"] = certificate_html(row).encode("utf-8")
    for stale in [p for p in badge_dir.iterdir() if p.is_file()]:
        if stale.name not in wanted:
            stale.unlink()
    written = []
    for name, blob in wanted.items():
        path = badge_dir / name
        path.write_bytes(blob)
        written.append(path)
    return written


def suite_rows(report: dict) -> str:
    out = []
    for suite in sorted(report["suites"], key=lambda s: s["suite"]):
        status = suite["status"]
        cls = {"PASS": "ok", "FAIL": "bad", "SKIP": "skip"}.get(status, "skip")
        reason = suite.get("reason") or ""
        out.append(
            f'<tr><td class="mono">{esc(suite["suite"])}</td>'
            f'<td class="{cls}">{esc(status)}</td>'
            f'<td class="why">{esc(reason)}</td></tr>'
        )
    return "\n".join(out)


def reproduction_blocks(data: dict) -> str:
    rows = data.get("reproductions", [])
    if not rows:
        return (
            '<p class="none">No independent reproduction has been recorded yet. '
            "A row appears here when someone other than the maintainer runs the "
            "checkers, reports the result somewhere public, and asks to be "
            "listed.</p>"
        )
    out = []
    for r in rows:
        suites = ", ".join(r.get("suites", []))
        badge = f"/badge/{r['slug']}.svg"
        snippet = (
            f"[![Vaara Conformance Results row {r['id']}]"
            f"(https://vaara.io{badge})](https://vaara.io/conformance.html)"
        )
        out.append(f"""
        <article class="repro">
          <div class="repro-head">
            <span class="num mono">#{esc(r['id'])}</span>
            <span class="date mono">{esc(r.get('date', ''))}</span>
            <span class="who">{esc(r.get('party', ''))}</span>
            <span class="aff">{esc(r.get('affiliation', ''))}</span>
          </div>
          <p class="result">{esc(r.get('result', ''))}</p>
          <dl>
            <dt>Suites</dt><dd class="mono">{esc(suites)}</dd>
            <dt>At commit</dt><dd class="mono">{esc(r.get('at_commit', ''))}</dd>
            <dt>Their scoping</dt><dd class="scoping">{esc(r.get('their_scoping', ''))}</dd>
            <dt>Record</dt><dd><a href="{esc(r.get('record', ''))}" rel="noopener">{esc(r.get('record_held_by', 'record'))}</a></dd>
            <dt>Listed under terms</dt><dd class="mono">{esc(r.get('terms_version', 'unversioned'))}</dd>
            <dt>Row digest</dt><dd class="mono">{esc(row_digest(r))}<br>over <a href="/badge/{esc(r['slug'])}.json">/badge/{esc(r['slug'])}.json</a></dd>
            <dt>Badge</dt><dd><img src="{esc(badge)}" alt="{esc('VCR row ' + str(r['id']))}" height="20"><br><code class="snippet">{esc(snippet)}</code></dd>
            <dt>Certificate</dt><dd><a href="/badge/{esc(r['slug'])}.html">printable sheet</a></dd>
          </dl>
        </article>""")
    return "\n".join(out)


def render(report: dict, repro: dict) -> str:
    t = report["totals"]
    generated = report.get("generated_at", "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE} | Vaara</title>
<meta name="description" content="Every Vaara conformance suite, its verdict, and every independent party who reproduced the vectors and said so in public. Generated from the runner's own report.">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://vaara.io/conformance.html">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<meta property="og:type" content="website">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="Every suite, every verdict, every independent reproduction. Recomputable from committed bytes with no Vaara import.">
<meta property="og:url" content="https://vaara.io/conformance.html">
<style>
  :root {{
    color-scheme: light;
    --bg:#ececec; --panel:#FFFFFF; --panel-2:#F1F4F0; --line:rgba(60,90,72,.18);
    --ink:#1A2226; --muted:#566159; --faint:#7C857E;
    --tri:#3E6B54; --tri-bright:#2C523E; --warn:#B4524F;
    --mono:ui-monospace,'SFMono-Regular','JetBrains Mono',Menlo,Consolas,monospace;
    --sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  }}
  html[data-theme="dark"], html:not([data-theme]) {{}}
  @media (prefers-color-scheme: dark) {{
    html:not([data-theme]) {{
      color-scheme: dark;
      --bg:#0F1417; --panel:#1A2226; --panel-2:#151D20; --line:rgba(123,156,138,.16);
      --ink:#DEE4E1; --muted:#8B9792; --faint:#5E6A66;
      --tri:#78A08A; --tri-bright:#93B8A3; --warn:#D98B8B;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
       line-height:1.55;-webkit-font-smoothing:antialiased}}
  .wrap{{max-width:940px;margin:0 auto;padding:56px 22px 96px}}
  a{{color:var(--tri-bright);text-decoration:none;border-bottom:1px solid var(--line)}}
  a:hover{{border-bottom-color:var(--tri)}}
  .back{{font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;
        color:var(--muted);border:0}}
  h1{{font-size:clamp(28px,4.2vw,40px);line-height:1.12;margin:26px 0 6px;letter-spacing:-.02em}}
  .kicker{{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;
          color:var(--tri);margin:0}}
  h2{{font-size:19px;margin:52px 0 10px;letter-spacing:-.01em}}
  p{{margin:0 0 14px;max-width:74ch}}
  .lede{{font-size:17px;color:var(--muted)}}
  .mono{{font-family:var(--mono);font-size:12.5px}}
  .stats{{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0 8px}}
  .stat{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
        padding:12px 16px;min-width:104px}}
  .stat b{{display:block;font-family:var(--mono);font-size:23px;color:var(--tri-bright);
          line-height:1.1}}
  .stat span{{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}}
  table{{width:100%;border-collapse:collapse;margin:14px 0 8px;background:var(--panel);
        border:1px solid var(--line);border-radius:10px;overflow:hidden}}
  th,td{{text-align:left;padding:8px 14px;border-bottom:1px solid var(--line);font-size:13.5px;
        vertical-align:top}}
  th{{font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;
     color:var(--faint);background:var(--panel-2)}}
  tr:last-child td{{border-bottom:0}}
  td.ok{{color:var(--tri-bright);font-family:var(--mono);font-size:12px}}
  td.bad{{color:var(--warn);font-family:var(--mono);font-size:12px;font-weight:600}}
  td.skip{{color:var(--faint);font-family:var(--mono);font-size:12px}}
  td.why{{color:var(--muted);font-size:12.5px}}
  .repro{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:18px 20px;margin:14px 0}}
  .repro-head{{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;margin-bottom:8px}}
  .repro-head .who{{font-weight:600}}
  .repro-head .aff{{color:var(--faint);font-size:13px}}
  .repro .result{{margin:0 0 12px}}
  dl{{display:grid;grid-template-columns:auto 1fr;gap:5px 16px;margin:0;font-size:13px}}
  dt{{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;
     color:var(--faint);padding-top:2px}}
  dd{{margin:0;overflow-wrap:anywhere}}
  .scoping{{color:var(--muted);font-style:italic}}
  .none{{color:var(--muted)}}
  .note{{border-left:2px solid var(--line);padding-left:16px;color:var(--muted);
        font-size:14px;margin:18px 0}}
  code{{font-family:var(--mono);font-size:12.5px;background:var(--panel-2);
       padding:1px 5px;border-radius:4px}}
  pre{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:14px 16px;overflow-x:auto}}
  pre code{{background:none;padding:0}}
  footer{{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);
         color:var(--faint);font-size:12.5px}}
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="/">&larr; vaara.io</a>

  <p class="kicker">Vaara Conformance Results</p>
  <h1>Every suite, every verdict, and everyone who checked it themselves.</h1>

  <p class="lede">Each suite below ships an independent checker that imports no
  Vaara code and recomputes its verdicts from the bytes of its own case files.
  This page is generated from the runner's report, so it says what the checkers
  did rather than what a document claims they do.</p>

  <div class="stats">
    <div class="stat"><b>{t['suites']}</b><span>suites</span></div>
    <div class="stat"><b>{t['passed']}</b><span>passed</span></div>
    <div class="stat"><b>{t['failed']}</b><span>failed</span></div>
    <div class="stat"><b>{t['skipped']}</b><span>skipped</span></div>
    <div class="stat"><b>{t['cases_passed']}</b><span>cases</span></div>
  </div>
  <p class="mono" style="color:var(--faint)">generated {esc(generated)}</p>

  <h2>Run it yourself</h2>
  <p>Nothing here needs Vaara installed. The checkers use the standard library
  plus <code>cryptography</code> and <code>rfc8785</code>.</p>
<pre><code>git clone https://github.com/vaaraio/vaara
cd vaara
pip install cryptography rfc8785
python scripts/conformance_runner.py</code></pre>
  <p>Point <code>--vectors-dir</code> at a directory laid out the same way and it
  grades those bytes instead. The runner collects; the checkers decide.</p>

  <h2>What a pass does not establish</h2>
  <p>These vectors are Vaara's and there is no ratification process behind them.
  The maintainer adds the cases and decides what a verdict means, and nobody else
  has a vote. That is a real limit on any neutrality claim and it is stated here
  rather than argued around.</p>
  <p class="note">What the corpus does give is narrower and checkable: every case
  recomputes from committed bytes with no Vaara import, so anyone can disagree
  with an expected result and show their work. Recompute is checkable by
  strangers. Authorship is not.</p>

  <h2>Independent reproductions</h2>
  <p>Parties other than the maintainer who ran the checkers and reported the
  outcome in public. Claims are quoted as each party scoped them. Everyone here
  asked to be listed, knowing the entry is permanent.
  To add your own run, <a
  href="https://github.com/vaaraio/vaara/issues/new?template=conformance-row.yml">open
  a row request</a>. Row numbers are permanent and each listed party gets its
  own badge, carrying that number, the date and the commit. A badge says the
  party ran the checkers at that commit on that date. It never says currently
  passing, and it is not a certification.</p>

  <h3>Checking a badge without trusting this page</h3>
  <p>Each badge carries the digest of its own row inside the SVG, and the row
  it commits to is served as JCS-canonical bytes (RFC 8785) next to it. The
  file holds those bytes and nothing else, so verifying a badge is one command
  and needs no parser, no canonicaliser and no Vaara installed.</p>
<pre><code>curl -s https://vaara.io/badge/&lt;slug&gt;.json | sha256sum
curl -s https://vaara.io/badge/&lt;slug&gt;.svg | grep digest

python scripts/vcr_chain.py            # nothing removed from the table</code></pre>
  <p class="note">The two agree or the badge is not describing that row. This
  is the same property the corpus runs on, applied to the badge itself: a
  claim that recomputes from bytes, checkable by someone who trusts neither
  the party nor Vaara.</p>

  <h3>Terms for a listed party, version {esc(repro.get('terms_version', FALLBACK_TERMS_VERSION))}</h3>
  <p class="note">Stated before there is anything to sell, so that anyone
  deciding whether to be listed can read the commercial position at the moment
  they decide rather than learn it afterwards.</p>
  <ul class="terms">
    <li><strong>A row is permanent.</strong> This records something that
    happened. A run took place on a date at a commit, and that stays true, so
    the entry stays. Asking will not remove it, and the maintainer cannot
    remove it either. Rows are chained, so a deletion or a reordering breaks
    every digest after it and anybody can see the break.</li>
    <li>Nothing is edited after the fact. A correction is a new row referring
    to the earlier one, never a rewrite of it.</li>
    <li>A row and its badge are free, and stay free. There is no paid tier of
    being listed and no version of a row that counts for more.</li>
    <li>Vaara may sell services built on these same vectors, including a
    countersigned result. Being listed here is never a condition of buying
    anything, and buying something is never a condition of being listed.</li>
    <li>No listed party's name, affiliation or badge is used to sell anything.
    Rows are a record, not a customer list and not a reference.</li>
    <li>Each row records the version of these terms it agreed to. Terms may
    change for rows added later. A row already listed keeps the terms it was
    listed under, and a change is never applied backwards.</li>
  </ul>
{reproduction_blocks(repro)}

  <h2>Suites</h2>
  <table>
    <thead><tr><th>Suite</th><th>Verdict</th><th>Note</th></tr></thead>
    <tbody>
{suite_rows(report)}
    </tbody>
  </table>

  <footer>
    Generated by <code>scripts/render_conformance_page.py</code> from
    <code>scripts/conformance_runner.py --json</code>. Independent reproductions
    are maintained in <code>conformance/reproductions.json</code> and added by
    pull request. Vaara is AGPL-3.0-or-later.
  </footer>
</div>
</body>
</html>
"""


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    report_path = Path(argv[0])
    check = "--check" in argv
    out = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("-") else DEFAULT_OUT

    report = json.loads(report_path.read_text(encoding="utf-8"))
    repro = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    page = render(report, repro)

    if check:
        # The generated timestamp changes every run, so compare everything else.
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        strip = lambda s: "\n".join(  # noqa: E731
            ln for ln in s.splitlines() if "generated " not in ln
        )
        stale_badges = badge_drift(repro)
        if strip(current) != strip(page) or stale_badges:
            detail = f" Badges out of sync: {', '.join(stale_badges)}." if stale_badges else ""
            print(
                f"{out} is stale.{detail} Regenerate:\n"
                "  python scripts/conformance_runner.py --json /tmp/report.json\n"
                f"  python scripts/render_conformance_page.py /tmp/report.json {out}",
                file=sys.stderr,
            )
            return 1
        print(f"{out} is current.")
        return 0

    out.write_text(page, encoding="utf-8")
    badges = write_badges(repro)
    print(
        f"wrote {out} ({t_len(page)} bytes, {report['totals']['suites']} suites, "
        f"{len(badges)} badges)"
    )
    return 0


def badge_drift(repro: dict, badge_dir: Path = BADGE_DIR) -> list[str]:
    """Names of badge files that do not match the rows, in either direction."""
    wanted: dict[str, bytes] = {}
    for r in repro.get("reproductions", []):
        wanted[f"{r['slug']}.svg"] = badge_svg(r).encode("utf-8")
        wanted[f"{r['slug']}.json"] = row_bytes(r)
        wanted[f"{r['slug']}.html"] = certificate_html(r).encode("utf-8")
    on_disk: set[str] = set()
    if badge_dir.exists():
        on_disk = {p.name for p in badge_dir.iterdir() if p.is_file()}
    drift = sorted(on_disk - set(wanted))
    for name, blob in wanted.items():
        path = badge_dir / name
        if not path.exists() or path.read_bytes() != blob:
            drift.append(name)
    return sorted(set(drift))


def t_len(s: str) -> int:
    return len(s.encode("utf-8"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
