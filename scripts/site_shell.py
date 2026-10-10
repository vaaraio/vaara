# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The frame every vaara.io page shares: the bar, the footer and the theme script.

One copy, so the pages cannot drift. The site build, the conformance renderer
and the hand-written pages (verify.html, surfaces.html) all take their bar and
footer from here, and tests/test_site_shell.py fails if a published page carries
any other bar.

The explorer, conformance and surfaces pages exist in English only. Their FI
link goes to the Finnish home page.
"""
from __future__ import annotations

import html

E = html.escape

#: Menu labels and the dropdown contents, per language. The sector and product
#: names repeat the page copy and the site build checks the two agree.
NAV = {
    "en": {
        "sectors": "Sectors", "products": "Products", "test": "Test it yourself", "hood": "Under the hood",
        "menu": "Main menu", "theme": "Switch between light and dark", "get": "Get Vaara",
        "sector_items": [("care", "Care and social services"), ("payments", "Payments and finance"),
                         ("public", "Public sector"), ("health", "Healthcare"),
                         ("infrastructure", "Energy, industry and infrastructure"), ("software", "Software teams")],
        "product_items": [("gate", "The gate"), ("receipts", "Receipts"), ("cage", "The cage"),
                          ("app", "The desktop app"), ("test-it-yourself", "Test it yourself")],
        "test_items": [("/products/test-it-yourself.html", "Test it yourself"),
                       ("/verify.html", "Receipt explorer"), ("/conformance.html", "Conformance results")],
        "foot_open": "Open source. Runs on your own machines. Nothing is sent to us.",
        "foot_code": "Code", "foot_draft": "Internet-Draft", "foot_conf": "Conformance results",
        "foot_tech": "Technical overview", "foot_pub": "Everything Vaara publishes",
        "foot_tm": "Vaara™ © 2026 Henri Sirkkavaara, Helsinki. AGPL-3.0-or-later.",
    },
    "fi": {
        "sectors": "Toimialat", "products": "Tuotteet", "test": "Testaa itse", "hood": "Konepellin alla",
        "menu": "Päävalikko", "theme": "Vaihda vaalean ja tumman välillä", "get": "Lataa Vaara",
        "sector_items": [("care", "Hoiva ja sosiaalipalvelut"), ("payments", "Maksut ja rahoitus"),
                         ("public", "Julkinen sektori"), ("health", "Terveydenhuolto"),
                         ("infrastructure", "Energia, teollisuus ja infrastruktuuri"), ("software", "Ohjelmistotiimit")],
        "product_items": [("gate", "Portti"), ("receipts", "Kuitit"), ("cage", "Häkki"),
                          ("app", "Työpöytäsovellus"), ("test-it-yourself", "Testaa itse")],
        "test_items": [("/fi/products/test-it-yourself.html", "Testaa itse"),
                       ("/verify.html", "Kuittien tarkistus"), ("/conformance.html", "Vaatimustenmukaisuustulokset")],
        "foot_open": "Avointa lähdekoodia. Toimii omilla koneillanne. Mitään ei lähetetä meille.",
        "foot_code": "Lähdekoodi", "foot_draft": "IETF-luonnos", "foot_conf": "Vaatimustenmukaisuustulokset",
        "foot_tech": "Tekninen yleiskuva", "foot_pub": "Kaikki, mitä Vaara julkaisee",
        "foot_tm": "Vaara™ © 2026 Henri Sirkkavaara, Helsinki. AGPL-3.0-or-later.",
    },
}

#: Same key the earlier pages used, so a visitor's saved choice carries over.
THEME_KEY = "vaara-theme"

#: The language a visitor picked with the EN | FI switch; it stops the home page choosing for them.
LANG_KEY = "vaara-lang"

#: In <head>, before anything paints: the saved theme, or the OS setting.
PREPAINT = ('<script>try{var t=localStorage.getItem("' + THEME_KEY + '");'
            'if(t!=="dark"&&t!=="light")t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";'
            'document.documentElement.dataset.theme=t;'
            # the English home sends a browser that lists Finnish to /fi/, once, until the visitor picks a language
            'var p=location.pathname;if((p==="/"||p==="/index.html")&&!localStorage.getItem("' + LANG_KEY + '")'
            '&&(navigator.languages||[navigator.language]).some(function(l){return/^fi\\b/i.test(l)}))location.replace("/fi/")'
            '}catch(e){}</script>')

#: Before </body>: the theme button and the phone menu.
BODY_SCRIPT = ('<script>document.querySelector(".bar .theme").onclick=function(){var d=document.documentElement,'
               't=d.dataset.theme==="dark"?"light":"dark";d.dataset.theme=t;try{localStorage.setItem("'
               + THEME_KEY + '",t)}catch(e){}};document.querySelector(".bar .burger").onclick=function(){'
               'var n=document.querySelector("nav.top"),o=n.classList.toggle("open");'
               'this.setAttribute("aria-expanded",o)};document.querySelectorAll(".bar .lang a").forEach(function(a){'
               'a.onclick=function(){try{localStorage.setItem("' + LANG_KEY + '",a.hreflang)}catch(e){}}})</script>')


def href(lang: str, path: str) -> str:
    return ("/fi" if lang == "fi" else "") + path


def bar(lang: str, path: str, nav_on: str = "", fi_path: str | None = None) -> str:
    """The sticky navy bar. ``fi_path`` is where the language switch points
    when the page has no Finnish twin; by default it is the same path."""
    n = NAV[lang]

    def drop(key: str, base: str, items) -> str:
        sub = "".join(f'<a href="{u}">{E(t)}</a>' for u, t in items)
        return (f'<div class="dd{" on" if key == nav_on else ""}"><a class="l" href="{base}" aria-haspopup="true">'
                f'{E(n[key])} <span class="caret">▾</span></a><div class="menu">{sub}</div></div>')

    links = (drop("sectors", href(lang, "/sectors/"),
                  [(href(lang, f"/sectors/{s}.html"), t) for s, t in n["sector_items"]])
             + drop("products", href(lang, "/products/"),
                    [(href(lang, f"/products/{s}.html"), t) for s, t in n["product_items"]])
             + drop("test", n["test_items"][0][0], n["test_items"])
             + f'<a class="l{" on" if nav_on == "hood" else ""}" href="{href(lang, "/under-the-hood.html")}">'
               f'{E(n["hood"])}</a>')
    other = fi_path if (fi_path is not None and lang == "en") else None
    if lang == "en":
        lang_sw = f'<b>EN</b> | <a href="{other or href("fi", path)}" hreflang="fi">FI</a>'
    else:
        lang_sw = f'<a href="{path[3:] if path.startswith("/fi/") else path}" hreflang="en">EN</a> | <b>FI</b>'
    return (f'<header class="bar"><div class="wrap">\n'
            f'<nav class="top" aria-label="{E(n["menu"])}"><a class="brand" href="{href(lang, "/")}">'
            f'<img src="/vaara-logo-dark.svg" alt="Vaara home" width="174" height="24"></a>{links}'
            f'<span class="lang">{lang_sw}</span>'
            f'<button class="burger" type="button" aria-label="{E(n["menu"])}" aria-expanded="false"></button>'
            f'<button class="theme" type="button" aria-label="{E(n["theme"])}" title="{E(n["theme"])}"></button>'
            f'<a class="get" href="https://github.com/vaaraio/vaara#readme">{E(n["get"])}</a></nav>\n'
            f'</div></header>')


def footer(lang: str) -> str:
    n = NAV[lang]
    return (f'<footer class="site"><div class="wrap"><div class="cols">\n'
            f'<div><b>VAARA</b><br>{E(n["foot_open"])}<br><a href="mailto:hello@vaara.io">hello@vaara.io</a></div>\n'
            f'<div><a href="https://github.com/vaaraio/vaara">{E(n["foot_code"])}</a><br>\n'
            f'<a href="https://datatracker.ietf.org/doc/draft-sirkkavaara-vaara-receipt/">{E(n["foot_draft"])}</a><br>\n'
            f'<a href="/conformance.html">{E(n["foot_conf"])}</a><br>'
            f'<a href="{href(lang, "/surfaces.html")}">{E(n["foot_pub"])}</a><br>'
            f'<a href="{href(lang, "/under-the-hood.html")}">{E(n["foot_tech"])}</a></div>\n'
            f'</div><p class="tm">{E(n["foot_tm"])}</p></div></footer>')


def band(kicker: str, h1: str, lead_html: str, crumbs: list[tuple[str, str]]) -> str:
    """The light heading area: breadcrumbs, kicker, h1, lead (lead is HTML)."""
    crumb = ('<nav class="crumbs" aria-label="Breadcrumb">'
             + " / ".join(f'<a href="{u}">{E(t)}</a>' for t, u in crumbs) + "</nav>") if crumbs else ""
    return (f'<div class="band"><div class="wrap"><div class="hero">{crumb}'
            f'<p class="kicker">{E(kicker)}</p><h1>{E(h1)}</h1><p class="lead">{lead_html}</p></div></div></div>')


#: For the pages written before the shell: their own stylesheet keeps styling
#: their content, its colour and font names are pointed at the site's, and the
#: rules it applied to bare elements are kept off the shell.
LEGACY_CSS = """<link rel="stylesheet" href="/style.css">
<style>
  html:root, html:root[data-theme] {
    --panel: var(--card); --panel-2: var(--tint); --deep: var(--tint); --line: var(--rule);
    --faint: var(--muted); --dim: var(--muted); --tri: var(--accent); --tri-bright: var(--accent);
    --ok: var(--accent); --no: var(--stop); --edge: 16px;
    --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  html:root body { background: var(--bg); color: var(--body); font: 17px/1.6 var(--sans); }
  html:root main.legacy { max-width: var(--max); margin: 0 auto; padding: 40px 24px 64px; }
  html:root main.legacy section { padding: 1rem 1.25rem; border: 1px solid var(--rule); }
  html:root main.legacy > h2 { font-size: 24px; text-transform: none; letter-spacing: 0; color: var(--ink); margin: 44px 0 12px; }
  html:root .bar a, html:root .band a, html:root footer.site a { border-bottom: 0; }
  html:root footer.site { margin: 0; padding: 40px 0; border-top: 0; color: var(--on-night-soft); font-size: 14px; line-height: 1.6; }
  html:root .band .lead { font-size: 17px; color: var(--body); max-width: 46em; margin: 0; }
  html:root .band h1 { font: 650 30px/1.2 var(--sans); letter-spacing: 0; color: var(--ink); margin: 0 0 10px; }
</style>"""
