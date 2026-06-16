# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""The console's single-page view.

One self-contained HTML string: inline CSS, vanilla JS, no build step, in the
house style of ``.vaara-site/site.py`` and ``ui/apps/clock.html``. Left column is
the chat with the sovereign local brain; right column is the live proof of that
chat. Receipt + signature verify on every turn; the hardware chain and the
second-model cross-check are deliberate, on-demand buttons. All DOM is built with
``textContent`` (no ``innerHTML``), so model output and verdict strings can never
inject markup.
"""

from __future__ import annotations

CONSOLE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vaara console</title>
<style>
  :root { --bg:#1a1a1a; --panel:#202225; --line:#34373d; --ink:#eaeaea;
          --dim:#8b8f96; --ok:#5fb878; --bad:#e0625a; --wait:#5a5f66;
          --accent:#5fb878; --sage:#8fae98; --field:#121315;
          --mono:'JetBrains Mono','Fira Code','SF Mono','Menlo','Consolas',
            'Liberation Mono',monospace; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
    font:14px/1.55 var(--mono); }
  .wrap { display:grid; grid-template-columns:1fr 372px; height:100%; }
  .col { display:flex; flex-direction:column; min-height:0; }
  .col.proof { border-left:1px solid var(--line); background:var(--panel); }
  header { padding:13px 16px; border-bottom:1px solid var(--line);
    display:flex; gap:10px; align-items:center; min-height:80px; }
  header b { font-weight:600; } header .dim { color:var(--dim); font-size:12px; }
  .brand { display:flex; align-items:center; gap:9px; }
  .brand img.brandimg { display:block; flex:none; height:56px; width:auto; }
  #log { flex:1; overflow:auto; padding:16px; }
  .msg { margin:0 0 14px; white-space:pre-wrap; }
  .msg .who { color:var(--dim); font-size:12px; margin-bottom:2px; }
  .msg.you .who { color:var(--accent); }
  .bar { display:flex; gap:8px; padding:12px 16px; border-top:1px solid var(--line); }
  textarea { flex:1; resize:none; height:46px; background:var(--field); color:var(--ink);
    border:1px solid var(--line); border-radius:8px; padding:10px; font:inherit; }
  textarea:focus { outline:none; border-color:var(--accent); }
  button { background:#26282c; color:var(--ink); border:1px solid var(--line);
    border-radius:8px; padding:8px 14px; cursor:pointer; font:inherit; }
  button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
  button:disabled { opacity:.45; cursor:default; }
  select.pick { background:var(--field); color:var(--ink); border:1px solid var(--line);
    border-radius:6px; padding:5px 8px; font:inherit; font-size:12px; max-width:180px; }
  select.pick:focus { outline:none; border-color:var(--accent); }
  .modlbl { margin-left:auto; color:var(--dim); font-size:12px; }
  #judge { margin-left:auto; }
  #judge[hidden] { display:none; }
  .proof .body { flex:1; overflow:auto; padding:16px; }
  .card { border:1px solid var(--line); border-radius:10px; padding:12px;
    margin-bottom:14px; background:var(--field); }
  .row { display:flex; align-items:center; gap:8px; padding:3px 0; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--wait); flex:none; }
  .dot.ok { background:var(--ok); } .dot.bad { background:var(--bad); }
  .row .k { color:var(--dim); } .row .v { margin-left:auto; font-variant:tabular-nums; }
  .v.continuous { color:var(--ok); font-weight:600; }
  .acts { display:flex; gap:8px; margin-top:10px; }
  .acts button { padding:6px 10px; font-size:12px; }
  .hint { color:var(--dim); font-size:12px; padding:0 16px 12px; }
  .ground { display:flex; align-items:center; gap:5px; color:var(--dim);
    font-size:12px; white-space:nowrap; user-select:none; }
  .ground[hidden] { display:none; }
  .ground input { accent-color:var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <div class="col">
    <header>
      <span class="brand">
        <img class="brandimg" src="__WORDMARK__" alt="Vaara — sovereign and governed">
      </span>
      <span class="modlbl">Model</span>
      <select id="model" class="pick" title="subject model"></select>
    </header>
    <div id="log"></div>
    <div class="bar">
      <textarea id="in" placeholder="Ask the local brain. Each turn is signed and verified."></textarea>
      <label class="ground" id="groundwrap" title="Answer using your local memory" hidden>
        <input type="checkbox" id="ground" checked> memory</label>
      <button id="send">Send</button>
    </div>
  </div>
  <div class="col proof">
    <header><b>Cross-proof</b>
      <select id="judge" class="pick" title="cross-check judge model" hidden></select></header>
    <div class="body" id="proof">
      <div class="hint">Send a message. Every turn emits a signed receipt and is
        verified here. Then prove it harder: trace it to the hardware root, or
        have a second model check the answer.</div>
    </div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const log = $("#log"), proof = $("#proof");
let messages = [];

function el(tag, cls, text){
  const n = document.createElement(tag);
  if(cls) n.className = cls;
  if(text != null) n.textContent = text;
  return n;
}
function mkrow(label, state, value){
  const r = el("div", "row");
  const d = el("div", "dot" + (state===true?" ok":state===false?" bad":""));
  const k = el("span", "k", label);
  const v = el("span", "v" + (value==="continuous"?" continuous":""), value||"");
  r.append(d, k, v); return r;
}
function add(role, text){
  const d = el("div", "msg " + (role==="user"?"you":"ai"));
  d.append(el("div", "who", role==="user"?"you":"local brain"));
  const body = el("div", null, text); d.append(body);
  log.append(d); log.scrollTop = log.scrollHeight; return body;
}
// Injection-safe markdown: render **bold** as <strong>, nothing else. Built
// from text nodes and elements (never innerHTML), so model output still cannot
// inject markup. Split on the "**" delimiter; each closed pair is bold, a
// trailing unmatched "**" (mid-stream, before its closer arrives) stays literal.
function renderMd(node, text){
  node.textContent = "";
  const parts = text.split("**");
  for(let i=0; i<parts.length; i++){
    if(i % 2 === 1 && i < parts.length - 1){
      node.append(el("strong", null, parts[i]));
    } else {
      node.appendChild(document.createTextNode(i % 2 === 1 ? "**" + parts[i] : parts[i]));
    }
  }
}

function renderTurn(t){
  const card = el("div", "card");
  if(!t || !t.available){ card.append(mkrow("no receipt emitted", false, "")); proof.prepend(card); return; }
  const v = t.verdict || {};
  card.append(mkrow("receipt signed", v.receiptSignature, v.tier||""));
  card.append(mkrow("signature verified", v.ok, "#"+t.counter));
  if("attestationFresh" in v)
    card.append(mkrow("attestation", v.attestationSignature, v.attestationFresh?"live":"archived"));
  if(t.grounded)
    card.append(mkrow("memory-grounded", true, t.grounded + (t.grounded===1?" slice":" slices")));
  const acts = el("div", "acts");
  const bc = el("button", null, "Verify hardware chain");
  const bx = el("button", null, "Cross-check (2nd model)");
  bc.onclick = () => chain(bc, card); bx.onclick = () => cross(bx, card);
  acts.append(bc, bx); card.append(acts); proof.prepend(card);
}

async function chain(btn, card){
  btn.disabled = true; const lbl = btn.textContent; btn.textContent = "verifying...";
  const r = await fetch("/api/verify-chain", {method:"POST"}).then(x=>x.json());
  if(!r.available) card.append(mkrow("hardware chain", null, r.reason||"unavailable"));
  else { card.append(mkrow("hardware chain", r.chainContinuous, r.chainTier||"?"));
    if(r.boundCount > r.nReceipts)
      // Manifest binds more turns than are present: a snapshot from a prior
      // session, not tamper. Show it honestly as stale (grey, not red) so a
      // pre-capture state never reads as a broken chain. The verdict stays
      // strict server-side -- this row asserts no coverage, it just stops
      // crying wolf.
      card.append(mkrow("receipts bound", null,
        "stale ("+r.boundCount+" bound, "+r.nReceipts+" present)"));
    else
      card.append(mkrow("receipts bound", r.manifestCoversPrefix, r.boundCount+" of "+r.nReceipts));
    if(r.unboundTail) card.append(mkrow("pending capture", null,
      r.unboundTail + (r.unboundTail===1?" turn":" turns"))); }
  btn.textContent = lbl; btn.disabled = false;
}
async function cross(btn, card){
  btn.disabled = true; const lbl = btn.textContent; btn.textContent = "asking 2nd model...";
  const jsel = $("#judge"); const jm = (jsel && jsel.value || "").trim();
  const r = await fetch("/api/crosscheck", {method:"POST",
    headers:{"content-type":"application/json"},
    body: JSON.stringify({judge_model: jm})}).then(x=>x.json());
  if(!r.available) card.append(mkrow("2nd model", null, r.reason||"unavailable"));
  else if(r.error) card.append(mkrow("2nd model", false, r.error));
  else card.append(mkrow("2nd model: "+r.agreement, r.agreement==="equivalent",
    r.diverse?"diverse":"same weights"));
  btn.textContent = lbl; btn.disabled = false;
}

function fill(sel, models, pick){
  const opts = models.slice();
  if(pick && opts.indexOf(pick) === -1) opts.unshift(pick);
  // Label drops the ":latest" tag (it crowds the narrow picker); the option
  // value keeps the full model ref so the backend still gets an exact name.
  opts.forEach(m => { const o = el("option", null, m.replace(/:latest$/, "")); o.value = m;
    if(m === pick) o.selected = true; sel.append(o); });
  return opts;
}

async function loadConfig(){
  let cfg = {};
  try { cfg = await fetch("/api/config").then(x=>x.json()); } catch(e){}
  let models = cfg.models || [];
  // Reveal surface: show only the Vaara-branded local models, never the raw
  // upstream ollama zoo. Fall back to the full list only in dev, when no
  // vaara-* model is installed.
  const branded = models.filter(m => m.toLowerCase().startsWith("vaara"));
  if(branded.length) models = branded;
  const subject = models.find(m => m.toLowerCase().includes("brain")) || models[0] || "";
  const opts = fill($("#model"), models, subject);
  if(!opts.length){ const o = el("option", null, "(start the proxy to list models)");
    o.value = ""; $("#model").append(o); }
  if(cfg.recall) $("#groundwrap").hidden = false;
  if(cfg.crosscheck){
    // Default the judge to the branded cross-verifier (else the configured
    // fallback, else any model that differs from the subject) so the diversity
    // check passes out of the box.
    const pick = models.find(m => m.toLowerCase().includes("verifier"))
      || cfg.judgeDefault || models.find(m => m !== subject) || subject;
    fill($("#judge"), models, pick);
    $("#judge").hidden = false;
  }
}

async function send(){
  const text = $("#in").value.trim(); if(!text) return;
  const model = $("#model").value.trim();
  if(!model){ add("local brain", "No model selected. Start the local proxy, then reload."); return; }
  $("#in").value = ""; $("#send").disabled = true;
  add("user", text); messages.push({role:"user", content:text});
  const body = add("assistant", ""); let acc = "";
  const ground = !$("#groundwrap").hidden && $("#ground").checked;
  const resp = await fetch("/api/chat", {method:"POST",
    headers:{"content-type":"application/json"},
    body: JSON.stringify({model, messages, stream:true, ground})});
  const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
  for(;;){ const {done, value} = await reader.read(); if(done) break;
    buf += dec.decode(value, {stream:true}); const lines = buf.split("\n"); buf = lines.pop();
    for(const ln of lines){ if(!ln.trim()) continue;
      try{ const o = JSON.parse(ln); const c = (o.message||{}).content;
        if(c){ acc += c; renderMd(body, acc); log.scrollTop = log.scrollHeight; } }catch(e){} } }
  messages.push({role:"assistant", content:acc});
  const t = await fetch("/api/turn/latest").then(x=>x.json());
  renderTurn(t); $("#send").disabled = false; $("#in").focus();
}
$("#send").onclick = send;
$("#in").addEventListener("keydown", e=>{
  if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send(); }});
loadConfig();
</script>
</body>
</html>
"""
