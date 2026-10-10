
"use strict";

const $ = (id) => document.getElementById(id);

// Every sentence a reader sees goes through say(). (Not t: the theme script in <head> owns a global t.) English is written inline as the
// default; a page in another language sets window.VAARA_T before this file loads.
const T = window.VAARA_T || {};
const say = (key, en, vars) => String(T[key] ?? en).replace(/\{(\w+)\}/g, (_, k) => (vars && k in vars) ? vars[k] : "");
const NUM = document.documentElement.lang || "en";
const enc = new TextEncoder();

function b64ToBytes(s) {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function hex(buf) {
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
}

// DSSE v1 pre-authentication encoding. The signature is over this, never over
// the raw payload, so a payload cannot be lifted into a different payloadType
// and still verify.
function pae(payloadType, payloadBytes) {
  const typeBytes = enc.encode(payloadType).length;
  const head = enc.encode(`DSSEv1 ${typeBytes} ${payloadType} ${payloadBytes.length} `);
  const out = new Uint8Array(head.length + payloadBytes.length);
  out.set(head, 0);
  out.set(payloadBytes, head.length);
  return out;
}

// SubjectPublicKeyInfo for Ed25519 is a fixed 12-byte prefix then the 32-byte
// key, so the raw key can be read out without a full ASN.1 parser.
function pemToSpki(pem) {
  const body = pem.replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
  return b64ToBytes(body);
}

async function importEd25519(spki) {
  return crypto.subtle.importKey("spki", spki, { name: "Ed25519" }, false, ["verify"]);
}

function row(state, what, detail) {
  const marks = { ok: ["ok", "✓"], no: ["no", "✗"], warn: ["warn", "!"], dim: ["dim", "·"] };
  const [cls, ch] = marks[state] || marks.dim;
  const d = detail ? `<div class="detail">${detail}</div>` : "";
  return `<div class="row"><div class="mark ${cls}">${ch}</div><div class="what">${what}${d}</div></div>`;
}

async function verify(envelope, publicKeyPem) {
  const checks = [];
  let allOk = true;

  if (!envelope.payload || !envelope.payloadType || !Array.isArray(envelope.signatures)) {
    return { checks: [row("no", say("notDsse", "Not a DSSE envelope"),
      say("notDsseD", "Needs payload, payloadType and signatures."))], allOk: false, statement: null };
  }

  const payloadBytes = b64ToBytes(envelope.payload);
  const preauth = pae(envelope.payloadType, payloadBytes);
  const paeDigest = hex(await crypto.subtle.digest("SHA-256", preauth));
  checks.push(row("ok", say("pae", "Pre-authentication encoding recomputed"),
    `sha256 ${paeDigest}`));

  let statement = null;
  try {
    statement = JSON.parse(new TextDecoder().decode(payloadBytes));
    checks.push(row("ok", say("parses", "Payload parses as the declared type"), envelope.payloadType));
  } catch (e) {
    checks.push(row("no", say("notJsonPayload", "Payload is not valid JSON"), String(e)));
    allOk = false;
  }

  if (!publicKeyPem) {
    checks.push(row("warn", say("noKey", "No public key supplied, signature not checked"),
      say("noKeyD", "Paste the signer's PEM to complete the check. Without it this page has "
      + "confirmed the encoding only.")));
    allOk = false;
  } else if (envelope.signatures.length === 0) {
    checks.push(row("no", say("noSigs", "The envelope carries no signatures"),
      say("noSigsD", "There is nothing to check the key against, so nothing is verified.")));
    allOk = false;
  } else {
    for (const s of envelope.signatures) {
      try {
        const key = await importEd25519(pemToSpki(publicKeyPem));
        const ok = await crypto.subtle.verify("Ed25519", key, b64ToBytes(s.sig), preauth);
        checks.push(row(ok ? "ok" : "no",
          ok ? say("sigOk", "Signature verifies over the pre-authentication encoding")
             : say("sigBad", "Signature does NOT verify"),
          `keyid ${s.keyid || "(none)"}`));
        if (!ok) allOk = false;
      } catch (e) {
        if (e && e.name === "NotSupportedError") {
          checks.push(row("warn", say("noEd", "This browser has no Ed25519 in WebCrypto"),
            say("noEdD", "Chrome 137+, Safari 17+ or Firefox 129+ can complete the check.")));
        } else {
          checks.push(row("no", say("sigErr", "Signature could not be checked"), String(e)));
        }
        allOk = false;
      }
    }
  }
  return { checks, allOk, statement, paeDigest };
}

function renderStatement(st) {
  const p = (st && st.predicate) || {};
  const pairs = [];
  const put = (k, v) => { if (v !== undefined && v !== null && v !== "") pairs.push([k, v]); };
  put(say("f_decision", "decision"), p.decision);
  put(say("f_agent", "agent"), p.agent_id);
  put(say("f_principal", "principal"), p.principal);
  put(say("f_decided_at", "decided at"), p.decided_at);
  put(say("f_predicate_type", "predicate type"), st.predicateType);
  // The predicate sits under the in-toto namespace because Vaara proposed it
  // there. Saying so keeps the reader from taking the URL as in-toto's
  // endorsement of a predicate that is still an open proposal.
  if (st.predicateType === "https://in-toto.io/attestation/agent-decision/v0.1") {
    put(say("f_predicate_status", "predicate status"),
      say("predStatus", "proposed by Vaara on in-toto/attestation#554, not yet a registered in-toto predicate"));
  }
  if (Array.isArray(p.policy_evaluations)) {
    p.policy_evaluations.forEach(e =>
      put(`policy ${e.policy_id || ""}`, `${e.result || "?"}${e.rule ? "  (" + e.rule + ")" : ""}`));
  }
  if (Array.isArray(p.tool_calls)) {
    p.tool_calls.forEach(t =>
      put(`tool ${t.name || "?"}`, t.args_state === "present"
        ? `args ${t.args_hash || "(no hash)"}`
        : `args ${t.args_state || "unknown"}`));
  }
  if (Array.isArray(st.subject)) {
    st.subject.forEach(s => {
      const dg = s.digest && (s.digest.sha256 || Object.values(s.digest)[0]);
      put(`subject ${s.name || ""}`.trim(), dg || "(no digest)");
    });
  }
  $("fields").innerHTML = pairs.map(([k, v]) =>
    `<dt>${k}</dt><dd>${String(v)}</dd>`).join("");
  $("content").classList.remove("hide");
}

// Stating the boundary is part of the product. A verifier that only shows
// green invites the reader to believe things the maths never established.
function renderNonClaims(st) {
  const p = (st && st.predicate) || {};
  const items = [
    [say("nc1", "that the key belongs to the party you think it does"),
     say("nc1d", "the signature binds the payload to a key, not to an organisation")],
    [say("nc2", "that the statement is true"),
     say("nc2d", "a signed claim of a deny is evidence the agent recorded a deny, not that a deny occurred")],
    [say("nc3", "when this happened"),
     say("nc3d", "decided_at is asserted by the signer{at}. an external timestamp authority is what makes a time independent",
       { at: p.decided_at ? " (" + p.decided_at + ")" : "" })],
    [say("nc4", "that this is the whole history"),
     say("nc4d", "one receipt shows one decision. only a chain, and a head somebody else witnessed, "
     + "shows nothing was removed")],
  ];
  $("nc").innerHTML = items.map(([a, b]) => row("dim", a, b)).join("");
  $("nonclaims").classList.remove("hide");
}

async function run() {
  const raw = $("input").value.trim();
  if (!raw) return;
  let envelope, pem = null;

  // Accept either a bare envelope, or an object carrying both so a single
  // paste can complete the whole check.
  try {
    const parsed = JSON.parse(raw);
    envelope = parsed.envelope || parsed;
    pem = parsed.publicKey || parsed.public_key || null;
  } catch (e) {
    $("out").classList.remove("hide");
    $("verdict").innerHTML = `<span class="verdict no">${say("vNotJson", "NOT JSON")}</span>`;
    $("checks").innerHTML = row("no", say("notJson", "Input is not valid JSON"), String(e));
    return;
  }
  if (!pem) {
    const m = raw.match(/-----BEGIN PUBLIC KEY-----[\s\S]+?-----END PUBLIC KEY-----/);
    if (m) pem = m[0];
  }

  const { checks, allOk, statement } = await verify(envelope, pem);
  $("out").classList.remove("hide");
  $("verdict").innerHTML = allOk
    ? `<span class="verdict ok">${say("vOk", "SIGNATURE VERIFIES")}</span>`
    : `<span class="verdict warn">${say("vIncomplete", "INCOMPLETE")}</span>`;
  $("checks").innerHTML = checks.join("");
  if (statement) { renderStatement(statement); renderNonClaims(statement); }
}

$("go").addEventListener("click", run);

// ---- explorer ----------------------------------------------------------
// Queries the public log straight from this tab. Rekor serves
// access-control-allow-origin: *, so no Vaara server sits in the middle and
// there is nothing here that could log who looked something up.
const REKOR = "https://rekor.sigstore.dev";

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function rekorSearchByKey(pem) {
  const b64 = btoa(pem.trim());
  const res = await fetch(`${REKOR}/api/v1/index/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // format must be one of pgp/x509/minisign/ssh/tuf; a SubjectPublicKeyInfo
    // PEM is x509 here, and the content is that PEM base64-encoded again.
    body: JSON.stringify({ publicKey: { format: "x509", content: b64 } }),
  });
  if (!res.ok) throw new Error(`index/retrieve returned ${res.status}`);
  return res.json();
}

async function rekorSearchByHash(hexDigest) {
  const res = await fetch(`${REKOR}/api/v1/index/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hash: `sha256:${hexDigest}` }),
  });
  if (!res.ok) throw new Error(`index/retrieve returned ${res.status}`);
  return res.json();
}

// The log hands back the entry as base64 JSON. Decode the fields here, in one
// place, so the page can lay them out itself instead of sending the reader to a
// raw API response and asking them to read it. Both lookup paths and the latest
// listing share this, so every route shows the same panel.
function decodeEntry(uuid, e) {
  let digest = null, hashAlg = null, kind = null, apiVersion = null, keyPem = null;
  try {
    const decoded = JSON.parse(atob(e.body));
    kind = decoded.kind;
    apiVersion = decoded.apiVersion;
    const spec = decoded.spec || {};
    const hash = (spec.data && spec.data.hash) || {};
    digest = hash.value || null;
    hashAlg = hash.algorithm || null;
    const sig = spec.signature || {};
    if (sig.publicKey && sig.publicKey.content) keyPem = atob(sig.publicKey.content).trim();
  } catch (_) { /* an entry shape this page does not decode; fields stay null */ }
  const verification = e.verification || {};
  const proof = verification.inclusionProof || {};
  return {
    uuid, digest, hashAlg, kind, apiVersion, keyPem,
    logIndex: e.logIndex, integratedTime: e.integratedTime, logID: e.logID,
    treeSize: proof.treeSize, rootHash: proof.rootHash,
    proofPath: Array.isArray(proof.hashes) ? proof.hashes.length : null,
    hasCheckpoint: Boolean(proof.checkpoint),
    hasSet: Boolean(verification.signedEntryTimestamp),
  };
}

async function rekorEntry(uuid) {
  const res = await fetch(`${REKOR}/api/v1/log/entries/${uuid}`);
  if (!res.ok) throw new Error(`entry fetch returned ${res.status}`);
  const body = await res.json();
  return decodeEntry(uuid, body[uuid]);
}

function renderEntries(list, total) {
  if (!list.length) {
    $("rolls").innerHTML = row("dim", say("notInLog", "Not in the log"),
      say("notInLogD", "Either it was never published, or it was published under a different digest. "
      + "Absence is not evidence of anything."));
    return;
  }
  // Never show a partial set as if it were the whole set. A digest can carry
  // thousands of entries, and quietly rendering ten reads as "there are ten".
  const note = total > list.length
    ? row("warn", say("showing", "Showing {n} of {total} entries", { n: list.length, total }),
        say("showingD", "This digest appears many times in the log. The rest are not listed here."))
    : "";
  SHOWN = {};
  list.forEach(e => { SHOWN[e.uuid] = e; });
  $("rolls").innerHTML = note + list.map(entryCard).join("");
}

// The log's own entry view is a raw API response. Everything a reader would
// take from it is laid out here in the page's own shape instead, with the raw
// response still one click away because that is the authority, not this render.
function entryCard(e) {
  const pairs = [];
  const put = (k, v) => { if (v !== undefined && v !== null && v !== "") pairs.push([k, v]); };
  put(say("e_log_index", "log index"), e.logIndex);
  put(say("e_integrated", "integrated"), e.integratedTime
    ? new Date(e.integratedTime * 1000).toISOString() : null);
  put(say("e_entry_type", "entry type"), [e.kind, e.apiVersion].filter(Boolean).join(" "));
  put(say("e_data_digest", "data digest"), e.digest ? `${e.hashAlg || "sha256"}:${e.digest}` : null);
  put(say("e_log_id", "log id"), e.logID);
  put(say("e_tree_size_at_inclusion", "tree size at inclusion"), e.treeSize);
  put(say("e_root_hash", "root hash"), e.rootHash);
  put(say("e_inclusion_proof", "inclusion proof"), e.rootHash
    ? (e.proofPath === null ? say("proofNoPath", "present, path not returned")
                           : say("proofPath", "present, {n} hashes", { n: e.proofPath }))
    : say("notReturned", "not returned"));
  put(say("e_checkpoint", "checkpoint"), e.hasCheckpoint ? say("present", "present") : say("absent", "absent"));
  put(say("e_signed_entry_timestamp", "signed entry timestamp"), e.hasSet ? say("present", "present") : say("absent", "absent"));
  put(say("e_entry_uuid", "entry uuid"), e.uuid);

  const key = e.keyPem ? `
      <div class="detail" style="margin-top:.7rem">${say("keyOnEntry", "public key carried on this entry")}</div>
      <div class="keyblock">${esc(e.keyPem)}</div>
      <button type="button" class="linkish" data-search-key="${esc(e.uuid)}">${say("searchKey", "Search everything published under this key")}</button>` : "";

  return `
    <div class="entry">
      <div>${esc(e.digest || say("notHashed", "(not a hashedrekord)"))}</div>
      <div class="detail">
        ${say("e_log_index", "log index")} ${esc(e.logIndex)} ·
        ${say("e_integrated", "integrated")} ${e.integratedTime
          ? esc(new Date(e.integratedTime * 1000).toISOString()) : say("unknown", "(unknown)")}
      </div>
      <details>
        <summary>${say("details", "details")}</summary>
        <dl>${pairs.map(([k, v]) =>
          `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
        ${key}
        <div class="detail" style="margin-top:.7rem">
          ${say("entryNote", "An entry shows this digest was in the log at that index and time. It "
          + "says nothing about who put it there, and nothing about whether the "
          + "statement behind the digest is true.")}
        </div>
        <div class="foot">
          <a href="${REKOR}/api/v1/log/entries/${esc(e.uuid)}" rel="noopener">${say("raw", "Raw response from the log")}</a>
        </div>
      </details>
    </div>`;
}

// Entries currently on screen, so the key button can reach the PEM without
// round-tripping a multi-line value through an HTML attribute.
let SHOWN = {};

$("rolls").addEventListener("click", ev => {
  const btn = ev.target.closest("[data-search-key]");
  if (!btn) return;
  const entry = SHOWN[btn.getAttribute("data-search-key")];
  if (!entry || !entry.keyPem) return;
  $("q").value = entry.keyPem;
  lookup(entry.keyPem);
});

async function lookup(input) {
  const raw = (input || "").trim();

  // A pasted public key means "show me everything I published". Refuse a
  // private key outright: nobody should ever be typing one into a web page,
  // and a tool that quietly accepts it teaches a dangerous habit.
  if (/-----BEGIN [A-Z ]*PRIVATE KEY-----/.test(raw)) {
    $("rolls").innerHTML = row("no", say("privKey", "That is a private key. Do not paste it anywhere"),
      say("privKeyD", "Only the public half is needed here, and nothing on this page ever needs "
      + "a private key. Close this tab and treat that key as compromised."));
    $("q").value = "";
    return;
  }
  if (/-----BEGIN PUBLIC KEY-----/.test(raw)) {
    $("rolls").innerHTML = row("dim", say("asking", "Asking the public log…"));
    try {
      const uuids = (await rekorSearchByKey(raw)) || [];
      const entries = [];
      for (const u of uuids.slice(0, 20)) entries.push(await rekorEntry(u));
      entries.sort((a, b) => b.integratedTime - a.integratedTime);
      renderEntries(entries, uuids.length);
    } catch (e) {
      $("rolls").innerHTML = row("warn", say("unreach", "Could not reach the public log"), esc(e.message));
    }
    return;
  }

  const clean = raw.replace(/^sha256:/i, "").toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(clean)) {
    $("rolls").innerHTML = row("no", say("badQuery", "That is not a sha256 digest or a public key"),
      say("badQueryD", "Expecting 64 hex characters, or a PEM public key block."));
    return;
  }
  $("rolls").innerHTML = row("dim", say("asking", "Asking the public log…"));
  try {
    const uuids = (await rekorSearchByHash(clean)) || [];
    const entries = [];
    for (const u of uuids.slice(0, 10)) entries.push(await rekorEntry(u));
    renderEntries(entries, uuids.length);
  } catch (e) {
    $("rolls").innerHTML = row("warn", say("unreach", "Could not reach the public log"), esc(e.message)
      + say("unreachD", ". Signature verification above is unaffected: it never needs the network."));
  }
}

$("look").addEventListener("click", () => lookup($("q").value));
$("q").addEventListener("keydown", ev => { if (ev.key === "Enter") lookup($("q").value); });

async function logInfo() {
  const res = await fetch(`${REKOR}/api/v1/log`);
  if (!res.ok) throw new Error(`log returned ${res.status}`);
  return res.json();
}

// The tiles are the shape of the log before any detail: how big it is now and
// what root that size hashes to. Both come from the log, not from us.
async function fillTiles() {
  try {
    const info = await logInfo();
    $("t-size").textContent = Number(info.treeSize).toLocaleString(NUM);
    $("t-root").textContent = String(info.rootHash).slice(0, 16) + "…";
  } catch (e) {
    $("t-size").textContent = say("unreachable", "unreachable");
    $("t-root").textContent = "—";
  }
}

async function entryAtIndex(i) {
  const res = await fetch(`${REKOR}/api/v1/log/entries?logIndex=${i}`);
  if (!res.ok) throw new Error(`index ${i} returned ${res.status}`);
  const body = await res.json();
  const uuid = Object.keys(body)[0];
  return decodeEntry(uuid, body[uuid]);
}

// Latest N rather than an endless list. An explorer's job is to show that the
// thing is alive and moving, not to page through two billion rows.
const LATEST_N = 6;

$("latest").addEventListener("click", async () => {
  $("rolls").innerHTML = row("dim", say("asking", "Asking the public log…"));
  try {
    const info = await logInfo();
    const top = Number(info.treeSize) - 1;
    const out = [];
    for (let i = top; i > top - LATEST_N && i >= 0; i--) {
      try { out.push(await entryAtIndex(i)); } catch (_) { /* skip a gap */ }
    }
    renderEntries(out, out.length);
  } catch (e) {
    $("rolls").innerHTML = row("warn", say("unreach", "Could not reach the public log"), esc(e.message));
  }
});

fillTiles();

const drop = $("drop");
["dragover", "dragenter"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e =>
  drop.addEventListener(e, () => drop.classList.remove("over")));
drop.addEventListener("drop", async ev => {
  ev.preventDefault();
  const f = ev.dataTransfer.files[0];
  if (f) { $("input").value = await f.text(); run(); }
});

$("demo").addEventListener("click", () => {
  $("input").value = JSON.stringify(DEMO, null, 1);
  run();
});

const DEMO = {"envelope": {"payload": "eyJfdHlwZSI6Imh0dHBzOi8vaW4tdG90by5pby9TdGF0ZW1lbnQvdjEiLCJwcmVkaWNhdGUiOnsiYWdlbnRfaWQiOiJhZ2VudDphY21lL2NoZWNrb3V0LWJvdC8xIiwiZGVjaWRlZF9hdCI6IjIwMjYtMDYtMjNUMTI6MDA6MDVaIiwiZGVjaXNpb24iOiJkZW55IiwicG9saWN5X2V2YWx1YXRpb25zIjpbeyJwb2xpY3lfaWQiOiJwb2xpY3k6c3BlbmQtY2FwL2V1ci01MDAiLCJyZXN1bHQiOiJkZW55IiwicnVsZSI6ImFtb3VudC5sZS41MDAwMCJ9XSwicHJpbmNpcGFsIjoidGVuYW50OmFjbWUtZXUiLCJ0b29sX2NhbGxzIjpbeyJhcmdzX2Nhbm9uaWNhbGl6YXRpb24iOiJKQ1MiLCJhcmdzX2hhc2giOiJzaGEyNTY6OTFjMGMyMTVjY2UzZjVlN2M0OWQyZGRmMGE1NDBlYjIyNTU1NTRlYTNlN2E4ODUwNjdkMzkxNjhiM2FjODU2NCIsImFyZ3Nfc3RhdGUiOiJwcmVzZW50IiwibmFtZSI6InBheW1lbnRzLnRyYW5zZmVyIn0seyJhcmdzX3N0YXRlIjoiYXJnc19yZWRhY3RlZCIsIm5hbWUiOiJjdXN0b21lci5sb29rdXAifV19LCJwcmVkaWNhdGVUeXBlIjoiaHR0cHM6Ly9pbi10b3RvLmlvL2F0dGVzdGF0aW9uL2FnZW50LWRlY2lzaW9uL3YwLjEiLCJzdWJqZWN0IjpbeyJkaWdlc3QiOnsic2hhMjU2IjoiYTI3MmUzMGM4NzY4NzI4NWQ3YThkY2Q2YzVlMjMyNTRjZDVkYjJjNjgyMDI1NGRmNTQyYzI5MThlMDAxMTdiZCJ9LCJuYW1lIjoidG9vbDpwYXltZW50cy50cmFuc2ZlciJ9XX0=", "payloadType": "application/vnd.in-toto+json", "signatures": [{"keyid": "vaara-agent-decision-conformance-ed25519-k1", "sig": "mW5uAfYeWoUkc7Co6ehJ21y9vnoajFvmCGj1SkXJgZOgLGH6tfquZcoZL5Z+yMhip3zFWSA5s6Rd+oqL9me2AQ=="}]}, "publicKey": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAvHy8tWNjdfodgkNNRmck2SN39TuYBpXdSdJtDOEiBaU=\n-----END PUBLIC KEY-----"};
