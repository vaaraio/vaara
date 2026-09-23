// Runs verify() from webpage/verify.html under Node, on envelopes built here,
// and prints one JSON line per case: {"case": name, "allOk": bool}.
import { readFileSync } from "node:fs";
import { generateKeyPairSync, sign } from "node:crypto";

const html = readFileSync(process.argv[2], "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const src = scripts.find(s => s.includes("async function verify("));
const body = src.slice(0, src.indexOf("function renderStatement"));
const { verify } = new Function("document", body + "\nreturn { verify };")(
  { getElementById: () => null });

const { publicKey, privateKey } = generateKeyPairSync("ed25519");
const pem = publicKey.export({ type: "spki", format: "pem" });
const other = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" });

function envelope(type, payload, signers) {
  const pae = Buffer.concat([
    Buffer.from(`DSSEv1 ${Buffer.byteLength(type)} ${type} ${payload.length} `), payload]);
  return {
    payload: payload.toString("base64"),
    payloadType: type,
    signatures: signers.map(k => ({ sig: sign(null, pae, k).toString("base64") })),
  };
}

const payload = Buffer.from(JSON.stringify({ hello: "world" }));
const type = "application/vnd.in-toto+json";
const good = envelope(type, payload, [privateKey]);
const cases = {
  good: [good, pem],
  wrong_key: [good, other],
  forged_payload: [{ ...good, payload: Buffer.from('{"hello":"w0rld"}').toString("base64") }, pem],
  no_signatures: [{ ...good, signatures: [] }, pem],
  no_key: [good, null],
  non_ascii_type: [envelope("application/vnd.vaara+json; x=ä", payload, [privateKey]), pem],
};
for (const [name, [env, key]] of Object.entries(cases)) {
  const r = await verify(env, key);
  console.log(JSON.stringify({ case: name, allOk: r.allOk }));
}
