# acp_checkout_v0 conformance vector

A recomputable `{statement, expected-verdict}` pair for an Agentic Commerce
Protocol (ACP) checkout session, for the governance binding discussed on
[agentic-commerce-protocol#231](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol/issues/231).
Conformance is anchored to the bytes, not to any one merchant or PSP.

## What it is

- `statement.json`, a real-shaped ACP `CheckoutSession` in its terminal
  `completed` status: the currency and totals that settled, the line items, the
  payment handler and PSP that took the money (`stripe`), and the `order` it
  produced with its confirmation number.
- `keys/` is absent on purpose. ACP objects are **not signed**: the protocol
  authenticates the API call at the transport (request signing), not the session
  record. There is no envelope and no signature to verify, which is the gap the
  mapping reports.
- `expected.json`, the verdict a conformant reader must reach.

## What passing means

Run the checker, which imports no Vaara code (standard library, `rfc8785` for
JCS):

    python tests/vectors/acp_checkout_v0/_check_independent.py

It recomputes, from the bytes alone:

1. `jcsSha256`, the sha256 of the JCS (RFC 8785) canonical statement bytes, a
   content commitment any reader reproduces from the document. It is a digest
   over the record, **not** a signature: ACP signs nothing here;
2. `status`, the terminal session status the document carries;
3. `normalized`, the SEP-2828 mapping (which evidence plane, which advisory
   fields lifted, what is still `missing`) reproduced from the shipped
   declarative profile `src/vaara/attestation/profiles/acp_checkout.json` with
   the checker's own code.

Step 3 is why the mapping is not self-confirming: it comes from the JSON spec
the product ships, reproduced by a second implementation, not from a Vaara
function.

## What the mapping says

An ACP checkout session is the commercial-outcome face of an agent purchase, the
closest foreign analog to a settled-transaction record. It sits on the `outcome`
plane, yet fills no SEP-2828 field on its own:

- nothing in the object is signed, so `alg`, `signature`, and `receiptAsserted`
  are absent: the session states an outcome it does not itself prove;
- there is no `backLink`, no `attestationDigest` pinning the attested request or
  the authorizing decision that a conformant receipt must answer, so the spend
  is reported without a link to what authorized it;
- `status` is a merchant-asserted state, lifted as advisory context
  (`advisory.status`, `advisory.orderStatus`), the commercial result a receipt
  would bind rather than assert.

This is the inverse of the `agent_decision_v0` vector, which carries the
authorizing decision but not the outcome. Neither is a receipt until the
decision and the outcome are bound under one signature.

## Regenerating

    python tests/vectors/acp_checkout_v0/_generate.py

and commit the result.
