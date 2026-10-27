# Receipt -15: an outcome status for a call the issuer did not observe

Table 4 of -14 gives `outcomeDerived.status` three values and says a consumer
MUST reject any other: `executed` (the action ran), `refused` (it did not run),
`errored` (it was attempted and failed).

An issuer that sat between a client and a server can be in a fourth position:
it sent the request, no reply was read (timeout, cut connection, malformed or
mismatched reply), and it cannot say whether the action ran. Writing
`errored` there says "attempted and failed" about an action that may have
completed. Writing `executed` says it ran. Neither is the issuer's
observation.

## Proposed value

`unobserved`: the request was sent and no outcome was observed. The issuer
cannot say whether the action ran. A consumer MUST NOT read it as `executed`,
and MUST NOT read it as `errored`. `resultCommitment` is absent. A later
receipt for the same `backLink` may settle it.

## The three states the MCP proxy keeps on its trail

| Trail status | What the proxy knows | Receipt status today | With the new value |
|---|---|---|---|
| `not_sent` | the request provably never left (closed client, egress floor, connection refused, name not resolved) | `errored` | `refused` as Table 4 words it ("it did not run"), or `errored` if `refused` is kept for a decision |
| `unobserved` | the request may have reached the server, no reply was read | `errored` | `unobserved` |
| `failed` | the server answered with an error status, or a JSON-RPC error or tool `isError` reply | `errored` | `errored` |

`not_sent` needs a wording choice in -15: Table 4 defines `refused` by the
action not running, but the decision receipts use `refused` for a refusal by
policy. Either say that `refused` covers both, or leave `not_sent` as
`errored` and accept that the pair reads as an attempt.

## Where this stands in code

The receipt keeps the -14 set. The trail record (`outcome_recorded`) carries
the three trail statuses above in its `description`, so the distinction is
already on the chain for anyone who reads it. Emitting `unobserved` or a
`refused` for `not_sent` on the receipt is a wire change and waits for -15.
