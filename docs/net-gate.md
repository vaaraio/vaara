# Net gate: an agent's network access, governed where it leaves

Vaara's gate decides what an agent asks to do. An agent that opens a socket
itself never asks. `vaara net-gate` covers that half: it is a gateway the
agent's traffic must leave through, and at that gateway every connection is

- **attributed** to one agent, by that agent's own credential;
- **checked** against that agent's mandate (the destinations it may reach)
  and against your policy;
- **recorded** on the hash-chained trail, allowed or refused, with the bytes
  moved and why the connection ended;
- **revocable**: revoking the agent ends every connection it has open, at
  once, mid-transfer, and the trail says so.

An unknown or wrong credential gets no reply at all. The socket closes
without a byte, so whoever is probing learns nothing about what is behind
the gateway.

## Quick start

```bash
# Give an agent a credential and a mandate. The token is printed once;
# only its sha256 is kept.
vaara net-gate add-agent builder --allow api.github.com --allow "*.pypi.org"

# Run the gateway
vaara net-gate serve --listen 127.0.0.1:8791

# Start the agent with the gateway as its proxy
HTTPS_PROXY=http://builder:<token>@127.0.0.1:8791 \
HTTP_PROXY=http://builder:<token>@127.0.0.1:8791 \
  your-agent

# The brake: every open connection of `builder` ends now, on the record
vaara net-gate revoke builder
```

`vaara net-gate list` shows each agent's mandate and whether it is revoked.

## What happens to one connection

| Step | Check | When it fails |
|---|---|---|
| 1 | The proxy credential names a known agent and its token matches | The socket closes with no reply (`--challenge` answers 407 instead) |
| 2 | The agent is not revoked | 403, recorded as `net_gate:revoked` |
| 3 | The destination is in the agent's mandate | 403, recorded as `net_mandate:<agent>` |
| 4 | Your policy allows `net.connect` | 403. A hold for a human is refused too and closed on the record as refused, because a proxied connection cannot wait for a person |
| 5 | The name resolves to an address outside loopback, link-local, private ranges and cloud metadata (unless the agent has `--allow-private`; metadata never) | 403. The socket then dials the exact address that passed, so a DNS answer that changes in between cannot redirect it |

Each decision is one `net.connect` entry on the trail. Each connection that
opened gets a `net.close` entry with `bytes_up`, `bytes_down`, `seconds` and
`reason` (`closed`, `revoked`, or what went wrong).

Plain `http://` requests are forwarded in origin form with the proxy headers
removed and `Connection: close` set, one request per connection, so a second
request on a kept-alive connection cannot reach a host the first was not
checked for. HTTPS goes through `CONNECT`; the gateway does not look inside
TLS.

## Mandates

Entries use the same language as `vaara run`'s egress list:

- `example.com`: that host, ports 443 and 80
- `*.example.com`: any subdomain (not `example.com` itself), ports 443 and 80
- `example.com:8443`: that port only
- `*`: any public host. The limits then come from your policy and the
  address floor.

## Making the gateway the only way out

The gateway decides what reaches it. Making it the only route is the job of
the layer under the agent:

- **`vaara run` / a cage**: point the launch's proxy at the gateway; the OS
  layer already lets the agent's tree connect to its proxy port and nothing
  else.
- **A container**: a network policy or firewall that allows egress only to
  the gateway.
- **WireGuard**: route the agent host's traffic to the gateway's host, and on
  that host allow forwarded traffic only to the gateway.

`vaara net-gate wg-config` writes the client side. With no `--direct` it is a
full tunnel. Each `--direct` range is cut out of `AllowedIPs` and goes
straight out (split tunnelling); everything else goes to the gateway:

```bash
vaara net-gate wg-config --address 10.70.0.2/32 --gateway-key <gateway pubkey> \
    --endpoint gw.example:51820 --direct 192.168.0.0/16
```

The private key line is a placeholder: generate it with `wg genkey` on the
agent's host. The key never passes through Vaara.

## What it does not do

- It does not see traffic that never reaches it. That is the job of the cage,
  the container network or the tunnel.
- It does not decrypt TLS. It decides by host and port, and records by host
  and port.
- It has no approval channel. A connection your policy holds for a human is
  refused and recorded as refused.
