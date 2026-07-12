# Running more than one proxy: the multi-replica pattern

Vaara's audit trail is a hash chain: every record binds to its predecessor, so a
chain has exactly one writer at a time. That is a feature, not a limitation, but
it means "three proxy replicas writing one trail" is not a thing Vaara does, and
this page tells you what to do instead.

## The unit of trust is one chain in one process

One `SQLiteAuditBackend` in one process owns one chain (or one chain per tenant,
when tenant IDs are in play). Within that process, writes from any thread are
serialized and safe. Everything on this page follows from that.

Do not point two processes at the same SQLite file. The write lock that protects
the chain is in-process; a second process interleaving records will produce a
chain neither process can verify. Nothing stops you from doing this today except
this paragraph, so treat it as load-bearing.

## Supported today

**One replica, many tenants.** A single proxy process fans out to multiple
upstream MCP servers (`--upstream name=cmd`, repeatable) and separates tenants
by header on the HTTP transport. Records carry `tenant_id`, purge and export
are tenant-scoped, and one process handles a fleet of agents as long as one
process can carry the traffic. This is the simplest deployment that gives you
a single DB to report from, and it is the one to exhaust first.

**Many replicas, one DB each.** When you scale past one process (multiple
hosts, one proxy per agent runtime, blast-radius isolation), give every replica
its own `--db` path and its own signing key or a shared operator key. Each
replica's trail is independently complete and independently verifiable: chain
verification, `vaara compliance report`, `vaara trail shadow-report`, and
`vaara trail rotate` all operate per DB and need nothing from the other
replicas.

Name the DBs so provenance is obvious in the archive:

```
/var/lib/vaara/audit-{host}-{service}.db
```

and set `--agent-id` per replica so records say which fleet slice produced them.

## Aggregating evidence across replicas

An auditor asking "what did your agents do" gets N signed archives, not one.
That is normal, and it mirrors how any distributed system hands over logs. The
workflow:

1. Rotate each replica on the same schedule:

   ```bash
   vaara trail rotate --db /var/lib/vaara/audit-host1-github.db \
     --out /archive/2026-07/host1-github.zip \
     --key /etc/vaara/signer.pem --retention-days 90 --all-tenants
   ```

   Rotate exports the whole trail, re-verifies the archive from its own bytes,
   and only then purges. A rotation whose archive fails verification purges
   nothing.

2. Keep an operator-level index of the archives: path, replica identity, and
   the SHA-256 of each zip. A plain manifest file, updated on each rotation and
   stored with the archives, is enough. Sign it with the same operator key if
   you want the index itself to be evidence.

3. Verification on the receiving side is per archive: `vaara trail verify
   --zip host1-github.zip` for each entry in the index, plus a check that the
   index covers every replica you claim to run.

What the per-replica model does not give you is a cryptographic proof that the
set of replicas is complete. If you run three proxies and hand over two
archives, each archive still verifies. Completeness across replicas is an
inventory statement by the operator, made checkable by the signed index, the
same way it works for any fleet of log-producing systems. State it in those
terms to an auditor and nobody is surprised.

## What this costs and when to revisit

The per-replica pattern trades a single query surface for horizontal scale.
Reports run per DB; a fleet-wide count is a loop over archives. For most
pilots, one process per upstream service is plenty, and the loop is a
five-line script.

Revisit when either of these stops being true:

- one process can carry the write load of one chain (the audit write is
  synchronous on the tool-call path; measured at roughly 0.2 ms per decision
  record on local disk, WAL mode, single writer), or
- your auditor requires one continuous chain across the fleet rather than N
  verifiable chains plus an index.

The primitives for a stronger roll-up already exist in the codebase (the RFC
6962 transparency log that backs inclusion proofs and COSE receipts), and a
fleet-level log anchoring each replica's archive hashes is the natural next
step. It is not shipped today, and this page will change when it is.
