# Privacy Policy

**Last updated: 2026-08-10**

Vaara is an open-source project. This page describes how the Vaara software, website, and email contact handle data.

## Vaara software (CLI, plugin, MCP server)

The Vaara software does not send telemetry. It does not phone home, beacon, or transmit usage data to any server. The audit database is a local SQLite file on your machine; its contents do not leave your system unless you choose to share them.

If you self-host an MCP proxy with Vaara, network behavior is under your control. Vaara opens no connection unless you ask it to, and the complete list of times it does is:

- **The optional ML classifier.** `pip install 'vaara[ml]'` downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face the first time a score needs an embedding, pinned to a fixed revision. It is cached after that and everything later is offline. The classifier is opt-in and the base install never reaches for it. Pre-fetch the model if you need a host that never has network access.
- **"Check for updates" in `vaara menu`.** Fetches the published version from pypi.org, and only when you pick that item from the menu. The request carries nothing about your installation; the comparison against your installed version happens locally.
- **Time anchoring.** Contacts the RFC 3161 or eIDAS timestamp authority you configure, at the address you give it. There is no default and no fallback authority.

Nothing on that list runs on its own schedule, in the background, or as a side effect of recording evidence. A base install writing to the trail opens no socket at all.

## Website (vaara.io)

The vaara.io website is hosted on a single VPS. The HTTP server retains standard access logs (IP address, user-agent, requested URL, timestamp) for operational and security purposes. Logs are retained for up to 30 days and are not shared with third parties.

No analytics, trackers, or cookies are used on vaara.io.

## Email (hello@vaara.io)

When you contact hello@vaara.io, the contents of your message and your email address are processed solely to reply to you. Messages are retained as long as needed for that purpose and then deleted.

## Sharing

Vaara does not sell, rent, or share personal data with third parties.

## Your rights under GDPR

If you are in the EU/EEA, you have rights of access, rectification, erasure, restriction, objection, and data portability with respect to your personal data. To exercise these rights, contact hello@vaara.io.

## Controller

Vaara is maintained by Henri Sirkkavaara in Finland. Contact: hello@vaara.io.

## Updates

This page will be updated if data handling practices change. The "Last updated" date above will reflect any changes.
