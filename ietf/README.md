# IETF Internet-Draft sources

This directory holds the Internet-Draft form of the Vaara receipt
specification. It exists to give `vaara.receipt/v1` a vendor-neutral, citable
identity on the IETF datatracker, backed by the shipped implementation and the
public conformance vectors under `tests/vectors/`.

## Documents

| File | Role |
|---|---|
| `draft-sirkkavaara-vaara-receipt-00.xml` | Source of truth (xml2rfc v3 vocabulary). |
| `draft-sirkkavaara-vaara-receipt-00.txt` | Rendered draft (committed for convenience). |

The draft mirrors the root [`SPEC.md`](../SPEC.md). `SPEC.md` stays the canonical
normative text for implementers; the I-D is the same content in the format the
datatracker ingests. When `SPEC.md` changes in a way that affects the normative
content, port the change into the XML and re-render.

## Toolchain

A single Python dependency renders the draft:

```sh
pip install xml2rfc        # 3.34.0 or newer
make                       # or: make text html
```

`make` writes `draft-sirkkavaara-vaara-receipt-00.txt` and `.html` next to the
XML. A clean run prints no warnings.

### Why XML and not kramdown-rfc

The plan considered authoring in kramdown-rfc Markdown. The XML v3 form was
chosen instead: it is the format the datatracker actually ingests, it renders
with one Python tool that matches this repository's language, and it needs no
Ruby toolchain. The cost is that the XML is more verbose than Markdown to edit
by hand, which is acceptable for a spec this size.

## Submitting

This is a `-00` skeleton for review, not yet submitted. Before any upload to the
datatracker, run `idnits` and confirm the author block, dates, and references.
Submission is an outbound action and is gated on Henri's go.
