# trail_rotation_v0 conformance vectors

A retention rotation archives the whole trail, then purges the records older
than the retention window. The first record that survives still links to the
last one purged. A signed `trail_rotated` record, written before the purge,
names that hash, so a verifier can tell a trail that begins at a rotation from
a trail with a hole at its start. See `src/vaara/audit/rotate.py`,
`src/vaara/audit/verify.py` and `docs/verifying-evidence.md`.

## The rule

A trail whose first record links to a hash outside the trail is accepted only
if a rotation record in the trail names that hash as `last_hash` and its
Ed25519 signature verifies; otherwise it is a chain break at the first record.

The signature covers the RFC 8785 (JCS) form of

```
{"type": "vaara.trail-rotation/v1", "lastHash", "purgedRecords", "archiveSha256",
 "archiveRecords", "archiveKind", "boundaryUtc", "rotatedAt", "signerFingerprint"}
```

taken from the record's `data` (`last_hash`, `purged_records`, `archive_sha256`,
`archive_records`, `archive_kind`, `boundary_utc`, `rotated_at`,
`signer_fingerprint`). The key is the one that signs the export, or a rotation
key named by the verifier. A record is `signed`, `unsigned` (no signature
block), `bad_signature`, or `malformed` (a field missing or of the wrong type,
a purged count below 1, or a signature block that is not the closed
`type`/`alg`/`signature` shape).

## Cases (`cases.json`)

Each case holds the records of a trail as exported (`records`), optionally the
name of a rotation key and an archive file, and the expected verdict: `verdict`,
`chain` (`ok` or `break`), `begins_at_seam`, the state of every rotation record,
and the archive check.

- `a_signed_seam_begins_the_trail`, `records_written_after_the_seam_still_chain`
  and `a_second_rotation_begins_at_its_own_record`: the accepted shapes.
- `a_trail_that_begins_at_its_first_record` and
  `a_seam_that_does_not_begin_the_trail`: no seam is needed, and a rotation
  record that is not at the start is listed, not used.
- `a_first_record_that_links_to_nothing_held`, `a_seam_that_names_another_hash`,
  `an_unsigned_seam`, `a_seam_signed_over_other_fields`,
  `a_seam_signed_by_another_key` and `a_malformed_seam`: each a chain break.
- `a_seam_signed_by_another_key_that_is_named`: the same record, accepted when
  the verifier names the key that signed it.
- `the_archive_is_the_one_the_record_names` and `the_archive_is_a_different_file`:
  the archive's SHA-256 against the digest in the record. The archive here is
  opaque bytes standing in for a sealed (encrypted) file, so only the digest is
  checked.

`signers` holds fixed TEST keys. A deployment's key is its own and is never
published.

## Honest limits

The vectors cover the link rule, the signature and the archive digest. Reading
a zip archive (that it verifies, holds the stated number of records and holds
the record the trail links to) is covered by `tests/test_trail_rotation_seam.py`.
The seam is evidence that an archive was made and what it held then. It does not
show the archive is still stored, and whoever holds the signing key can sign a
record for records they dropped, so the key should be pinned out of band.

## Reproduce

Independent checker (standard library plus `rfc8785` and `cryptography`, no
Vaara import):

```
python tests/vectors/trail_rotation_v0/_check_independent.py
```

Exit code 0 means every case reproduced. Regenerate `cases.json` (stable: the
keys, the clock and the record ids are fixed) with:

```
python tests/vectors/trail_rotation_v0/_generate.py
```
