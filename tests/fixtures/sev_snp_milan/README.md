# Real AMD SEV-SNP report (Milan)

- `report.bin`: an attestation report from a Milan SEV-SNP guest, and
  `vcek.der`: the VCEK AMD's KDS issued for that chip and TCB. Both from
  google/go-sev-guest, `verify/testdata/attestation.bin` and
  `verify/testdata/vcek.testcer`, Copyright 2022 Google LLC, Apache License 2.0.
- `ask.pem`, `ark.pem`: AMD's Milan chain, and `genoa_ask.pem`,
  `genoa_ark.pem`: AMD's Genoa chain, as served by
  `https://kdsintf.amd.com/vcek/v1/{Milan,Genoa}/cert_chain` (read 2026-09-28).
