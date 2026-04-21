# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/vaaraio/vaara/security/advisories/new)
feature. **Do not open a public issue for anything that could be exploited.**

For communication outside GitHub, reach the maintainers at
`security@vaara.io`. Use PGP if you prefer end-to-end-encrypted email; the
current public key is published at
<https://github.com/vaaraio/vaara/blob/main/docs/signing-keys.md>.

Please include:
- A description of the vulnerability and potential impact
- Steps to reproduce (proof-of-concept if possible)
- The affected versions
- Any suggested mitigations

## Response Timeline

- **Acknowledgement**: within 3 business days
- **Initial assessment**: within 10 business days
- **Fix or mitigation timeline**: communicated after initial assessment

Coordinated disclosure is welcome. We will credit reporters in release notes
unless anonymity is requested.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.4.x   | Yes       |
| < 0.4   | No        |

Only the latest minor release receives security fixes. Older minor lines may
receive patches at our discretion for severe issues.

## Scope

**In scope:**
- The `vaara` Python package and its public APIs
- Audit trail integrity and signed-export verification
- Framework integrations shipped in this repository
- Supply-chain integrity of published PyPI artefacts

**Out of scope:**
- Third-party frameworks (LangChain, CrewAI, etc.): report those upstream
- Issues in consumer applications that use Vaara
- Vulnerabilities in dependencies already reported upstream

## Disclosure Process

1. Reporter submits via the channels above.
2. Maintainer acknowledges within 3 business days.
3. Maintainer investigates and confirms.
4. Fix developed in a private branch.
5. CVE requested where warranted.
6. Patch released and advisory published.
7. Reporter credited in release notes and GitHub Security Advisory.

## Security Considerations for Users

- Always pin vaara to a specific version in production.
- Verify PyPI artefacts against published checksums and (future) Sigstore
  attestations.
- When deploying signed audit exports, protect the signing private key using
  OS-level key management (hardware-backed keystore or HSM recommended).

## References

- OWASP Top 10: <https://owasp.org/www-project-top-ten/>
- EU AI Act Article 15 (accuracy, robustness, cybersecurity) informs the
  threat model for runtime governance components.
