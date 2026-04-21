# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/vaaraio/vaara/security/advisories/new)
feature. Do **not** open a public issue for anything that could be exploited.

We aim to acknowledge reports within 3 business days and provide an initial
assessment within 10 business days. Coordinated disclosure is welcome, and we will
credit reporters in release notes unless anonymity is requested.

## Supported versions

Only the latest minor release receives security fixes. Older minor lines may
receive patches at our discretion for severe issues.

## Scope

In scope:
- The `vaara` Python package and its public APIs
- Audit trail integrity and signed-export verification
- Framework integrations shipped in this repository

Out of scope:
- Third-party frameworks (LangChain, CrewAI, etc.): report those upstream
- Issues in consumer applications that use Vaara
