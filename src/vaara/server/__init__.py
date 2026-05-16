"""Vaara HTTP API reference server.

Exposes the Vaara scorer and audit trail as a network service following the
contract in `docs/openapi.yaml`. The spec is authoritative; this module is a
reference implementation suitable for local development, integration testing,
and modest production loads. Production deployments with sustained traffic
should provide their own implementation against the same spec.

Install: ``pip install vaara[server]``.

Run::

    vaara serve --host 0.0.0.0 --port 8000

Or programmatically::

    from vaara.server import create_app
    app = create_app()
"""

from vaara.server.app import create_app

__all__ = ["create_app"]
