# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Container image for `vaara proxy`, the model-endpoint proxy that fronts an
# OpenAI-compatible or ollama server and records every tool call the model
# requests into a hash-chained audit trail.
#
# The base is SUSE's own SLE Base Container Image. BCI needs no subscription
# to pull or redistribute, and building on it means the certified artifact and
# the platform it is certified against share a userland.
#
# Build:
#   docker build -t ghcr.io/vaaraio/vaara:dev .
# Run:
#   docker run -p 8788:8788 -v vaara-trail:/var/lib/vaara ghcr.io/vaaraio/vaara:dev

FROM registry.suse.com/bci/python:3.12 AS build

WORKDIR /src
COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY src/ ./src/

# --no-cache-dir everywhere: the layer is thrown away, and a pip cache in an
# intermediate stage still costs build time and disk on the runner.
RUN python3 -m pip install --no-cache-dir build \
    && python3 -m build --wheel --outdir /dist


FROM registry.suse.com/bci/python:3.12

ARG VERSION=0.0.0
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="Vaara" \
      org.opencontainers.image.description="Governance and receipt layer for AI agents: policy-gated tool calls and a hash-chained, independently verifiable audit trail" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/vaaraio/vaara" \
      org.opencontainers.image.url="https://vaara.io" \
      org.opencontainers.image.vendor="Vaara" \
      org.opencontainers.image.base.name="registry.suse.com/bci/python:3.12"

# proxy: fastapi, uvicorn, httpx (the listener and the upstream client).
# attestation: cbor2, cryptography, rfc8785 (signed attestation and receipt
# pairs). Without the second extra --signing-key fails at import time, which
# is the one flag an evidence deployment is most likely to set.
COPY --from=build /dist/*.whl /tmp/
RUN python3 -m pip install --no-cache-dir "$(ls /tmp/vaara-*.whl)[proxy,attestation]" \
    && rm -f /tmp/*.whl

# Fixed UID so a PersistentVolume written by one release stays writable by the
# next. The chart sets the same number in fsGroup; changing it here without
# changing it there leaves an existing trail unwritable and the pod crash-looping.
RUN groupadd --gid 10001 vaara \
    && useradd --uid 10001 --gid 10001 --home-dir /var/lib/vaara \
       --no-create-home --shell /sbin/nologin vaara \
    && mkdir -p /var/lib/vaara /var/lib/vaara/receipts \
    && chown -R 10001:10001 /var/lib/vaara

USER 10001:10001
WORKDIR /var/lib/vaara

# The audit trail is a hash chain with exactly one writer, so this path must
# be a volume owned by one pod. See docs/multi-replica-deployment.md.
VOLUME ["/var/lib/vaara"]

EXPOSE 8788

ENTRYPOINT ["vaara"]
CMD ["proxy", \
     "--listen", "0.0.0.0:8788", \
     "--upstream", "http://ollama:11434", \
     "--trail", "/var/lib/vaara/audit.db"]
