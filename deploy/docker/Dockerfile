FROM registry.suse.com/bci/python:3.12

LABEL org.opencontainers.image.title="Vaara MCP Governance Server"
LABEL org.opencontainers.image.description="Runtime evidence layer for AI agents — policy-gated tool calls, tamper-evident audit trails, EU AI Act Article 12 compliant"
LABEL org.opencontainers.image.source="https://github.com/vaaraio/vaara"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

RUN pip install --no-cache-dir "vaara[server,attestation,timeanchor]"

# Non-root user
RUN useradd -r -u 1000 -g users vaara
USER vaara

EXPOSE 8000

ENV VAARA_HOST=0.0.0.0
ENV VAARA_PORT=8000

CMD ["vaara-mcp-server", "--host", "0.0.0.0", "--port", "8000"]
