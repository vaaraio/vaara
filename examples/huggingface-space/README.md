---
title: Vaara
emoji: 🛡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Runtime governance for AI agent tool calls
thumbnail: https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-wordmark-light.png
tags:
  - guardrails
  - llm-security
  - ai-agents
  - ai-governance
  - eu-ai-act
---

# Vaara on HuggingFace Spaces

Interactive demo of [Vaara](https://github.com/vaaraio/vaara), an open-source
runtime governance layer for AI agent tool calls.

Vaara is an open-source guardrail for AI agents. It runs at runtime, not at
config-time, and complements input filters, output validators, and red-team
harnesses by sitting at the point of action.

## What this shows

Each interception runs the real Vaara pipeline:

1. Classify the tool call against the action registry
2. Score it with a conformal risk interval (distribution-free coverage)
3. Decide allow, escalate, or deny
4. Write a hash-chained audit record aligned with EU AI Act Articles 12 and 14

The same call you can make from Python, the Anthropic SDK, LangChain, OpenAI,
or any MCP-capable agent host. The Space wraps the public `Pipeline` API.

## Run it locally

```bash
pip install vaara gradio
python app.py
```

Or install Vaara on its own and use it inside your agent loop:

```bash
pip install vaara
```

## Links

- Source: <https://github.com/vaaraio/vaara>
- PyPI: <https://pypi.org/project/vaara/>
- Site: <https://vaara.io>
- License: Apache-2.0
