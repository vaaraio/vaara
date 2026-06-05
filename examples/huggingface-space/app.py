"""Vaara runtime governance demo for HuggingFace Spaces.

Pick a tool call the way an agent would propose it. Vaara classifies the
action, emits a conformal risk interval, decides allow/escalate/deny, and
writes a hash-chained audit record aligned with EU AI Act Articles 12
(record-keeping) and 14 (human oversight).

Repo: https://github.com/vaaraio/vaara
PyPI: https://pypi.org/project/vaara/
Site: https://vaara.io
"""
from __future__ import annotations

import json

import gradio as gr

from vaara import Pipeline

pipeline = Pipeline()

PRESETS: dict[str, dict] = {
    "Transfer funds": {
        "tool_name": "transfer_funds",
        "parameters": {
            "from_account": "1234",
            "to_account": "9876",
            "amount_eur": 10000,
        },
    },
    "Delete user": {
        "tool_name": "delete_user",
        "parameters": {"user_id": "u-42"},
    },
    "Drop database table": {
        "tool_name": "drop_table",
        "parameters": {"table": "customers"},
    },
    "Send public post": {
        "tool_name": "post_public",
        "parameters": {"channel": "twitter", "text": "Hello world"},
    },
    "Read public doc": {
        "tool_name": "fetch_url",
        "parameters": {"url": "https://example.com"},
    },
}

DECISION_LABEL = {
    "allow": "ALLOW",
    "escalate": "ESCALATE",
    "deny": "DENY",
}


def intercept(agent_id: str, tool_name: str, parameters_json: str):
    try:
        parameters = json.loads(parameters_json) if parameters_json.strip() else {}
    except json.JSONDecodeError as exc:
        err = f"Invalid JSON in parameters: {exc}"
        return err, "", "", "", ""

    result = pipeline.intercept(
        agent_id=agent_id or "demo-agent",
        tool_name=tool_name,
        parameters=parameters,
    )

    badge = DECISION_LABEL.get(result.decision, result.decision.upper())
    risk_line = (
        f"Point estimate: {result.risk_score:.3f}   "
        f"Conformal interval: [{result.risk_interval[0]:.3f}, {result.risk_interval[1]:.3f}]"
    )
    action_line = f"{result.action_type.name} (action_id={result.action_id})"

    full_json = json.dumps(
        {
            "allowed": result.allowed,
            "decision": result.decision,
            "action_id": result.action_id,
            "risk_score": result.risk_score,
            "risk_interval": list(result.risk_interval),
            "reason": result.reason,
            "signals": result.signals,
            "action_type": result.action_type.name,
            "evaluation_ms": result.evaluation_ms,
        },
        indent=2,
    )

    return badge, risk_line, action_line, result.reason, full_json


def apply_preset(name: str):
    preset = PRESETS[name]
    return preset["tool_name"], json.dumps(preset["parameters"], indent=2)


WORDMARK_HTML = """
<div style="display: flex; justify-content: center; align-items: center; padding: 1.5rem 0; width: 100%;">
  <img src="https://raw.githubusercontent.com/vaaraio/vaara/main/docs/vaara-wordmark-dark.png" alt="Vaara" style="max-width: 500px; width: 80%; display: block; margin: 0 auto;">
</div>
"""

INTRO = """
Runtime governance for AI agent tool calls. Each interception classifies the
action, emits a conformal risk interval, decides allow/escalate/deny, and
writes a hash-chained audit record aligned with EU AI Act Articles 12 and 14.

Pick a preset or write your own tool call. The decision happens in a few
milliseconds against the same pipeline you would run in production.

[GitHub](https://github.com/vaaraio/vaara) ·
[PyPI](https://pypi.org/project/vaara/) ·
[vaara.io](https://vaara.io)
"""

VAARA_THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#1c1f25",
    body_background_fill_dark="#1c1f25",
    body_text_color="#eaeaea",
    body_text_color_dark="#eaeaea",
    body_text_color_subdued="#9a9a9a",
    body_text_color_subdued_dark="#9a9a9a",
    background_fill_primary="#1c1f25",
    background_fill_primary_dark="#1c1f25",
    background_fill_secondary="#262931",
    background_fill_secondary_dark="#262931",
    block_background_fill="#262931",
    block_background_fill_dark="#262931",
    block_label_background_fill="#262931",
    block_label_background_fill_dark="#262931",
    block_label_text_color="#eaeaea",
    block_label_text_color_dark="#eaeaea",
    block_title_text_color="#eaeaea",
    block_title_text_color_dark="#eaeaea",
    border_color_primary="#34383f",
    border_color_primary_dark="#34383f",
    input_background_fill="#1f242c",
    input_background_fill_dark="#1f242c",
    input_border_color="#34383f",
    input_border_color_dark="#34383f",
    button_primary_background_fill="#8fb8d6",
    button_primary_background_fill_dark="#8fb8d6",
    button_primary_background_fill_hover="#6f9cbe",
    button_primary_background_fill_hover_dark="#6f9cbe",
    button_primary_text_color="#1c1f25",
    button_primary_text_color_dark="#1c1f25",
)

VAARA_CSS = """
html, body, gradio-app, .gradio-container, .main, .app, footer {
    background-color: #1c1f25 !important;
    color: #eaeaea !important;
}
.gradio-container {
    background: #1c1f25 !important;
}
.prose, .prose p, .prose li, .markdown, .markdown p, .markdown li, .markdown a {
    color: #eaeaea !important;
}
.prose a, .markdown a {
    color: #8fb8d6 !important;
}
.gradio-row {
    gap: 1rem !important;
}
.gradio-column {
    gap: 0.75rem !important;
}
"""

with gr.Blocks(title="Vaara", theme=VAARA_THEME, css=VAARA_CSS) as demo:
    gr.HTML(WORDMARK_HTML)
    gr.Markdown(INTRO)

    with gr.Row(equal_height=True):
        with gr.Column():
            preset = gr.Dropdown(
                choices=list(PRESETS.keys()),
                label="Preset",
                value="Transfer funds",
            )
            agent_id = gr.Textbox(label="Agent ID", value="demo-agent")
            tool_name = gr.Textbox(
                label="Tool name",
                value=PRESETS["Transfer funds"]["tool_name"],
            )
        with gr.Column():
            decision_out = gr.Textbox(label="Decision", interactive=False)
            risk_out = gr.Textbox(label="Risk", interactive=False)
            action_out = gr.Textbox(label="Action type", interactive=False)
            reason_out = gr.Textbox(label="Reason", interactive=False, lines=2)

    with gr.Row(equal_height=True):
        parameters_json = gr.Code(
            label="Parameters (JSON)",
            value=json.dumps(PRESETS["Transfer funds"]["parameters"], indent=2),
            language="json",
            lines=14,
        )
        json_out = gr.Code(
            label="Full InterceptionResult",
            language="json",
            lines=14,
        )

    run_btn = gr.Button("Intercept", variant="primary")

    preset.change(apply_preset, inputs=[preset], outputs=[tool_name, parameters_json])
    run_btn.click(
        intercept,
        inputs=[agent_id, tool_name, parameters_json],
        outputs=[decision_out, risk_out, action_out, reason_out, json_out],
    )

if __name__ == "__main__":
    demo.launch()
