# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VaaraCallbackHandler must cover LangChain's whole event surface.

The handler is duck-typed so langchain stays an optional dependency,
which means nothing forces it to keep up when LangChain adds an event.
LangChain dispatches with ``getattr(handler, event_name)`` and re-raises
on error when ``raise_error`` is set, and Vaara sets it, because a
governance handler that silently swallows failures is worse than one
that stops the agent.

The consequence is that a missing method is not a degraded feature, it
is an AttributeError that kills the chain. That already happened once:
``on_stream_event`` arrived in langchain-core 1.x and every streaming
chain running Vaara crashed.

These tests skip when langchain-core is absent, so they cost nothing in
the default matrix, and fail loudly wherever it is installed.
"""

from __future__ import annotations

import inspect

import pytest

from vaara.integrations.langchain import VaaraCallbackHandler

langchain_callbacks = pytest.importorskip(
    "langchain_core.callbacks", reason="pip install langchain-core",
)
BaseCallbackHandler = langchain_callbacks.BaseCallbackHandler


def _public(obj) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


def test_handler_implements_every_base_callback_event():
    missing = sorted(_public(BaseCallbackHandler) - _public(VaaraCallbackHandler))
    assert not missing, (
        "VaaraCallbackHandler is missing "
        f"{missing}. LangChain re-raises when raise_error is set, so each "
        "of these is an AttributeError that stops a deployer's agent. Add "
        "a no-op to the protocol-compliance block in "
        "src/vaara/integrations/langchain.py."
    )


def test_raise_error_is_on_which_is_why_the_above_matters():
    handler = VaaraCallbackHandler.__new__(VaaraCallbackHandler)
    assert handler.raise_error is True


@pytest.mark.parametrize("event", ["on_stream_event", "on_custom_event", "on_retry"])
def test_dispatching_an_unhandled_event_does_not_raise(event):
    handle_event = langchain_callbacks.manager.handle_event
    handler = VaaraCallbackHandler.__new__(VaaraCallbackHandler)
    # Would raise AttributeError before on_stream_event existed.
    handle_event([handler], event, None, {"event": "on_chain_stream"})


@pytest.mark.parametrize("event", ["on_tool_start", "on_tool_end", "on_tool_error"])
def test_governed_event_signatures_match_langchain(event):
    """The three events Vaara actually acts on must be call-compatible."""
    theirs = inspect.signature(getattr(BaseCallbackHandler, event)).parameters
    ours = inspect.signature(getattr(VaaraCallbackHandler, event)).parameters

    for name, parameter in theirs.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        assert name in ours, f"{event} does not accept {name!r}"
        assert ours[name].kind is parameter.kind, (
            f"{event}: {name!r} is {ours[name].kind}, LangChain passes it as "
            f"{parameter.kind}"
        )
