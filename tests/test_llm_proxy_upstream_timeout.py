"""The llm-proxy upstream client must not run on httpx's default timeout.

2026-09-14: the proxy built its client as `httpx.AsyncClient(base_url=...,
http2=True)` with no timeout, so it inherited httpx's default of 5 seconds on
every phase, including each read of a buffered response. `_handle_chat` uses
`client.post`, not `client.stream`, so a generation that pauses longer than 5
seconds mid-response raised `ReadTimeout` and the caller got a 502. Long
generations pause for longer than that routinely. Observed live as an 11-in-83
failure rate against api.anthropic.com, intermittent and self-recovering, with
an empty reason string because `ReadTimeout` stringifies to nothing.

The sibling `_infer_proxy_app` already passed `httpx.Timeout(None)`. Only this
one was missed, which is exactly the kind of drift a test should hold shut.

This pins the property rather than a spelling: every AsyncClient this module
constructs must set a timeout explicitly, so a future branch that adds another
client cannot silently inherit the 5 second default either.
"""
import inspect
import re

import httpx

from vaara.integrations import _llm_proxy_app


def _async_client_calls():
    """Every `httpx.AsyncClient(...)` construction in the module source.

    Read from source because the clients are built inside `build_app` and
    closed over by the handlers, so there is no live object to introspect
    without standing up the whole app and a pipeline behind it.
    """
    src = inspect.getsource(_llm_proxy_app)
    return re.findall(r"httpx\.AsyncClient\((.*?)\)", src, re.DOTALL)


def test_the_module_builds_at_least_one_upstream_client():
    calls = _async_client_calls()
    assert calls, "no httpx.AsyncClient construction found; test is stale"


def test_every_upstream_client_sets_an_explicit_timeout():
    missing = [c.strip() for c in _async_client_calls() if "timeout" not in c]
    assert not missing, (
        "httpx.AsyncClient built without an explicit timeout, so it inherits "
        f"the 5 second default and long generations 502: {missing}"
    )


def test_httpx_default_is_still_the_five_seconds_this_guards_against():
    """If httpx ever changes its default, this test's premise needs revisiting.

    Not a guard on our code. It fails loudly the day the assumption behind the
    fix stops holding, rather than leaving a stale comment behind.
    """
    assert httpx.AsyncClient().timeout == httpx.Timeout(5.0)
