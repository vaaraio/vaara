"""tools/call arguments are checked against the tool's advertised inputSchema.

OVERT TOOL-2.2 asks for parameter validation before execution and TOOL-2.3
for a rejection receipt that names the parameter violation. The proxy keeps
each tool's inputSchema from tools/list, refuses a call whose arguments fall
outside it before the call is scored or forwarded, and writes an
``action_blocked`` record with ``violation_type: parameter_schema`` and the
failing parameter in the reason.
"""

import json
from unittest.mock import MagicMock

from vaara.integrations._mcp_input_schema import check_arguments, unchecked_keywords

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1, "pattern": "^/srv/"},
        "count": {"type": "integer", "minimum": 1, "maximum": 10},
        "mode": {"enum": ["read", "tail"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
    "required": ["path"],
    "additionalProperties": False,
}


# The checker


def test_conforming_arguments_have_no_violations():
    args = {"path": "/srv/a", "count": 3, "mode": "tail", "tags": ["x"]}
    assert check_arguments(args, SCHEMA) == []


def test_each_violation_names_its_parameter():
    args = {"count": 11, "mode": "write", "tags": ["a", 2, "c"], "extra": 1}
    found = check_arguments(args, SCHEMA)
    assert "path: required and missing" in found
    assert "count: 11 is above the maximum 10" in found
    assert any(v.startswith("mode: must be one of") for v in found)
    assert "tags: more than 2 items" in found
    assert "tags[1]: expected string, got integer" in found
    assert "extra: not a declared parameter" in found


def test_pattern_and_length():
    assert check_arguments({"path": "/var/log/app.log"}, SCHEMA) == [
        "path: does not match the pattern '^/srv/'"
    ]
    assert "path: shorter than 1 characters" in check_arguments({"path": ""}, SCHEMA)


def test_boolean_is_not_an_integer():
    assert check_arguments({"path": "/srv/a", "count": True}, SCHEMA) == [
        "count: expected integer, got boolean"
    ]


def test_integral_float_is_an_integer():
    assert check_arguments({"path": "/srv/a", "count": 3.0}, SCHEMA) == []


def test_nested_objects_and_exclusive_bounds():
    schema = {
        "type": "object",
        "properties": {
            "limits": {
                "type": "object",
                "properties": {"rate": {"type": "number", "exclusiveMaximum": 1}},
                "required": ["rate"],
            }
        },
    }
    assert check_arguments({"limits": {"rate": 1}}, schema) == [
        "limits.rate: 1 must be less than 1"
    ]
    assert check_arguments({"limits": {}}, schema) == ["limits.rate: required and missing"]


def test_schema_false_refuses_and_empty_schema_accepts():
    schema = {"type": "object", "properties": {"x": False}}
    assert check_arguments({"x": 1}, schema) == ["x: no value is allowed here"]
    assert check_arguments({"anything": [1, {"a": None}]}, {}) == []


def test_a_bad_pattern_does_not_raise():
    schema = {"type": "object", "properties": {"q": {"type": "string", "pattern": "("}}}
    assert check_arguments({"q": "x"}, schema) == []


def test_unchecked_keywords_are_reported():
    schema = {
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "string"}]}, "b": {"$ref": "#/x"}},
    }
    assert unchecked_keywords(schema) == ["$ref", "anyOf"]
    assert unchecked_keywords(SCHEMA) == []


# The proxy


def _proxy(monkeypatch, *, enforce=True):
    from vaara.audit.trail import AuditTrail
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=AuditTrail(on_record=lambda _r: None), enforce=enforce)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)
    upstream = MagicMock()
    p._upstream = upstream
    return p, pipeline, upstream


def _list(p, upstream, tools):
    upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {"tools": tools}}
    return p._handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})


def _call(p, upstream, args, tool="read_log"):
    upstream.request.reset_mock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    return p._handle_request({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    })


def _gates(monkeypatch, p):
    seen: list[list[str]] = []
    real = p._overt_emit

    def spy(**kw):
        seen.append(list((kw.get("extra") or {}).get("gates", [])))
        return real(**kw)

    monkeypatch.setattr(p, "_overt_emit", spy)
    return seen


def test_violation_is_refused_before_scoring_or_forwarding(monkeypatch):
    p, pipeline, upstream = _proxy(monkeypatch)
    _list(p, upstream, [{"name": "read_log", "inputSchema": SCHEMA}])
    scored = MagicMock(side_effect=AssertionError("must not be scored"))
    monkeypatch.setattr(pipeline, "intercept", scored)
    gates = _gates(monkeypatch, p)

    resp = _call(p, upstream, {"path": "/var/log/app.log", "count": 50})

    payload = json.loads(resp["result"]["content"][0]["text"])
    assert resp["result"]["isError"] is True
    assert payload["decision"] == "DENY"
    assert "count: 50 is above the maximum 10" in payload["violations"]
    assert gates[-1] == ["operator_filter:pass", "deny_rules:pass", "parameter_schema:deny"]
    upstream.request.assert_not_called()


def test_refusal_on_the_chain_names_the_parameter(monkeypatch):
    from vaara.audit.trail import EventType

    p, pipeline, upstream = _proxy(monkeypatch)
    _list(p, upstream, [{"name": "read_log", "inputSchema": SCHEMA}])
    _call(p, upstream, {"path": "/srv/a", "count": 0})

    [record] = pipeline.trail.get_records_by_type(EventType.ACTION_BLOCKED)
    assert record.data["policy_id"] == "upstream_input_schema"
    assert record.data["violation_type"] == "parameter_schema"
    assert "count: 0 is below the minimum 1" in record.data["reason"]


def test_conforming_call_passes_the_gate(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch)
    _list(p, upstream, [{"name": "read_log", "inputSchema": SCHEMA}])
    gates = _gates(monkeypatch, p)

    resp = _call(p, upstream, {"path": "/srv/app.log", "count": 5})

    assert resp["result"]["content"][0]["text"] == "ok"
    assert gates[-1][:3] == ["operator_filter:pass", "deny_rules:pass", "parameter_schema:pass"]


def test_a_schema_the_checker_only_partly_reads_says_so(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch)
    schema = {"type": "object", "properties": {"q": {"oneOf": [{"type": "string"}]}}}
    _list(p, upstream, [{"name": "read_log", "inputSchema": schema}])
    gates = _gates(monkeypatch, p)

    _call(p, upstream, {"q": "x"})

    assert gates[-1][2] == "parameter_schema:pass_partial:oneOf"


def test_shadow_mode_records_and_forwards(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch, enforce=False)
    _list(p, upstream, [{"name": "read_log", "inputSchema": SCHEMA}])
    gates = _gates(monkeypatch, p)

    _call(p, upstream, {"path": "/var/log/app.log"})

    assert "parameter_schema:shadow" in gates[-1]
    upstream.request.assert_called_once()


def test_a_tool_with_no_schema_says_the_gate_had_nothing_to_check(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch)
    _list(p, upstream, [{"name": "read_log"}])
    gates = _gates(monkeypatch, p)

    _call(p, upstream, {"anything": 1})

    assert gates[-1][2] == "parameter_schema:no_schema"


def test_a_relisted_tool_uses_its_new_schema(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch)
    _list(p, upstream, [{"name": "read_log", "inputSchema": SCHEMA}])
    _list(p, upstream, [{"name": "read_log", "inputSchema": {"type": "object"}}])
    gates = _gates(monkeypatch, p)

    _call(p, upstream, {"path": "/var/log/app.log"})

    assert gates[-1][2] == "parameter_schema:pass"


# Fan-out: the schema belongs to the upstream that advertised it


def _fanout_proxy(monkeypatch):
    from vaara.audit.trail import AuditTrail
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=AuditTrail(on_record=lambda _r: None), enforce=True)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(upstreams={"a": ["echo"], "b": ["echo"]}, pipeline=pipeline)
    clients = {"a": MagicMock(), "b": MagicMock()}
    p._upstreams["a"], p._upstreams["b"] = clients["a"], clients["b"]
    p._upstreams["default"] = clients["a"]
    return p, clients


def _on(name, fn, *args, **kw):
    from vaara.integrations import mcp_proxy

    token = mcp_proxy._REQUEST_UPSTREAM.set(name)
    try:
        return fn(*args, **kw)
    finally:
        mcp_proxy._REQUEST_UPSTREAM.reset(token)


def test_same_named_tools_on_two_upstreams_keep_their_own_schemas(monkeypatch):
    p, clients = _fanout_proxy(monkeypatch)
    open_schema = {"type": "object"}
    _on("a", _list, p, clients["a"], [{"name": "read_log", "inputSchema": SCHEMA}])
    _on("b", _list, p, clients["b"], [{"name": "read_log", "inputSchema": open_schema}])
    gates = _gates(monkeypatch, p)

    # Listed last, b's open schema must not stand in for a's strict one.
    resp = _on("a", _call, p, clients["a"], {"path": "/var/log/app.log"})
    assert resp["result"]["isError"] is True
    assert gates[-1][2] == "parameter_schema:deny"
    clients["a"].request.assert_not_called()

    # And a's strict schema must not refuse a call b's schema allows.
    resp = _on("b", _call, p, clients["b"], {"path": "/var/log/app.log"})
    assert resp["result"]["content"][0]["text"] == "ok"
    assert gates[-1][2] == "parameter_schema:pass"


def test_a_tool_listed_on_one_upstream_has_no_schema_on_the_other(monkeypatch):
    p, clients = _fanout_proxy(monkeypatch)
    _on("a", _list, p, clients["a"], [{"name": "read_log", "inputSchema": SCHEMA}])
    gates = _gates(monkeypatch, p)

    _on("b", _call, p, clients["b"], {"path": "/var/log/app.log"})

    assert gates[-1][2] == "parameter_schema:no_schema"
