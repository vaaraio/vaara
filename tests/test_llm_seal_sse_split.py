"""A placeholder split across two SSE events is restored, and counted.

MEASURED 2026-09-19 on the live llm-proxy trail: 1,881 outcome records, a
placeholder restored once, an unrestored one reported never, while every
request sealed 30 to 45. The byte-level unsealer held back a carry sized for
an HTTP chunk boundary; a client streaming tool input emits many small
input_json_delta events, and a placeholder that straddles two of them has
event framing between its halves. The placeholders went back literal and the
record said clean. These tests pin the frame-aware unsealer.
"""

from __future__ import annotations

import json

from vaara.integrations.llm_seal import (
    SealRegistry,
    SseUnsealer,
    StreamUnsealer,
    _partial_placeholder_tail,
    placeholder_for,
)

TERM = "anti-note"
REG = SealRegistry({"concept": TERM})
PH = placeholder_for(TERM)


def _delta(text, index=0, dtype="input_json_delta", field="partial_json"):
    return ("event: content_block_delta\ndata: "
            + json.dumps({"type": "content_block_delta", "index": index,
                          "delta": {"type": dtype, field: text}})
            + "\n\n").encode()


def _stop(index=0):
    return ("event: content_block_stop\ndata: "
            + json.dumps({"type": "content_block_stop", "index": index})
            + "\n\n").encode()


def _texts(out: bytes) -> str:
    """Concatenate the delta text a client would assemble from the stream."""
    parts = []
    for frame in out.decode().split("\n\n"):
        for ln in frame.split("\n"):
            if ln.startswith("data:"):
                d = json.loads(ln[5:])
                if d.get("type") == "content_block_delta":
                    delta = d["delta"]
                    parts.append(delta.get("partial_json") or delta.get("text")
                                 or delta.get("thinking") or "")
    return "".join(parts)


def test_split_across_two_events_is_restored():
    cut = len(PH) // 2
    stream = _delta("x = " + PH[:cut]) + _delta(PH[cut:] + "_version") + _stop()
    u = SseUnsealer(REG)
    out = u.feed(stream) + u.flush()
    assert _texts(out) == "x = " + TERM + "_version"
    assert u.restored == 1
    assert u.unmapped == []


def test_the_byte_unsealer_could_not_do_this():
    cut = len(PH) // 2
    stream = _delta("x = " + PH[:cut]) + _delta(PH[cut:]) + _stop()
    u = StreamUnsealer(REG)
    out = u.feed(stream) + u.flush()
    assert PH[:cut] in out.decode()  # left literal, the defect this file pins
    assert u.restored == 0


def test_split_one_char_per_event():
    text = "before " + PH + " after"
    stream = b"".join(_delta(c) for c in text) + _stop()
    u = SseUnsealer(REG)
    out = b"".join(u.feed(stream[i:i + 7]) for i in range(0, len(stream), 7))
    out += u.flush()
    assert _texts(out) == "before " + TERM + " after"
    assert u.restored == 1


def test_whole_placeholder_in_one_event():
    u = SseUnsealer(REG)
    out = u.feed(_delta("a " + PH + " b") + _stop()) + u.flush()
    assert _texts(out) == "a " + TERM + " b"
    assert u.restored == 1


def test_unknown_placeholder_is_reported_not_swallowed():
    other = placeholder_for("something-else")
    cut = 5
    u = SseUnsealer(REG)
    out = u.feed(_delta("q " + other[:cut]) + _delta(other[cut:]) + _stop()) + u.flush()
    assert _texts(out) == "q " + other
    assert u.unmapped == [other]
    assert u.restored == 0


def test_text_without_placeholders_passes_byte_identical():
    frames = _delta("hello ", dtype="text_delta", field="text") \
        + _delta("world", dtype="text_delta", field="text") + _stop()
    u = SseUnsealer(REG)
    out = u.feed(frames) + u.flush()
    assert _texts(out) == "hello world"
    # Every frame is still a parseable event with the same shape.
    for frame in out.decode().strip().split("\n\n"):
        assert frame.startswith("event: content_block_")


def test_held_tail_is_flushed_before_a_non_delta_event():
    # A delta ending in a possible placeholder head, then the block stops.
    u = SseUnsealer(REG)
    out = u.feed(_delta("x VAARA_SE") + _stop()) + u.flush()
    assert _texts(out) == "x VAARA_SE"
    assert out.index(b"content_block_stop") > out.rindex(b"VAARA_SE")


def test_non_sse_bytes_fall_back_to_the_regex():
    u = SseUnsealer(REG)
    out = u.feed(b'{"content":"' + PH.encode() + b'"}') + u.flush()
    assert out == b'{"content":"' + TERM.encode() + b'"}'
    assert u.restored == 1


def test_message_delta_and_ping_frames_pass_through():
    ping = b"event: ping\ndata: {\"type\": \"ping\"}\n\n"
    u = SseUnsealer(REG)
    out = u.feed(ping + _delta(PH) + ping + _stop()) + u.flush()
    assert out.count(b"event: ping") == 2
    assert _texts(out) == TERM


def test_inactive_registry_is_a_passthrough():
    u = SseUnsealer(SealRegistry())
    assert u.feed(b"anything\n\n") == b"anything\n\n"


def test_partial_tail_lengths():
    assert _partial_placeholder_tail("abc") == 0
    assert _partial_placeholder_tail("abc V") == 1
    assert _partial_placeholder_tail("abc VAARA_SEAL_") == len("VAARA_SEAL_")
    assert _partial_placeholder_tail("VAARA_SEAL_0a1b") == len("VAARA_SEAL_0a1b")
    assert _partial_placeholder_tail("VAARA_SEAL_0a1z") == 0
    assert _partial_placeholder_tail(PH) == 0  # complete, nothing to hold
