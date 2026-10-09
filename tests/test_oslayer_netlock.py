# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The address rule's programs, decoded and run by a tiny interpreter.

Attaching needs root, which the Linux OS layer CI job has; the hardened e2e
test there checks the rule on the real kernel. Here the program text is
checked against what the kernel would compute.
"""

from __future__ import annotations

import socket
import struct

import pytest

from vaara.oslayer import netlock


def _run(program: bytes, ip: str, port: int) -> int:
    """Interpret the handful of instructions netlock emits."""
    ctx = {netlock._USER_IP4: struct.unpack("=I", socket.inet_aton(ip))[0],
           netlock._USER_PORT: struct.unpack("=I", struct.pack("!H", port) + b"\0\0")[0]}
    insns = [struct.unpack("<BBhi", program[i:i + 8]) for i in range(0, len(program), 8)]
    regs = [0] * 11
    pc = 0
    while True:
        code, regsel, off, imm = insns[pc]
        dst, src = regsel & 0xF, regsel >> 4
        if code == netlock._LDX_W:
            assert src == 1
            regs[dst] = ctx[off]
        elif code == netlock._MOV64_K:
            regs[dst] = imm
        elif code == netlock._JEQ_K:
            if regs[dst] == imm:
                pc += off
        elif code == netlock._JNE_K:
            if regs[dst] != imm:
                pc += off
        elif code == netlock._EXIT:
            return regs[0]
        else:
            raise AssertionError(f"unexpected opcode {code:#x}")
        pc += 1


def test_connect4_allows_only_loopback_on_the_proxy_ports():
    prog = netlock.connect4_program([41234, 8080])
    assert _run(prog, "127.0.0.1", 41234) == 1
    assert _run(prog, "127.0.0.1", 8080) == 1
    assert _run(prog, "127.0.0.1", 41235) == 0
    assert _run(prog, "203.0.113.9", 41234) == 0  # the proxy's port on a remote host
    assert _run(prog, "127.0.0.2", 41234) == 0


def test_refuse_program_refuses():
    assert _run(netlock.refuse_program(), "127.0.0.1", 41234) == 0


def test_bad_ports_are_refused_before_anything_loads(tmp_path):
    for ports in ([], [0], [70000]):
        with pytest.raises(netlock.NetlockError):
            netlock.lock(tmp_path, ports)
