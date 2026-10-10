# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The address rule on a launch's cgroup: connect only to the egress proxy.

Landlock's network rules name ports, not addresses, so with egress locked a
tree could still reach the egress proxy's port number on a remote host. The
guard closes that as root: when a launch locks egress, it attaches eBPF
programs to the launch's cgroup (``BPF_PROG_TYPE_CGROUP_SOCK_ADDR``) that
the kernel runs on every ``connect()`` and UDP ``sendmsg()`` from a process
in it:

- IPv4 ``connect()``: allowed to ``127.0.0.1`` on the given ports, refused
  with ``EPERM`` everywhere else.
- IPv6 ``connect()``, IPv4 and IPv6 UDP ``sendmsg()``: refused. The proxy
  listens on IPv4 loopback, and the seccomp filter already refuses UDP.

Unix sockets are not inet sockets and these hooks do not see them, so the
hook relay and the guard's socket keep working. A program attached this way
stays with the cgroup and goes when the guard removes it at the end of the
launch. Nothing in the tree can detach it: that takes ``CAP_BPF`` and
``CAP_NET_ADMIN`` (or ``CAP_SYS_ADMIN``) in the initial namespace.

The programs are a few instructions written out here, so the guard needs no
compiler and no library; the instruction encoding and the ``bpf_sock_addr``
offsets are from ``include/uapi/linux/bpf.h``.
"""

from __future__ import annotations

import ctypes
import os
import socket
import struct
from pathlib import Path
from typing import Iterable

# bpf(2) commands, program and attach types (include/uapi/linux/bpf.h)
BPF_PROG_LOAD = 5
BPF_PROG_ATTACH = 8
BPF_PROG_TYPE_CGROUP_SOCK_ADDR = 18
BPF_CGROUP_INET4_CONNECT = 10
BPF_CGROUP_INET6_CONNECT = 11
BPF_CGROUP_UDP4_SENDMSG = 14
BPF_CGROUP_UDP6_SENDMSG = 15
# Every attached program runs, and any one refusing refuses the call; the
# exclusive mode would fail under an ancestor that attached with this flag.
BPF_F_ALLOW_MULTI = 2

# struct bpf_sock_addr: user_family, user_ip4, user_ip6[4], user_port
_USER_IP4 = 4
_USER_PORT = 24

# Instruction encoding
_LDX_W = 0x61     # BPF_LDX | BPF_MEM | BPF_W
_JEQ_K = 0x15     # BPF_JMP | BPF_JEQ | BPF_K
_JNE_K = 0x55     # BPF_JMP | BPF_JNE | BPF_K
_MOV64_K = 0xB7   # BPF_ALU64 | BPF_MOV | BPF_K
_EXIT = 0x95      # BPF_JMP | BPF_EXIT

_BPF_NR = {"x86_64": 321, "aarch64": 280}


class NetlockError(OSError):
    pass


def _insn(code: int, dst: int = 0, src: int = 0, off: int = 0, imm: int = 0) -> bytes:
    return struct.pack("<BBhi", code, (src << 4) | dst, off, imm)


def _u32(raw: bytes) -> int:
    """A field the kernel keeps in network order, as the program loads it.

    The load zero-extends and a jump's immediate is sign-extended, so a value
    at or above 2**31 would never match; loopback and ports stay below it.
    """
    value = struct.unpack("=I", raw)[0]
    if value >= 1 << 31:
        raise NetlockError(0, f"value {value:#x} does not fit a jump immediate")
    return value


def connect4_program(ports: Iterable[int]) -> bytes:
    """Allow ``127.0.0.1`` on ``ports``, refuse every other IPv4 connect."""
    ports = list(ports)
    loopback = _u32(socket.inet_aton("127.0.0.1"))
    code = [
        _insn(_LDX_W, dst=2, src=1, off=_USER_IP4),
        None,  # if r2 != 127.0.0.1 goto deny, filled in below
        _insn(_LDX_W, dst=2, src=1, off=_USER_PORT),
    ]
    code += [None] * len(ports)  # if r2 == port goto allow
    deny = len(code)
    code += [_insn(_MOV64_K, dst=0, imm=0), _insn(_EXIT)]
    allow = len(code)
    code += [_insn(_MOV64_K, dst=0, imm=1), _insn(_EXIT)]
    code[1] = _insn(_JNE_K, dst=2, off=deny - 2, imm=loopback)
    for i, port in enumerate(ports):
        at = 3 + i
        wire = _u32(struct.pack("!H", port) + b"\0\0")
        code[at] = _insn(_JEQ_K, dst=2, off=allow - at - 1, imm=wire)
    return b"".join(code)  # type: ignore[arg-type]


def refuse_program() -> bytes:
    return _insn(_MOV64_K, dst=0, imm=0) + _insn(_EXIT)


def _machine() -> str:
    m = os.uname().machine.lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def _bpf(cmd: int, attr: ctypes.Array) -> int:
    nr = _BPF_NR.get(_machine())
    if nr is None:
        raise NetlockError(0, f"no bpf syscall number for {_machine()}")
    libc = ctypes.CDLL(None, use_errno=True)
    ret = libc.syscall(nr, ctypes.c_int(cmd), attr, ctypes.c_uint(len(attr)))
    if ret < 0:
        e = ctypes.get_errno()
        raise NetlockError(e, os.strerror(e))
    return ret


def _load(program: bytes, attach_type: int) -> int:
    insns = ctypes.create_string_buffer(program, len(program))
    license_ = ctypes.create_string_buffer(b"GPL")
    log = ctypes.create_string_buffer(4096)
    attr = ctypes.create_string_buffer(128)
    struct.pack_into("=IIQQIIQ", attr, 0, BPF_PROG_TYPE_CGROUP_SOCK_ADDR, len(program) // 8,
                     ctypes.addressof(insns), ctypes.addressof(license_), 1, len(log),
                     ctypes.addressof(log))
    struct.pack_into("=16s", attr, 48, b"vaara_netlock")
    struct.pack_into("=I", attr, 68, attach_type)  # expected_attach_type
    try:
        return _bpf(BPF_PROG_LOAD, attr)
    except NetlockError as exc:
        detail = log.value.decode("utf-8", "replace").strip()
        raise NetlockError(exc.errno or 0, f"the kernel refused the program: {exc.strerror}"
                           + (f" ({detail})" if detail else "")) from None


def lock(cgroup_dir: Path, ports: Iterable[int]) -> list[str]:
    """Attach the address rule to ``cgroup_dir``. Returns what was attached."""
    ports = [int(p) for p in ports]
    if not ports or not all(0 < p < 65536 for p in ports):
        raise NetlockError(0, f"egress ports out of range: {ports}")
    plan = [("connect4", BPF_CGROUP_INET4_CONNECT, connect4_program(ports)),
            ("connect6", BPF_CGROUP_INET6_CONNECT, refuse_program()),
            ("sendmsg4", BPF_CGROUP_UDP4_SENDMSG, refuse_program()),
            ("sendmsg6", BPF_CGROUP_UDP6_SENDMSG, refuse_program())]
    cg = os.open(cgroup_dir, os.O_RDONLY | os.O_DIRECTORY)
    attached = []
    try:
        for name, attach_type, program in plan:
            prog = _load(program, attach_type)
            try:
                attr = ctypes.create_string_buffer(struct.pack("=IIII", cg, prog, attach_type,
                                                               BPF_F_ALLOW_MULTI),
                                                   32)
                try:
                    _bpf(BPF_PROG_ATTACH, attr)
                except NetlockError as exc:
                    raise NetlockError(exc.errno or 0,
                                       f"could not attach {name}: {exc.strerror}") from None
            finally:
                os.close(prog)  # the attachment holds the program
            attached.append(name)
    finally:
        os.close(cg)
    return attached
