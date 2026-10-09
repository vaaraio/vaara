# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``vaara run`` adds around the agent beside the AppArmor floor.

Three kernel layers, applied in the forked child before it execs the agent,
so the agent and everything it starts inherit them and none can lift them:

- ``no_new_privs``: no exec in the tree gains privileges (setuid, file
  capabilities). The other two layers require it.
- A seccomp filter that refuses, with ``EPERM``, the system calls an agent's
  tools have no use for and an escape needs: loading kernel modules, kexec,
  eBPF, perf, ptrace and cross-process memory access, mounts and the new
  mount API, swap, reboot, the kernel keyring, ``userfaultfd``,
  ``open_by_handle_at``, setting the clock, and io_uring (whose socket and
  file operations seccomp never sees). System calls of a foreign ABI (32-bit
  compat, x32) are refused as a whole. With the network locked it also
  refuses UDP and raw IP sockets, so nothing leaves by DNS or ICMP, and
  stream sockets of any protocol but TCP (MPTCP, SCTP).
- Landlock network rules (Landlock ABI 4 and later): TCP connections reach
  only the ports given, which is the egress proxy's. With ABI 6 and later
  the tree is also scoped so it cannot signal processes outside itself.
  Abstract unix sockets are left reachable: the hook relay to ``vaara run``
  is one (:mod:`vaara.oslayer.forward`).

Landlock rules name ports, not addresses, so the proxy's port number on a
remote host is reachable too. The guard closes that where it runs as root
by an address rule on the launch's cgroup; this module says which layers it
applied and the caller records them.

The system call numbers are pinned from the kernel's own tables
(``arch/x86/entry/syscalls/syscall_64.tbl`` and
``include/uapi/asm-generic/unistd.h``, which arm64 uses).
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import struct
import sys
from typing import Iterable, Optional

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000

AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}

# BPF opcodes used here.
_LD_W_ABS = 0x20
_JEQ_K = 0x15
_JGE_K = 0x35
_AND_K = 0x54
_RET_K = 0x06

# Offsets into struct seccomp_data (little-endian on both arches).
_NR, _ARCH, _ARG0, _ARG1, _ARG2 = 0, 4, 16, 24, 32

AF_INET, AF_INET6, AF_PACKET = 2, 10, 17
SOCK_STREAM = 1
IPPROTO_TCP = 6
SOCK_TYPE_MASK = 0xF

# The calls refused outright, by name, then per arch.
DENIED = (
    "init_module", "finit_module", "delete_module", "kexec_load", "kexec_file_load",
    "bpf", "perf_event_open", "ptrace", "process_vm_readv", "process_vm_writev",
    "mount", "umount2", "pivot_root", "open_tree", "move_mount", "fsopen", "fsconfig",
    "fsmount", "fspick", "swapon", "swapoff", "reboot", "keyctl", "add_key",
    "request_key", "userfaultfd", "open_by_handle_at", "settimeofday", "clock_settime",
    "acct", "quotactl", "iopl", "ioperm", "io_uring_setup", "io_uring_enter",
    "io_uring_register",
)

SYSCALLS: dict[str, dict[str, int]] = {
    "x86_64": {
        "init_module": 175, "finit_module": 313, "delete_module": 176, "kexec_load": 246,
        "kexec_file_load": 320, "bpf": 321, "perf_event_open": 298, "ptrace": 101,
        "process_vm_readv": 310, "process_vm_writev": 311, "mount": 165, "umount2": 166,
        "pivot_root": 155, "open_tree": 428, "move_mount": 429, "fsopen": 430,
        "fsconfig": 431, "fsmount": 432, "fspick": 433, "swapon": 167, "swapoff": 168,
        "reboot": 169, "keyctl": 250, "add_key": 248, "request_key": 249,
        "userfaultfd": 323, "open_by_handle_at": 304, "settimeofday": 164,
        "clock_settime": 227, "acct": 163, "quotactl": 179, "iopl": 172, "ioperm": 173,
        "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
        "socket": 41,
    },
    "aarch64": {
        "init_module": 105, "finit_module": 273, "delete_module": 106, "kexec_load": 104,
        "kexec_file_load": 294, "bpf": 280, "perf_event_open": 241, "ptrace": 117,
        "process_vm_readv": 270, "process_vm_writev": 271, "mount": 40, "umount2": 39,
        "pivot_root": 41, "open_tree": 428, "move_mount": 429, "fsopen": 430,
        "fsconfig": 431, "fsmount": 432, "fspick": 433, "swapon": 224, "swapoff": 225,
        "reboot": 142, "keyctl": 219, "add_key": 217, "request_key": 218,
        "userfaultfd": 282, "open_by_handle_at": 265, "settimeofday": 170,
        "clock_settime": 112, "acct": 89, "quotactl": 60,
        "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
        "socket": 198,
    },
}

#: Changes whenever the filter's meaning changes; recorded with the launch.
FILTER_VERSION = "vaara-seccomp/1"

# Landlock.
_LL_CREATE, _LL_ADD_RULE, _LL_RESTRICT = 444, 445, 446
_LL_CREATE_RULESET_VERSION = 1
_LL_RULE_NET_PORT = 2
_LL_ACCESS_NET_CONNECT_TCP = 1 << 1
_LL_SCOPE_SIGNAL = 1 << 1


class HardenError(OSError):
    """A layer that was asked for could not be applied."""


def machine() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def _libc():
    return ctypes.CDLL(None, use_errno=True)


# ---------------------------------------------------------------- seccomp


class _Asm:
    """A few BPF instructions with forward labels."""

    def __init__(self) -> None:
        self.code: list[tuple[int, Optional[str], Optional[str], int]] = []
        self.labels: dict[str, int] = {}

    def op(self, code: int, k: int = 0, jt: Optional[str] = None,
           jf: Optional[str] = None) -> None:
        self.code.append((code, jt, jf, k))

    def label(self, name: str) -> None:
        self.labels[name] = len(self.code)

    def assemble(self) -> bytes:
        out = bytearray()
        for i, (code, jt, jf, k) in enumerate(self.code):
            def rel(target: Optional[str]) -> int:
                if target is None:
                    return 0
                offset = self.labels[target] - i - 1
                if not 0 <= offset <= 255:
                    raise ValueError(f"jump to {target} out of range")
                return offset
            out += struct.pack("<HBBI", code, rel(jt), rel(jf), k & 0xFFFFFFFF)
        return bytes(out)


def seccomp_program(arch: Optional[str] = None, lock_network: bool = False) -> bytes:
    """The filter as raw ``struct sock_filter`` instructions."""
    arch = arch or machine()
    if arch not in SYSCALLS:
        raise HardenError(errno.ENOSYS, f"no seccomp table for {arch}")
    table = SYSCALLS[arch]
    deny = SECCOMP_RET_ERRNO | errno.EPERM
    a = _Asm()
    a.op(_LD_W_ABS, _ARCH)
    a.op(_JEQ_K, AUDIT_ARCH[arch], jt="native", jf="deny")
    a.label("native")
    a.op(_LD_W_ABS, _NR)
    if arch == "x86_64":
        a.op(_JGE_K, 0x40000000, jt="deny", jf="nr")  # the x32 ABI
        a.label("nr")
    for name in DENIED:
        if name in table:
            a.op(_JEQ_K, table[name], jt="deny")
    if lock_network:
        a.op(_JEQ_K, table["socket"], jt="socket", jf="allow")
        a.label("socket")
        a.op(_LD_W_ABS, _ARG0)
        a.op(_JEQ_K, AF_PACKET, jt="deny")
        a.op(_JEQ_K, AF_INET, jt="inet")
        a.op(_JEQ_K, AF_INET6, jt="inet", jf="allow")
        a.label("inet")
        a.op(_LD_W_ABS, _ARG1)
        a.op(_AND_K, SOCK_TYPE_MASK)
        a.op(_JEQ_K, SOCK_STREAM, jt="stream", jf="deny")
        # Landlock's TCP rules bind TCP sockets only: a stream socket of
        # another protocol (MPTCP, SCTP) would connect anywhere.
        a.label("stream")
        a.op(_LD_W_ABS, _ARG2)
        a.op(_JEQ_K, 0, jt="allow")
        a.op(_JEQ_K, IPPROTO_TCP, jt="allow", jf="deny")
    a.label("allow")
    a.op(_RET_K, SECCOMP_RET_ALLOW)
    a.label("deny")
    a.op(_RET_K, deny)
    return a.assemble()


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _install_seccomp(program: bytes) -> None:
    buf = ctypes.create_string_buffer(program, len(program))
    prog = _SockFprog(len(program) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = _libc()
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        e = ctypes.get_errno()
        raise HardenError(e, f"seccomp filter refused: {os.strerror(e)}")


def no_new_privs() -> None:
    if _libc().prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        e = ctypes.get_errno()
        raise HardenError(e, f"no_new_privs refused: {os.strerror(e)}")


# ---------------------------------------------------------------- Landlock


def landlock_abi() -> int:
    """The kernel's Landlock ABI version, 0 when Landlock is unavailable."""
    if not sys.platform.startswith("linux"):
        return 0
    r = _libc().syscall(_LL_CREATE, None, ctypes.c_size_t(0),
                        ctypes.c_uint32(_LL_CREATE_RULESET_VERSION))
    return int(r) if r > 0 else 0


def _landlock_network(ports: Iterable[int], abi: int) -> list[str]:
    libc = _libc()
    scoped = _LL_SCOPE_SIGNAL if abi >= 6 else 0
    attr = struct.pack("<QQQ", 0, _LL_ACCESS_NET_CONNECT_TCP, scoped)
    size = 24 if abi >= 6 else 16
    buf = ctypes.create_string_buffer(attr[:size], size)
    fd = libc.syscall(_LL_CREATE, buf, ctypes.c_size_t(size), ctypes.c_uint32(0))
    if fd < 0:
        e = ctypes.get_errno()
        raise HardenError(e, f"Landlock ruleset refused: {os.strerror(e)}")
    try:
        for port in ports:
            rule = ctypes.create_string_buffer(
                struct.pack("<QQ", _LL_ACCESS_NET_CONNECT_TCP, int(port)), 16)
            if libc.syscall(_LL_ADD_RULE, ctypes.c_int(fd), ctypes.c_int(_LL_RULE_NET_PORT),
                            rule, ctypes.c_uint32(0)) != 0:
                e = ctypes.get_errno()
                raise HardenError(e, f"Landlock rule for port {port} refused: "
                                     f"{os.strerror(e)}")
        if libc.syscall(_LL_RESTRICT, ctypes.c_int(fd), ctypes.c_uint32(0)) != 0:
            e = ctypes.get_errno()
            raise HardenError(e, f"Landlock restrict refused: {os.strerror(e)}")
    finally:
        os.close(fd)
    layers = ["landlock_net"]
    if scoped:
        layers.append("landlock_scope")
    return layers


# ---------------------------------------------------------------- apply


def apply(egress_ports: Optional[Iterable[int]] = None) -> dict:
    """Harden the calling process and everything it will exec.

    ``egress_ports``: when given, TCP may connect only to these ports and
    UDP and raw IP sockets are refused. Requires Landlock ABI 4; without it
    the call fails rather than leave the network open while the caller
    believes it closed. Returns what was applied, for the record.
    """
    lock = egress_ports is not None
    ports = sorted({int(p) for p in egress_ports or ()})
    abi = landlock_abi()
    if lock and abi < 4:
        raise HardenError(errno.ENOSYS, "locking egress needs Landlock ABI 4 (Linux 6.7 "
                                        f"or later); this kernel has {abi or 'none'}")
    no_new_privs()
    layers = ["no_new_privs"]
    if lock:
        layers += _landlock_network(ports, abi)
    _install_seccomp(seccomp_program(lock_network=lock))
    layers.append("seccomp")
    return {"layers": layers, "seccomp": FILTER_VERSION, "arch": machine(),
            "landlock_abi": abi, "egress_ports": ports if lock else None}
