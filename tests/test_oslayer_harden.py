# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hardening ``vaara run`` applies before exec, on the real kernel.

Each check runs in a child process: a seccomp filter and a Landlock domain
cannot be lifted from the process that installed them.
"""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import textwrap

import pytest

from vaara.oslayer import harden

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux kernel layers")
known_arch = pytest.mark.skipif(harden.machine() not in harden.SYSCALLS,
                                reason="no seccomp table for this arch")


def _child(body: str) -> dict:
    code = textwrap.dedent('''
        import ctypes, errno, json, os, socket, sys
        from vaara.oslayer import harden
        out = {}
        libc = ctypes.CDLL(None, use_errno=True)
        def call(nr, *args):
            r = libc.syscall(nr, *args)
            return 0 if r >= 0 else ctypes.get_errno()
    ''') + textwrap.dedent(body) + "\nprint(json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_every_denied_call_has_a_number_on_x86_64_and_arm64():
    for arch, table in harden.SYSCALLS.items():
        missing = [n for n in harden.DENIED if n not in table]
        # iopl and ioperm exist on x86 only.
        assert set(missing) <= ({"iopl", "ioperm"} if arch == "aarch64" else set()), arch
        assert "socket" in table


def test_the_program_is_well_formed():
    for arch in harden.SYSCALLS:
        for lock in (False, True):
            prog = harden.seccomp_program(arch, lock_network=lock)
            assert len(prog) % 8 == 0 and len(prog) // 8 < 4096
            last = struct.unpack("<HBBI", prog[-8:])
            assert last == (harden._RET_K, 0, 0, harden.SECCOMP_RET_ERRNO | 1)


@linux
@known_arch
def test_denied_calls_get_eperm_and_ordinary_work_goes_on(tmp_path):
    t = harden.SYSCALLS[harden.machine()]
    out = _child(f'''
        out["applied"] = harden.apply()
        out["bpf"] = call({t["bpf"]}, 0, 0, 0)
        out["ptrace"] = call({t["ptrace"]}, 0, 0, 0, 0)
        out["keyctl"] = call({t["keyctl"]}, 0, 0, 0, 0, 0)
        out["io_uring"] = call({t["io_uring_setup"]}, 1, 0)
        out["mount"] = call({t["mount"]}, 0, 0, 0, 0, 0)
        open({str(tmp_path / "w.txt")!r}, "w").write("still writes")
        out["file"] = open({str(tmp_path / "w.txt")!r}).read()
        out["child"] = os.system("true")
        out["udp"] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM).fileno() > 0
    ''')
    assert out["applied"]["layers"] == ["no_new_privs", "seccomp"]
    assert out["applied"]["seccomp"] == harden.FILTER_VERSION
    for name in ("bpf", "ptrace", "keyctl", "io_uring", "mount"):
        assert out[name] == 1, name  # EPERM
    assert out["file"] == "still writes" and out["child"] == 0
    assert out["udp"] is True  # network not locked: UDP stays open


@linux
@known_arch
@pytest.mark.skipif(harden.landlock_abi() < 4, reason="Landlock network needs ABI 4")
def test_locked_egress_reaches_only_the_proxy_port():
    proxy = socket.socket()
    proxy.bind(("127.0.0.1", 0))
    proxy.listen(8)
    other = socket.socket()
    other.bind(("127.0.0.1", 0))
    other.listen(8)
    pp, op = proxy.getsockname()[1], other.getsockname()[1]
    try:
        out = _child(f'''
            out["applied"] = harden.apply(egress_ports=[{pp}])
            def connect(port):
                s = socket.socket()
                try:
                    s.connect(("127.0.0.1", port)); return 0
                except OSError as e:
                    return e.errno
                finally:
                    s.close()
            out["proxy"] = connect({pp})
            out["other"] = connect({op})
            for kind in ("SOCK_DGRAM", "SOCK_RAW"):
                try:
                    socket.socket(socket.AF_INET, getattr(socket, kind)); out[kind] = 0
                except OSError as e:
                    out[kind] = e.errno
            try:
                socket.socket(socket.AF_INET6, socket.SOCK_DGRAM); out["udp6"] = 0
            except OSError as e:
                out["udp6"] = e.errno
            a, b = socket.socketpair(); out["unix"] = a.fileno() > 0
            for name, proto in (("tcp0", 0), ("tcp6", 6), ("mptcp", 262), ("sctp", 132)):
                try:
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM, proto).close(); out[name] = 0
                except OSError as e:
                    out[name] = e.errno
        ''')
    finally:
        proxy.close()
        other.close()
    assert "landlock_net" in out["applied"]["layers"]
    assert out["applied"]["egress_ports"] == [pp]
    assert out["proxy"] == 0
    assert out["other"] == 13  # EACCES from Landlock
    assert out["SOCK_DGRAM"] == 1 and out["SOCK_RAW"] == 1 and out["udp6"] == 1
    assert out["unix"] is True
    # Landlock binds TCP only: other stream protocols are refused outright.
    assert out["tcp0"] == 0 and out["tcp6"] == 0
    assert out["mptcp"] == 1 and out["sctp"] == 1


def test_locking_egress_without_landlock_net_refuses(monkeypatch):
    monkeypatch.setattr(harden, "landlock_abi", lambda: 3)
    with pytest.raises(harden.HardenError, match="Landlock ABI 4"):
        harden.apply(egress_ports=[8080])
