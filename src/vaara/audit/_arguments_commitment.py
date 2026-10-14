# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A salted commitment to the arguments a decision was made over.

A decision receipt leaves the machine without the tool arguments, so on its
own it cannot show which arguments were decided on. A plain digest of them
would not help: most arguments are low entropy (a path, a role, an amount),
and anyone holding the receipt could confirm a guess by hashing it.

So each action request draws a fresh 16-byte salt, kept on the trail beside
the arguments, and the receipt carries::

    sha256( "vaara.arguments.v1\\n" || salt || JCS(arguments) )

To show one decision's arguments to someone, the trail's holder hands over
those arguments and that salt. The receiver recomputes the digest and compares
it with the receipt. Nothing else is opened: every other decision has its own
salt, and no key is shared.

What it does not promise: two decisions over the same arguments carry
different commitments, so receipts alone cannot show that two decisions saw
the same input. JCS here is RFC 8785 as the ``rfc8785`` package writes it,
floats included, because tool arguments carry floats.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Optional

DOMAIN = b"vaara.arguments.v1\n"
SALT_BYTES = 16
_SALT = re.compile(r"^[0-9a-f]{32}$")


def new_salt() -> str:
    """A fresh salt, as 32 lowercase hex characters."""
    return os.urandom(SALT_BYTES).hex()


def commit(arguments: Any, salt: str) -> Optional[str]:
    """``sha256:`` and the commitment, or None when the arguments cannot be canonicalised."""
    if not _SALT.match(salt or ""):
        raise ValueError("salt must be 32 lowercase hex characters")
    try:
        import rfc8785

        canonical = rfc8785.dumps(arguments)
    except Exception:
        return None
    digest = hashlib.sha256(DOMAIN + bytes.fromhex(salt) + canonical).hexdigest()
    return "sha256:" + digest


def verify(commitment: str, arguments: Any, salt: str) -> bool:
    """True when ``arguments`` under ``salt`` produce ``commitment``."""
    try:
        recomputed = commit(arguments, salt)
    except ValueError:
        return False
    return recomputed is not None and hmac.compare_digest(recomputed, commitment or "")
