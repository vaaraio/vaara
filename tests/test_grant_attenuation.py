"""Delegated-privilege attenuation: capability subsumption + chain check.

The subsumption predicate is sound and conservative — it must NEVER approve a
broadening, and may conservatively reject a narrowing it cannot prove. These
tests pin both the true-narrowings it accepts and the broadenings/ambiguities
it rejects.
"""

from __future__ import annotations

from vaara.credential import (
    Capability,
    capability_subsumes,
    chain_is_attenuating,
)


def _subsumes(parent, child) -> bool:
    ok, _ = capability_subsumes(parent, child)
    return ok


# ── Numeric bounds ────────────────────────────────────────────────────────

def test_tighter_upper_bound_is_attenuation():
    parent = (Capability("amount", "le", "500"),)
    child = (Capability("amount", "le", "200"),)
    assert _subsumes(parent, child)


def test_looser_upper_bound_broadens():
    parent = (Capability("amount", "le", "500"),)
    child = (Capability("amount", "le", "1000"),)
    ok, reason = capability_subsumes(parent, child)
    assert not ok and reason == "broadens:amount"


def test_unbounded_child_broadens_a_bounded_parent():
    # Child constrains a *lower* bound only -> unbounded above -> can exceed le.
    parent = (Capability("amount", "le", "500"),)
    child = (Capability("amount", "ge", "0"),)
    assert not _subsumes(parent, child)


def test_eq_within_numeric_bound_is_attenuation():
    parent = (Capability("amount", "le", "500"),)
    child = (Capability("amount", "eq", "300"),)
    assert _subsumes(parent, child)


def test_eq_outside_numeric_bound_broadens():
    parent = (Capability("amount", "le", "500"),)
    child = (Capability("amount", "eq", "900"),)
    assert not _subsumes(parent, child)


def test_ge_tighter_lower_bound_is_attenuation():
    parent = (Capability("amount", "ge", "100"),)
    child = (Capability("amount", "ge", "250"),)
    assert _subsumes(parent, child)


# ── Membership / equality ─────────────────────────────────────────────────

def test_in_subset_is_attenuation():
    parent = (Capability("vendor", "in", ("acme", "globex", "initech")),)
    child = (Capability("vendor", "in", ("acme", "globex")),)
    assert _subsumes(parent, child)


def test_in_superset_broadens():
    parent = (Capability("vendor", "in", ("acme", "globex")),)
    child = (Capability("vendor", "in", ("acme", "globex", "evilcorp")),)
    assert not _subsumes(parent, child)


def test_eq_within_parent_in_set_is_attenuation():
    parent = (Capability("vendor", "in", ("acme", "globex")),)
    child = (Capability("vendor", "eq", "acme"),)
    assert _subsumes(parent, child)


def test_eq_outside_parent_in_set_broadens():
    parent = (Capability("vendor", "in", ("acme", "globex")),)
    child = (Capability("vendor", "eq", "evilcorp"),)
    assert not _subsumes(parent, child)


def test_child_interval_cannot_satisfy_parent_membership():
    # Parent pins a string set; child offers a numeric range -> not provable.
    parent = (Capability("vendor", "in", ("acme", "globex")),)
    child = (Capability("vendor", "le", "500"),)
    assert not _subsumes(parent, child)


# ── Arg-set alignment ─────────────────────────────────────────────────────

def test_differing_arg_sets_fail_closed():
    parent = (Capability("amount", "le", "500"),)
    child = (
        Capability("amount", "le", "200"),
        Capability("vendor", "eq", "acme"),
    )
    ok, reason = capability_subsumes(parent, child)
    assert not ok and reason == "arg_set_mismatch"


def test_multi_arg_all_must_attenuate():
    parent = (
        Capability("amount", "le", "500"),
        Capability("vendor", "in", ("acme", "globex")),
    )
    ok_child = (
        Capability("amount", "le", "200"),
        Capability("vendor", "eq", "acme"),
    )
    bad_child = (
        Capability("amount", "le", "200"),
        Capability("vendor", "eq", "evilcorp"),
    )
    assert _subsumes(parent, ok_child)
    assert not _subsumes(parent, bad_child)


def test_identical_grants_subsume():
    caps = (Capability("amount", "le", "500"),)
    assert _subsumes(caps, caps)


def test_empty_grants_subsume():
    assert _subsumes((), ())


# ── Chain-level attenuation ───────────────────────────────────────────────

def test_chain_all_narrowing_is_clean():
    chain = [
        (Capability("amount", "le", "1000"),),  # planner
        (Capability("amount", "le", "500"),),   # researcher
        (Capability("amount", "le", "100"),),   # executor
    ]
    report = chain_is_attenuating(chain)
    assert report.ok
    assert report.first_broadening_index == -1


def test_chain_flags_first_broadening_hop():
    chain = [
        (Capability("amount", "le", "500"),),   # planner
        (Capability("amount", "le", "200"),),   # researcher (ok)
        (Capability("amount", "le", "900"),),   # executor exceeds researcher
    ]
    report = chain_is_attenuating(chain)
    assert not report.ok
    assert report.first_broadening_index == 2
    assert report.reason == "broadens:amount"


def test_single_and_empty_chains_are_trivially_clean():
    assert chain_is_attenuating([]).ok
    assert chain_is_attenuating([(Capability("amount", "le", "5"),)]).ok
