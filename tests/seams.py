# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Producer keys against consumer keys at every in-process dict seam.

A producer is a function whose return value is a dict of known keys: a dict
literal, dict(k=...), a local built up by subscript stores / update /
setdefault, or the return of another producer (followed). A consumer is a
function that binds a call's result to a name and reads keys off it:
v["k"], v.get("k"), "k" in v, v.pop("k"), or hands v to a function that does
(parameter flow, followed three levels).

The check is per producer. ExternalScorer answered `decision` while the
pipeline read `action`; every other scorer wrote `action`, so a union over all
producers named `evaluate` would have passed. Here each producer is compared
alone: a key the consumer reads that this producer never writes on any path is
missing. A `.get` read is reported too, because that bug was one: the default
hid it. Findings whose producer has a path of unknown shape are left out; their
reader is usually JSON from outside the process.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Fn:
    def __init__(self, fid, mod, qual, node, cls, path):
        self.id, self.mod, self.qual, self.node, self.cls, self.path = fid, mod, qual, node, cls, path
        self.name = qual.rsplit(".", 1)[-1]
        self.params = [a.arg for a in node.args.posonlyargs + node.args.args]
        self.kwparams = [a.arg for a in node.args.kwonlyargs]


fns: dict[str, Fn] = {}
by_name: dict[str, list[Fn]] = defaultdict(list)
mod_top: dict[str, dict[str, Fn]] = defaultdict(dict)
mod_imports: dict[str, dict[str, tuple]] = defaultdict(dict)


def own_nodes(fn_node):
    """Walk a function body without descending into nested defs/classes."""
    stack = list(fn_node.body)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.append(c)


def _index(package: Path) -> None:
    for p in sorted(package.rglob("*.py")):
        rel = p.relative_to(package.parent).with_suffix("")
        mod = ".".join(rel.parts).removesuffix(".__init__")
        tree = ast.parse(p.read_text(errors="ignore"))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                base = n.module
                if n.level:
                    pkg = mod if p.name == "__init__.py" else mod.rsplit(".", 1)[0]
                    for _ in range(n.level - 1):
                        pkg = pkg.rsplit(".", 1)[0]
                    base = f"{pkg}.{n.module}"
                for a in n.names:
                    mod_imports[mod][a.asname or a.name] = (base, a.name)

        def walk(body, prefix, cls):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = prefix + n.name
                    f = Fn(f"{mod}:{q}", mod, q, n, cls, str(p.relative_to(package.parent.parent)))
                    fns[f.id] = f
                    by_name[n.name].append(f)
                    if not prefix:
                        mod_top[mod][n.name] = f
                    walk(n.body, q + ".<locals>.", None)
                elif isinstance(n, ast.ClassDef):
                    walk(n.body, prefix + n.name + ".", n.name)
        walk(tree.body, "", None)


def resolve_call(caller: Fn, call: ast.Call) -> list[Fn]:
    f = call.func
    if isinstance(f, ast.Name):
        if f.id in mod_top[caller.mod]:
            return [mod_top[caller.mod][f.id]]
        imp = mod_imports[caller.mod].get(f.id)
        if imp and imp[0] in mod_top and imp[1] in mod_top[imp[0]]:
            return [mod_top[imp[0]][imp[1]]]
        # nested helper defined in the same enclosing function
        return [g for g in by_name.get(f.id, []) if g.mod == caller.mod and "<locals>" in g.qual]
    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name) and f.value.id in ("self", "cls") and caller.cls:
            same = [g for g in by_name.get(f.attr, []) if g.mod == caller.mod and g.qual.startswith(caller.cls + ".")]
            if same:
                return same
        if isinstance(f.value, ast.Name):
            typed = [g for c in receiver_classes(caller, f.value.id)
                     for g in by_name.get(f.attr, []) if g.cls == c]
            if typed:
                return typed
        return [g for g in by_name.get(f.attr, []) if g.cls is not None or g.mod != caller.mod]
    return []


def _class_names(ann) -> set:
    """Class names an annotation can denote: X, mod.X, Optional[X], X | None, "X"."""
    out = set()
    if ann is None:
        return out
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            return _class_names(ast.parse(ann.value, mode="eval").body)
        except SyntaxError:
            return out
    if isinstance(ann, ast.Name):
        out.add(ann.id)
    elif isinstance(ann, ast.Attribute):
        out.add(ann.attr)
    elif isinstance(ann, ast.Subscript):
        out |= _class_names(ann.slice)
    elif isinstance(ann, ast.Tuple):
        for e in ann.elts:
            out |= _class_names(e)
    elif isinstance(ann, ast.BinOp):
        out |= _class_names(ann.left) | _class_names(ann.right)
    return out - {"Optional", "None", "Union", "Any", "dict", "list", "str", "int", "float", "bool"}


def receiver_classes(caller: Fn, var: str) -> set:
    a = caller.node.args
    for arg in a.posonlyargs + a.args + a.kwonlyargs:
        if arg.arg == var and arg.annotation is not None:
            return _class_names(arg.annotation)
    out = set()
    for n in own_nodes(caller.node):
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == var:
            out |= _class_names(n.annotation)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in n.targets):
            v = n.value
            if isinstance(v, ast.Call):
                fn_ = v.func
                if isinstance(fn_, ast.Name) and fn_.id[:1].isupper():
                    out.add(fn_.id)
                elif isinstance(fn_, ast.Attribute) and fn_.attr[:1].isupper():
                    out.add(fn_.attr)
    return out


# ------------------------------------------------------------- producers
_shape_cache: dict[str, tuple] = {}


def dict_keys_of(expr, fn: Fn, depth: int):
    """(keys, partial, is_dict) for an expression in fn."""
    if isinstance(expr, ast.Dict):
        keys, partial = set(), False
        for k, v in zip(expr.keys, expr.values):
            if k is None:
                ks, pt, isd = dict_keys_of(v, fn, depth)
                keys |= ks
                partial |= pt or not isd
            elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)
            else:
                partial = True
        return keys, partial, True
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "dict":
        keys = {kw.arg for kw in expr.keywords if kw.arg}
        partial = any(kw.arg is None for kw in expr.keywords) or bool(expr.args)
        return keys, partial, True
    if isinstance(expr, ast.Name):
        return local_dict_keys(expr.id, fn, depth)
    if isinstance(expr, ast.Await):
        return dict_keys_of(expr.value, fn, depth)
    if isinstance(expr, ast.IfExp):
        a = dict_keys_of(expr.body, fn, depth)
        b = dict_keys_of(expr.orelse, fn, depth)
        return a[0] | b[0], a[1] or b[1] or not (a[2] and b[2]), a[2] or b[2]
    if isinstance(expr, ast.Call) and depth < 4:
        targets = resolve_call(fn, expr)
        if len(targets) == 1:
            ks, pt, isd = shape(targets[0], depth + 1)
            return ks, pt, isd
        return set(), True, False
    return set(), True, False


def local_dict_keys(var: str, fn: Fn, depth: int):
    keys, partial, isd = set(), False, False
    for n in own_nodes(fn.node):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == var for t in targets) and n.value is not None:
                ks, pt, d = dict_keys_of(n.value, fn, depth)
                keys |= ks
                partial |= pt
                isd |= d
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == var:
                    s = t.slice
                    if isinstance(s, ast.Constant) and isinstance(s.value, str):
                        keys.add(s.value)
                    else:
                        partial = True
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == var:
            if n.func.attr == "update":
                for a in n.args:
                    ks, pt, d = dict_keys_of(a, fn, depth)
                    keys |= ks
                    partial |= pt or not d
                keys |= {kw.arg for kw in n.keywords if kw.arg}
            elif n.func.attr == "setdefault" and n.args and isinstance(n.args[0], ast.Constant):
                keys.add(n.args[0].value)
    return keys, partial, isd


def shape(fn: Fn, depth: int = 0):
    """Union of keys over every return path: (keys, partial, is_dict)."""
    if fn.id in _shape_cache:
        return _shape_cache[fn.id]
    _shape_cache[fn.id] = (set(), True, False)  # recursion guard
    keys, partial, any_dict, any_other = set(), False, False, False
    for n in own_nodes(fn.node):
        if isinstance(n, ast.Return) and n.value is not None:
            if isinstance(n.value, ast.Constant) and n.value.value is None:
                continue
            ks, pt, isd = dict_keys_of(n.value, fn, depth)
            if isd:
                any_dict = True
                keys |= ks
                partial |= pt
            else:
                any_other = True
    res = (keys, partial or any_other, any_dict)
    _shape_cache[fn.id] = res
    return res


# ------------------------------------------------------------- consumers
def reads_on(var: str, fn: Fn):
    """Keys read off `var` in fn, and calls var is handed to: (reads, handoffs, escapes)."""
    reads: dict[str, str] = {}
    handoffs: list[tuple[ast.Call, int | str]] = []
    escapes = False
    for n in own_nodes(fn.node):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == var \
                and isinstance(n.ctx, ast.Load):
            s = n.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                reads.setdefault(s.value, "[]")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == var:
            if n.func.attr in ("get", "pop") and n.args and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str):
                reads.setdefault(n.args[0].value, "." + n.func.attr)
            elif n.func.attr in ("items", "keys", "values", "copy"):
                escapes = True
        elif isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)) \
                and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) \
                and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == var:
            reads.setdefault(n.left.value, "in")
        elif isinstance(n, ast.Call):
            for i, a in enumerate(n.args):
                if isinstance(a, ast.Name) and a.id == var:
                    handoffs.append((n, i))
                elif isinstance(a, ast.Starred):
                    pass
            for kw in n.keywords:
                if isinstance(kw.value, ast.Name) and kw.value.id == var:
                    if kw.arg:
                        handoffs.append((n, kw.arg))
                    else:
                        escapes = True
        elif isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id == var:
            escapes = True
    return reads, handoffs, escapes


def writes_on(var: str, fn: Fn) -> set:
    """Keys stored on `var` inside fn (v["k"] = .., v.update(..), v.setdefault("k", ..))."""
    out = set()
    for n in own_nodes(fn.node):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == var \
                        and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                    out.add(t.slice.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and isinstance(n.func.value, ast.Name) and n.func.value.id == var:
            if n.func.attr == "update":
                for a in n.args:
                    ks, _, _ = dict_keys_of(a, fn, 0)
                    out |= ks
                out |= {kw.arg for kw in n.keywords if kw.arg}
            elif n.func.attr == "setdefault" and n.args and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str):
                out.add(n.args[0].value)
    return out


_pw_cache: dict = {}


def param_writes(fn: Fn, pos, depth=0) -> set:
    key = (fn.id, pos)
    if key in _pw_cache:
        return _pw_cache[key]
    _pw_cache[key] = set()
    params = fn.params
    if isinstance(pos, int):
        idx = pos + (1 if params and params[0] in ("self", "cls") else 0)
        if idx >= len(params):
            return set()
        name = params[idx]
    else:
        name = pos if pos in fn.params or pos in fn.kwparams else None
        if name is None:
            return set()
    out = writes_on(name, fn)
    if depth < 3:
        _, handoffs, _ = reads_on(name, fn)
        for call, p in handoffs:
            for g in resolve_call(fn, call)[:3]:
                out |= param_writes(g, p, depth + 1)
    _pw_cache[key] = out
    return out


_param_cache: dict = {}


def param_reads(fn: Fn, pos, depth=0):
    key = (fn.id, pos)
    if key in _param_cache:
        return _param_cache[key]
    _param_cache[key] = {}
    params = fn.params
    if isinstance(pos, int):
        idx = pos + (1 if params and params[0] in ("self", "cls") else 0)
        if idx >= len(params):
            return {}
        name = params[idx]
    else:
        name = pos if pos in fn.params or pos in fn.kwparams else None
        if name is None:
            return {}
    reads, handoffs, _ = reads_on(name, fn)
    out = {k: f"{v} in {fn.qual}" for k, v in reads.items()}
    if depth < 3:
        for call, p in handoffs:
            for g in resolve_call(fn, call)[:3]:
                for k, v in param_reads(g, p, depth + 1).items():
                    out.setdefault(k, v)
    _param_cache[key] = out
    return out


def analyse(pkg: Path = ROOT / "src" / "vaara") -> list[dict]:
    """Every finding under the package directory ``pkg``, one per producer and consumer site."""
    for table in (fns, by_name, mod_top, mod_imports, _shape_cache, _pw_cache, _param_cache):
        table.clear()
    _index(pkg)
    rows = []
    for fn in fns.values():
        bound: dict[str, list[ast.Call]] = defaultdict(list)
        for n in own_nodes(fn.node):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                v = n.value
                if isinstance(v, ast.Await):
                    v = v.value
                if isinstance(v, ast.BoolOp) and isinstance(v.values[0], ast.Call):
                    v = v.values[0]
                if isinstance(v, ast.Call):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        if isinstance(t, ast.Name):
                            bound[t.id].append(v)
            # direct: call(...)["k"] / call(...).get("k")
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Call) and isinstance(n.slice, ast.Constant) \
                    and isinstance(n.slice.value, str):
                for p in resolve_call(fn, n.value):
                    ks, pt, isd = shape(p)
                    if isd and not pt and n.slice.value not in ks:
                        rows.append({"producer": p.id, "producer_at": f"{p.path}:{p.node.lineno}",
                                     "consumer": fn.id, "consumer_at": f"{fn.path}:{n.lineno}",
                                     "missing": [n.slice.value], "read": {n.slice.value: "[] direct"},
                                     "produced": sorted(ks)})
        for var, calls in bound.items():
            reads, handoffs, escapes = reads_on(var, fn)
            written = writes_on(var, fn)
            for call, pos in handoffs:
                for g in resolve_call(fn, call)[:3]:
                    for k, how in param_reads(g, pos).items():
                        reads.setdefault(k, how)
                    written |= param_writes(g, pos)
            if not reads:
                continue
            for call in calls:
                for p in resolve_call(fn, call):
                    ks, pt, isd = shape(p)
                    if not isd:
                        continue
                    missing = sorted(k for k in reads if k not in ks and k not in written)
                    rows.append({"producer": p.id, "producer_at": f"{p.path}:{p.node.lineno}",
                                 "consumer": fn.id, "consumer_at": f"{fn.path}:{call.lineno}", "var": var,
                                 "partial_producer": pt, "missing": missing,
                                 "read": reads, "produced": sorted(ks), "escapes": escapes})

    # Group by consumer site; a by-name resolution yields several candidate
    # producers, and only those sharing a key with what the consumer reads are
    # plausibly the one that answers there.
    sites = defaultdict(list)
    for r in rows:
        sites[(r["consumer"], r["consumer_at"], r.get("var"))].append(r)
    findings = []
    for (consumer, at, var), rs in sites.items():
        multi = len(rs) > 1
        for r in rs:
            if not r["missing"]:
                continue
            reads = set(r["read"])
            overlap = reads & set(r["produced"])
            if multi and not overlap:
                continue
            others = [o for o in rs if o is not r and (set(o["read"]) & set(o["produced"]))]
            cls = []
            for k in r["missing"]:
                if any(k in o["produced"] for o in others):
                    cls.append((k, "SIBLING-HAS-IT"))
                else:
                    cls.append((k, "NOBODY-WRITES"))
            findings.append({**r, "classes": cls, "candidates": len(rs),
                             "producer_known": not r.get("partial_producer")})
    return findings
