# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which functions and classes under src/vaara a user can reach, and from where.

A name count says a symbol is referenced somewhere. It does not say the
reference is itself reachable, and two dead symbols that name each other pass
it: ``ExternalScorer`` and ``CompositeScorer`` did for five months. Here a
definition counts only through a chain that starts at a door a user has:

- a console script in pyproject, a module run with ``python -m``, a script in
  ``scripts/``, or a Claude Code plugin hook script;
- a name a package exports in ``__all__``, or a module a document names
  declares in its own;
- an import a document shows a reader: README.md, docs/, examples/.

Resolution is exact where Python allows it (module bindings, import chains,
attribute chains on modules and classes) and generous where it cannot be
(``obj.name`` on an object of unknown type reaches every method called
``name``). Generosity can only hide dead code, never invent it, so whatever
this reports unreachable has no door.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Decorators that do not register the function anywhere.
_PLAIN_DECORATORS = {
    "staticmethod", "classmethod", "property", "setter", "getter", "deleter",
    "cached_property", "lru_cache", "cache", "wraps", "dataclass",
    "total_ordering", "contextmanager", "asynccontextmanager", "abstractmethod",
    "overload", "runtime_checkable", "final", "override", "unique",
}
#: Methods a framework calls on a subclass of one of its classes.
_CALLBACK = re.compile(
    r"^(do_[A-Z]+|on_\w+|https?_(open|request|response)|http_error_\w+|redirect_request"
    r"|emit|default|run|_run|_arun|handle|setup|finish|log_message|log_request|log_error"
    r"|flush|close|encode|iterencode|format|filter|connection_made|data_received"
    r"|connection_lost|eof_received|model_post_init|validate|dispatch)$"
)
_MODULE_STRING = re.compile(r"^vaara(\.[A-Za-z_]\w*)+(:[A-Za-z_][\w.]*)?$")
#: One import statement: a parenthesised list may span lines, a bare list
#: ends with its line. A greedy class that crossed newlines swallowed the
#: `from vaara` of the next statement, so every second import a document
#: showed was not a door.
_DOC_IMPORT = re.compile(r"from\s+(vaara[\w.]*)\s+import\s+(?:\(([^)]*)\)|([^\n]+))")
_DOC_DOTTED = re.compile(r"\b(vaara(?:\.[A-Za-z_]\w*)+)")


def _imported_names(clause: str) -> list[str]:
    """The names an import clause binds: `a, b as c  # note` gives a and b."""
    names = []
    for part in re.sub(r"#.*", "", clause).split(","):
        words = part.split()
        if words:
            names.append(words[0])
    return names


def _declared_exports(statement) -> list[str]:
    """Names a statement adds to ``__all__``: the assignment, ``+=``, or
    ``__all__.extend([...])`` under an ``if`` that guards an optional extra.
    A package that exports a name only when its extra is installed still
    exports it."""
    names: list[str] = []
    for node in ast.walk(statement):
        value = None
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            value = node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "__all__":
            value = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "extend" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "__all__" and node.args:
            value = node.args[0]
        if value is not None:
            try:
                names += [n for n in ast.literal_eval(value) if isinstance(n, str)]
            except ValueError:
                pass
    return names


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return None


class _Body:
    """What one definition, or a module's top level, refers to."""

    def __init__(self):
        self.names, self.chains, self.attrs = set(), set(), set()
        self.imports, self.module_strings, self.strings, self.prefixes = [], set(), set(), set()


class _Collector(ast.NodeVisitor):
    def __init__(self, body, skip_all=False):
        self.b, self.skip_all = body, skip_all

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load):
            self.b.names.add(n.id)

    def visit_Attribute(self, n):
        c = _chain(n)
        if c:
            self.b.chains.add(c)
        self.b.attrs.add(n.attr)
        self.generic_visit(n)

    def visit_Import(self, n):
        self.b.imports += [(a.name, None, 0, a.asname) for a in n.names]

    def visit_ImportFrom(self, n):
        self.b.imports += [(n.module or "", a.name, n.level, a.asname) for a in n.names]

    def visit_Constant(self, n):
        if isinstance(n.value, str):
            if _MODULE_STRING.match(n.value):
                self.b.module_strings.add(n.value)
            elif re.fullmatch(r"[A-Za-z_]\w*", n.value):
                self.b.strings.add(n.value)

    def visit_JoinedStr(self, n):
        first = n.values[0] if n.values else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str) \
                and re.fullmatch(r"[A-Za-z_]\w*", first.value):
            self.b.prefixes.add(first.value)
        self.generic_visit(n)

    def visit_Assign(self, n):
        if self.skip_all and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
            return
        self.generic_visit(n)

    def visit_Expr(self, n):
        if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            return
        self.generic_visit(n)


class Definition:
    def __init__(self, module, qual, kind, node, cls, path):
        self.module, self.qual, self.kind, self.node, self.cls, self.path = module, qual, kind, node, cls, path
        self.id = f"{module}:{qual}"
        self.name = qual.rsplit(".", 1)[-1]
        self.body = _Body()
        self.registered = False
        self.bases = []
        self.external_base = False


class _Module:
    def __init__(self, name, path, is_package):
        self.name, self.path, self.is_package = name, path, is_package
        self.init = _Body()
        self.top = {}
        self.bindings = defaultdict(list)
        self.runnable = False
        self.exports = []


class Graph:
    """Parse once, answer reachability from the doors listed in the module doc."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.modules: dict[str, _Module] = {}
        self.defs: dict[str, Definition] = {}
        self.methods = defaultdict(list)
        for p in sorted((root / "src" / "vaara").rglob("*.py")):
            self._load(p, *self._modname(p, root / "src"))
        for p in sorted((root / "plugins").rglob("*.py")):
            self._load(p, "plugin::" + self._modname(p, root / "plugins")[0], False)
        for p in sorted((root / "scripts").rglob("*.py")):
            self._load(p, "script::" + self._modname(p, root / "scripts")[0], False)

    # -------------------------------------------------------------- parsing
    @staticmethod
    def _modname(path, base):
        parts = list(path.relative_to(base).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        return ".".join(parts[:-1] if is_package else parts), is_package

    def _bind(self, m, name, binding):
        if binding not in m.bindings[name]:
            m.bindings[name].append(binding)

    def _absolute(self, m, module, level):
        if not level:
            return module
        package = m.name if m.is_package else m.name.rsplit(".", 1)[0]
        for _ in range(level - 1):
            package = package.rsplit(".", 1)[0]
        return f"{package}.{module}" if module else package

    def _bind_import(self, m, module, name, level, alias):
        target = self._absolute(m, module, level)
        if name is None:
            if alias:
                self._bind(m, alias, ("module", target))
            else:
                self._bind(m, target.split(".")[0], ("module", target.split(".")[0]))
        elif name == "*":
            self._bind(m, "*" + target, ("star", target))
        else:
            self._bind(m, alias or name, ("from", target, name))

    def _definition(self, m, node, qual, cls):
        kind = "class" if isinstance(node, ast.ClassDef) else ("method" if cls else "function")
        d = Definition(m.name, qual, kind, node, cls, m.path)
        self.defs[d.id] = d
        collect = _Collector(d.body)
        for decorator in node.decorator_list:
            collect.visit(decorator)
            target = _chain(decorator.func if isinstance(decorator, ast.Call) else decorator)
            if not target or target[-1] not in _PLAIN_DECORATORS:
                d.registered = True
        if isinstance(node, ast.ClassDef):
            for base in node.bases + [k.value for k in node.keywords]:
                collect.visit(base)
                d.bases.append(_chain(base))
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.methods[statement.name].append(
                        self._definition(m, statement, f"{qual}.{statement.name}", d))
                elif isinstance(statement, ast.ClassDef):
                    self._definition(m, statement, f"{qual}.{statement.name}", d)
                else:
                    collect.visit(statement)
        else:
            args = node.args
            for default in args.defaults + [k for k in args.kw_defaults if k is not None]:
                collect.visit(default)
            for statement in node.body:
                collect.visit(statement)
            for arg in args.args + args.kwonlyargs + args.posonlyargs + [a for a in (args.vararg, args.kwarg) if a]:
                if arg.annotation is not None:
                    collect.visit(arg.annotation)
            if node.returns is not None:
                collect.visit(node.returns)
        return d

    def _load(self, path, name, is_package):
        m = _Module(name, path, is_package)
        self.modules[name] = m
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        collect = _Collector(m.init, skip_all=True)
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                d = self._definition(m, statement, statement.name, None)
                m.top[statement.name] = d
                self._bind(m, statement.name, ("def", d.id))
                continue
            if isinstance(statement, ast.If) and "__main__" in ast.unparse(statement.test):
                m.runnable = True
            m.exports += _declared_exports(statement)
            collect.visit(statement)
            for node in ast.walk(statement):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        c = _chain(node.value) if isinstance(node.value, (ast.Name, ast.Attribute)) else None
                        if isinstance(target, ast.Name) and c:
                            self._bind(m, target.id, ("alias", c))
        for imp in m.init.imports:
            self._bind_import(m, *imp)

    # -------------------------------------------------------------- resolving
    def resolve_in(self, module, name, depth=0):
        m = self.modules.get(module)
        if m is None or depth > 20:
            return []
        bindings = m.bindings.get(name) or []
        # An explicit binding wins over a same-named submodule: `from
        # vaara.govern import govern` in vaara/__init__ rebinds vaara.govern.
        if not bindings and f"{module}.{name}" in self.modules:
            return [("module", f"{module}.{name}")]
        if not bindings:
            for key, values in list(m.bindings.items()):
                for value in values:
                    if value[0] == "star":
                        found = self.resolve_in(value[1], name, depth + 1)
                        if found:
                            return found
            return []
        out = []
        for binding in bindings:
            out += self._resolve_binding(m, binding, depth + 1)
        return out

    def _resolve_binding(self, m, binding, depth):
        kind = binding[0]
        if kind == "def":
            return [("def", binding[1])]
        if kind == "module":
            return [("module", binding[1])] if binding[1] in self.modules else []
        if kind == "from":
            return self.resolve_in(binding[1], binding[2], depth + 1)
        if kind == "alias":
            return self._resolve_chain(m, binding[1], depth + 1)
        return []

    def _resolve_chain(self, m, chain, depth=0):
        if depth > 20:
            return []
        current = []
        for binding in m.bindings.get(chain[0]) or []:
            current += self._resolve_binding(m, binding, depth + 1)
        for part in chain[1:]:
            following = []
            for kind, target in current:
                if kind == "module":
                    following += self.resolve_in(target, part, depth + 1)
                elif kind == "def" and self.defs[target].kind == "class" and f"{target}.{part}" in self.defs:
                    following.append(("def", f"{target}.{part}"))
            current = following
            if not current:
                return []
        return current

    # -------------------------------------------------------------- doors
    def _documents(self):
        return [self.root / "README.md"] + sorted((self.root / "docs").rglob("*.md")) \
            + sorted((self.root / "examples").rglob("*.py")) + sorted((self.root / "examples").rglob("*.md"))

    def documented_calls(self) -> set[str]:
        """Attribute names the documented Python calls: `trail.record_key_lifecycle(...)`.

        A method is a door when a document shows a reader calling it on an
        object the document imported. Code blocks that do not parse (shell
        mixed in, ellipses for elided arguments) are read line by line.
        """
        names = set()
        for document in self._documents():
            text = document.read_text(encoding="utf-8", errors="replace")
            blocks = [text] if document.suffix == ".py" else re.findall(r"```python\n(.*?)```", text, re.S)
            for block in blocks:
                try:
                    trees = [ast.parse(block)]
                except SyntaxError:
                    trees = []
                    for line in block.splitlines():
                        try:
                            trees.append(ast.parse(line.strip()))
                        except SyntaxError:
                            continue
                for tree in trees:
                    names.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute))
        return names

    def doors(self):
        """(definition ids, module names) a user starts from."""
        defs, modules = set(), set()
        project = tomllib.loads((self.root / "pyproject.toml").read_text(encoding="utf-8"))
        for target in project["project"]["scripts"].values():
            module, function = target.split(":")
            defs.add(f"{module}:{function}")
        written = "\n".join(d.read_text(encoding="utf-8", errors="replace")
                            for d in self._documents() + [self.root / "SPEC.md"])
        for name, m in self.modules.items():
            if m.runnable or name.startswith(("plugin::", "script::")):
                modules.add(name)
            # A module's __all__ is a door when a document names the module:
            # declared API nobody is told about is not a way in.
            path = str(m.path.relative_to(self.root / "src" / "vaara")) if name.startswith("vaara") else ""
            if m.exports and name.startswith("vaara") and (
                    m.is_package or name in written or path in written):
                for exported in m.exports:
                    for kind, target in self.resolve_in(name, exported):
                        (defs if kind == "def" else modules).add(target)
        for document in self._documents():
            text = document.read_text(encoding="utf-8", errors="replace")
            for match in _DOC_IMPORT.finditer(text):
                for name in _imported_names(match.group(2) or match.group(3) or ""):
                    for kind, target in self.resolve_in(match.group(1), name):
                        (defs if kind == "def" else modules).add(target)
            for match in _DOC_DOTTED.finditer(text):
                parts = match.group(1).split(".")
                for i in range(len(parts), 1, -1):
                    module = ".".join(parts[:i])
                    if module in self.modules:
                        modules.add(module)
                        if i < len(parts):
                            for kind, target in self.resolve_in(module, parts[i]):
                                (defs if kind == "def" else modules).add(target)
                        break
        return defs, modules

    # -------------------------------------------------------------- reach
    def reachable(self) -> set[str]:
        start_defs, start_modules = self.doors()
        reached, loaded, dynamic = set(), set(), set()
        attrs, prefixes = set(self.documented_calls()), set()
        work_defs = [d for d in start_defs if d in self.defs]
        reached.update(work_defs)
        work_modules = list(start_modules)

        def load(module, dynamic_load=False):
            parts = module.split(".")
            for i in range(1, len(parts) + 1):
                sub = ".".join(parts[:i])
                if sub in self.modules and sub not in loaded:
                    loaded.add(sub)
                    work_modules.append(sub)
            if dynamic_load and module in self.modules:
                dynamic.add(module)

        def reach(def_id):
            if def_id in self.defs and def_id not in reached:
                reached.add(def_id)
                work_defs.append(def_id)

        def take(results):
            for kind, target in results:
                (reach if kind == "def" else load)(target)

        def walk(m, body):
            for module, name, level, alias in body.imports:
                target = self._absolute(m, module, level)
                load(target)
                if name and name != "*" and f"{target}.{name}" in self.modules:
                    load(f"{target}.{name}")
                self._bind_import(m, module, name, level, alias)
            for name in body.names:
                for binding in m.bindings.get(name) or []:
                    take(self._resolve_binding(m, binding, 0))
            for chain in body.chains:
                for i in range(1, len(chain) + 1):
                    take(self._resolve_chain(m, chain[:i]))
            attrs.update(body.attrs)
            prefixes.update(body.prefixes)
            for string in body.module_strings:
                module, _, attr = string.partition(":")
                load(module, dynamic_load=True)
                if attr:
                    take(self.resolve_in(module, attr.split(".")[0]))
                for plain in body.strings:
                    if module in self.modules and plain in self.modules[module].top:
                        reach(self.modules[module].top[plain].id)
            attrs.update(s for s in body.strings if s in self.methods)

        changed = True
        while changed:
            changed = False
            while work_modules or work_defs:
                while work_modules:
                    m = self.modules[work_modules.pop()]
                    walk(m, m.init)
                    for d in m.top.values():
                        if d.registered:
                            reach(d.id)
                while work_defs:
                    d = self.defs[work_defs.pop()]
                    m = self.modules[d.module]
                    load(d.module)
                    walk(m, d.body)
                    if d.kind == "class":
                        for base in d.bases:
                            found = [r for r in (self._resolve_chain(m, base) if base else []) if r[0] == "def"]
                            take(found)
                            if base and not found:
                                d.external_base = True
                        for other in self.defs.values():
                            if other.cls is d and other.kind == "class":
                                reach(other.id)
            for name, methods in self.methods.items():
                for d in methods:
                    if d.id in reached or d.cls is None or d.cls.id not in reached:
                        continue
                    if (name.startswith("__") and name.endswith("__")) or name in attrs or d.registered \
                            or (d.cls.external_base and _CALLBACK.match(name)) \
                            or any(name.startswith(p) for p in prefixes):
                        reach(d.id)
                        changed = True
            for module in list(dynamic):
                for name, d in self.modules[module].top.items():
                    if d.id not in reached and name in attrs:
                        reach(d.id)
                        changed = True
            if work_defs or work_modules:
                changed = True
        return reached

    def unreachable(self) -> list[Definition]:
        reached = self.reachable()
        return [d for d in self.defs.values() if d.module.startswith("vaara") and d.id not in reached]
