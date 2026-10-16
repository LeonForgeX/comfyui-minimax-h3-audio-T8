"""Read-only source/closure checks for explicitly audited draft adapters.

No imports or evaluation of third-party source during inspection. File names,
callable names, attributes and a matching on-disk file alone grant no exemption.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from types import FunctionType

from .patch_stack_policy import UnverifiedModelStack


def require(value, message):
    if not value:
        raise UnverifiedModelStack(message)


def source_digest(path):
    # Git's Windows EOL conversion is not an implementation change.
    return hashlib.sha256((Path(path).read_text(encoding='utf-8').rstrip() + '\n').encode()).hexdigest()


def _literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'frozenset' and not node.args and not node.keywords:
        return frozenset()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift):
        return _literal(node.left) << _literal(node.right)
    return ast.literal_eval(node)


class SourceAudit:
    def __init__(self, module, digest, *, alternate_digests=()):
        from .progressive_native_bypass_identity import _symbol_codes
        require(module is not None and type(getattr(module, '__file__', None)) is str,
                'Draft adapter source module is not loaded')
        self.module = module
        path = Path(module.__file__)
        try:
            self.digest = source_digest(path)
        except (OSError, UnicodeError):
            raise UnverifiedModelStack('Draft adapter source is unavailable') from None
        require(self.digest == digest or self.digest in alternate_digests,
                'Draft adapter source revision is not audited')
        source = path.read_bytes()
        self.tree = ast.parse(source)
        self.compiled = compile(source, str(path), 'exec', dont_inherit=True)
        self.codes = dict(_symbol_codes(self.compiled))
        self.declarations = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                key = (node.name, min([node.lineno] + [d.lineno for d in node.decorator_list]))
                self.declarations.setdefault(key, []).append(node)

    def function(self, function, symbol, *, attributes=()):
        require(type(function) is FunctionType and function.__globals__ is vars(self.module)
                and function.__code__ == self.codes.get(symbol)
                and set(vars(function)) == set(attributes),
                'Draft adapter live callable differs from its audited factory')
        declarations = self.declarations.get((function.__name__, function.__code__.co_firstlineno), [])
        require(len(declarations) == 1, 'Draft adapter source declaration is ambiguous')
        node = declarations[0]
        try:
            positional = tuple(_literal(n) for n in node.args.defaults) or None
            keyword = {a.arg: _literal(n) for a, n in zip(node.args.kwonlyargs, node.args.kw_defaults) if n is not None} or None
            require(function.__defaults__ == positional and function.__kwdefaults__ == keyword,
                    'Draft adapter callable defaults were changed')
        except (TypeError, ValueError) as error:
            if isinstance(error, UnverifiedModelStack):
                raise
            raise UnverifiedModelStack('Draft adapter callable has unbound defaults') from None
        return dict(zip(function.__code__.co_freevars,
                        (cell.cell_contents for cell in function.__closure__ or ())))

    def top_functions(self):
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and not node.decorator_list:
                self.function(getattr(self.module, node.name, None), node.name)

    def plain_class(self, cls, name, *, constants=()):
        from .progressive_native_bypass_identity import _class_metadata
        declarations = [n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == name]
        require(len(declarations) == 1 and cls is getattr(self.module, name, None)
                and cls.__bases__ == (object,), 'Draft adapter class was replaced')
        node = declarations[0]
        methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        generated = _class_metadata(cls, node, self.compiled)
        require(set(vars(cls)) == methods | generated | set(constants) | {'__module__', '__doc__', '__dict__', '__weakref__'},
                'Draft adapter class has unknown execution state')
        for name in methods:
            self.function(vars(cls)[name], cls.__name__ + '.' + name)


def module_for(function):
    return sys.modules.get(getattr(function, '__module__', None))
