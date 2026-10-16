"""Read-only native INT8 serialization proof for synchronous draft boundaries.

Only source-audited Core/Kitchen readers qualify. This does not cache weights,
dequantize tensors, alter the model, or authenticate a sampling implementation.
Unknown serializers retain the original independent identity captures.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType

import torch

from .relay_kj_backend import _ast_hash, _codes


# Core 8010b48 + Kitchen's native TensorWiseINT8 serializer. AST representation
# includes empty fields on Python 3.13; changed source must be reviewed again.
_READERS = {
    'mixed_precision_ops.<locals>.MixedPrecisionOps.Linear.state_dict':
        'bfa6d28caacb47b8284f97f5b4d15fee512ca3a32932f65a1a1baf8450386279',
    '_quantized_weight_state_dict': 'c3ed3298b2a9875bc9f711546e076bd83b331de6563187d7ecb4bdd0ecc0cfeb',
    'QuantizedTensor.state_dict': '1cb576fbc4ea439699e413ca4a13d1aa29b45809ec58f19374e618916fd7547e',
    'QuantizedTensor.layout_cls': '9e7586fc8a4bf150626873299cb504efb756ac9d2ec3fc0312a94efe15fbf5d0',
    'QuantizedTensor.params': '594ff2628658fa62bbaa8fe8e3ab6bd447ff3ef570b1d4ed4b83f2b99724a7fb',
    'get_layout_class': '9447e92aa6228197e7bc75d4429f2ba7e39785333711cef0e2a76847786e9e82',
    'TensorWiseINT8Layout.state_dict_tensors':
        'db6d5772bef5a89a3719e2834c6032285dc180580e9fd4edd65163601d69b94b',
}


def _plain_tensor(value):
    return type(value) in (torch.Tensor, torch.nn.Parameter) and not vars(value)


class NativeINT8Readers:
    def __init__(self):
        self.sources = {}
        self.declarations = {}

    def _function(self, function, symbol):
        if (type(function) is not FunctionType or function.__qualname__ != symbol
                or function.__closure__):
            return False
        module = sys.modules.get(function.__module__)
        if module is None or function.__globals__ is not vars(module):
            return False
        path = inspect.getsourcefile(function)
        if not path:
            return False
        if path not in self.sources:
            data = Path(path).read_bytes()
            nodes = {min([node.lineno] + [d.lineno for d in node.decorator_list]): node
                     for node in ast.walk(ast.parse(data)) if isinstance(node, ast.FunctionDef)}
            codes = {code.co_qualname: code for code in _codes(compile(data, path, 'exec', dont_inherit=True))}
            self.sources[path] = nodes, codes
        key = path, symbol, function.__code__.co_firstlineno
        if key not in self.declarations:
            nodes, codes = self.sources[path]
            node = nodes.get(key[2])
            if node is None or _ast_hash(node) != _READERS[symbol]:
                return False
            defaults = tuple(ast.literal_eval(x) for x in node.args.defaults) or None
            keywords = {arg.arg: ast.literal_eval(x) for arg, x in
                        zip(node.args.kwonlyargs, node.args.kw_defaults) if x is not None} or None
            self.declarations[key] = codes.get(symbol), defaults, keywords
        code, defaults, keywords = self.declarations[key]
        return (function.__code__ == code and function.__defaults__ == defaults
                and function.__kwdefaults__ == keywords)

    def _owners(self):
        from comfy import ops
        import comfy_kitchen.tensor.base as base
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
        if (ops.QuantizedTensor is not QuantizedTensor or ops.json is not json
                or type(base.LAYOUTS) is not dict
                or base.LAYOUTS.get('TensorWiseINT8Layout') is not TensorWiseINT8Layout):
            return False
        fields = [(ops._quantized_weight_state_dict, '_quantized_weight_state_dict'),
                  (inspect.getattr_static(QuantizedTensor, 'state_dict'), 'QuantizedTensor.state_dict'),
                  (inspect.getattr_static(base, 'get_layout_class'), 'get_layout_class')]
        for name in ('layout_cls', 'params'):
            descriptor = inspect.getattr_static(QuantizedTensor, name)
            if type(descriptor) is not property:
                return False
            fields.append((descriptor.fget, 'QuantizedTensor.' + name))
        descriptor = inspect.getattr_static(TensorWiseINT8Layout, 'state_dict_tensors')
        if type(descriptor) is not classmethod:
            return False
        fields.append((descriptor.__func__, 'TensorWiseINT8Layout.state_dict_tensors'))
        return all(self._function(function, symbol) for function, symbol in fields)

    def tensor(self, value):
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
        if _plain_tensor(value) or value is None:
            return True
        if type(value) is not QuantizedTensor:
            return False
        members = vars(value)
        if (set(members) not in ({'_qdata', '_layout_cls', '_params'},
                                {'_qdata', '_layout_cls', '_params', '_is_param'})
                or members.get('_is_param', True) is not True
                or type(members['_layout_cls']) is not str
                or members['_layout_cls'] != 'TensorWiseINT8Layout'):
            return False
        raw, params = members['_qdata'], members['_params']
        if (not _plain_tensor(raw) or raw.dtype != torch.int8
                or type(params) is not TensorWiseINT8Layout.Params
                or type(params).__getattribute__ is not object.__getattribute__):
            return False
        values = vars(params)
        return (set(values) == {'scale', 'orig_dtype', 'orig_shape', 'is_weight',
                               'convrot', 'convrot_groupsize', 'transposed'}
                and _plain_tensor(values['scale'])
                and type(values['orig_dtype']) is torch.dtype
                and values['orig_dtype'] in (torch.float32, torch.float16, torch.bfloat16)
                and type(values['orig_shape']) in (tuple, torch.Size)
                and all(type(x) is int for x in values['orig_shape'])
                and values['is_weight'] is True and values['transposed'] is False
                and type(values['convrot']) is bool and type(values['convrot_groupsize']) is int
                and self._owners())

    def module(self, module):
        try:
            from comfy import ops
            if (type(module).__module__ != ops.__name__
                    or inspect.getattr_static(module, '__getattr__') is not torch.nn.Module.__getattr__
                    or not self._function(inspect.getattr_static(module, 'state_dict'),
                        'mixed_precision_ops.<locals>.MixedPrecisionOps.Linear.state_dict')
                    or not self._owners()):
                return False
            # The audited reader accesses only these scalar fields and native
            # tensor parameters/buffers; descriptors or extra scale readers fail.
            for name in ('quant_format', '_full_precision_mm_config'):
                value = inspect.getattr_static(module, name, None)
                if type(value) not in (str, bool, type(None)):
                    return False
            if module.quant_format not in (None, 'int8_tensorwise'):
                return False
            members = vars(module)
            for name in ('weight', 'bias', 'input_scale', 'pre_quant_scale'):
                value = (members.get('_parameters', {}).get(name,
                         members.get('_buffers', {}).get(name,
                         inspect.getattr_static(module, name, None))))
                if not self.tensor(value):
                    return False
            return True
        except (AttributeError, ImportError, OSError, SyntaxError, TypeError, ValueError):
            return False
