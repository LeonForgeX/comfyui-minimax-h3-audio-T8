"""Keep audited KJ AST fingerprints identical across CPython 3.12 / 3.13."""
import ast
import hashlib
import inspect
from pathlib import Path
import sys

import pytest
from comfy.ldm.modules import attention

from h3_audio_t8_pkg.relay_kj_backend import _ast_hash, _KJ_AST
from h3_audio_t8_pkg.relay_kj_memory import _SAGE_FORWARD_AST


@pytest.mark.parametrize('filename,symbol,digest', [
    ('ltxv_nodes.py', 'minimax_sageattn_forward', _SAGE_FORWARD_AST),
    *[('model_optimization_nodes.py', name, digest) for name, digest in _KJ_AST.items()],
])
def test_reviewed_fingerprint_retains_empty_ast_fields(filename, symbol, digest):
    root = Path(inspect.getsourcefile(attention)).parents[3] / 'custom_nodes/ComfyUI-KJNodes/nodes'
    path = root / filename
    if not path.is_file():
        pytest.skip('actual audited KJ sources required')
    node = next(n for n in ast.parse(path.read_bytes()).body if getattr(n, 'name', None) == symbol)
    assert _ast_hash(node) == digest
    if sys.version_info >= (3, 13):
        # This is the actual old implementation and the user's false decline.
        assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() != digest
    # The fix preserves empty fields; it must not ignore real implementation changes.
    node.body.append(ast.Return(value=ast.Constant(value='changed execution')))
    assert _ast_hash(node) != digest
