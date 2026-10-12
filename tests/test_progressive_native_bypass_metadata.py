"""Python 3.13 compiler metadata simulated on real Core under Python 3.12.

The source/bytecode authentication path uses the current interpreter. These CPU
tests do not claim an actual Python 3.13 or Windows/GPU execution.
"""
import ast
from pathlib import Path

import comfy.samplers
import pytest

from h3_audio_t8_pkg import progressive_native_bypass_identity as native
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.progressive_first_pass_draft import low_model_identity
from test_progressive_first_pass_draft import bypass_model, run, same
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401


def _classes():
    from comfy import patcher_extension
    from comfy.weight_adapter import base, bypass, lora
    return ((bypass, bypass.BypassForwardHook), (bypass, bypass.BypassInjectionManager),
            (lora, lora.LoRAAdapter), (base, base.WeightAdapterBase),
            (patcher_extension, patcher_extension.PatcherInjection))


def _install_metadata(monkeypatch):
    for module, cls in _classes():
        tree = ast.parse(Path(module.__file__).read_bytes())
        declaration, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__]
        attributes = {node.attr for function in declaration.body
                      if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                      for node in ast.walk(function)
                      if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                      and isinstance(node.value, ast.Name) and node.value.id == 'self'}
        monkeypatch.setattr(cls, '__firstlineno__', declaration.lineno, raising=False)
        monkeypatch.setattr(cls, '__static_attributes__', tuple(sorted(attributes)), raising=False)


def test_source_bound_python313_class_metadata_keeps_native_identity_portable(monkeypatch):
    before = native._sources()
    sampler = comfy.samplers.ksampler('euler')
    original = low_model_identity(bypass_model()[0], sampler)
    _install_metadata(monkeypatch)
    assert native._sources() == before
    current = low_model_identity(bypass_model()[0], sampler)
    assert original == current
    assert current['weights'].get('portable_cache_reuse') is not False


@pytest.mark.parametrize('change', ['wrong_line', 'line_type', 'static_list', 'static_extra',
                                  'static_missing', 'static_duplicate', 'missing_field', 'foreign_member'])
def test_forged_or_foreign_class_metadata_cannot_grant_native_identity(monkeypatch, change):
    _install_metadata(monkeypatch)
    cls = _classes()[0][1]
    if change == 'wrong_line':
        monkeypatch.setattr(cls, '__firstlineno__', cls.__firstlineno__ + 1)
    elif change == 'line_type':
        monkeypatch.setattr(cls, '__firstlineno__', str(cls.__firstlineno__))
    elif change == 'static_list':
        monkeypatch.setattr(cls, '__static_attributes__', list(cls.__static_attributes__))
    elif change == 'static_extra':
        monkeypatch.setattr(cls, '__static_attributes__', cls.__static_attributes__ + ('foreign',))
    elif change == 'static_missing':
        monkeypatch.setattr(cls, '__static_attributes__', cls.__static_attributes__[1:])
    elif change == 'static_duplicate':
        monkeypatch.setattr(cls, '__static_attributes__', cls.__static_attributes__ * 2)
    elif change == 'missing_field':
        monkeypatch.delattr(cls, '__firstlineno__')
    else:
        monkeypatch.setattr(cls, 'foreign_execution', lambda value: None, raising=False)
    with pytest.raises(UnverifiedModelStack):
        native._sources()
    result = low_model_identity(bypass_model()[0], comfy.samplers.ksampler('euler'))
    assert result['weights']['portable_cache_reuse'] is False
    assert result['weights']['invalid_reason_code'] == (
        'native_core_class_state' if change == 'foreign_member' else 'native_class_metadata_unverified')


def test_current_compiler_class_body_constants_override_ast_fallback():
    source = "class MetadataFixture:\n __firstlineno__ = 7\n __static_attributes__ = ('bound',)\n"
    declaration = ast.parse(source).body[0]
    compiled = compile(source, '<metadata-fixture>', 'exec', dont_inherit=True)
    cls = type('MetadataFixture', (), {'__firstlineno__': 7, '__static_attributes__': ('bound',)})
    assert native._class_metadata(cls, declaration, compiled) == {'__firstlineno__', '__static_attributes__'}
    cls.__static_attributes__ = ()
    with pytest.raises(UnverifiedModelStack, match='compiler class metadata'):
        native._class_metadata(cls, declaration, compiled)
    absent = type('MetadataFixture', (), {})
    with pytest.raises(UnverifiedModelStack, match='compiler class metadata'):
        native._class_metadata(absent, declaration, compiled)


@pytest.mark.parametrize('shared', [False, True])
def test_native_python313_metadata_saves_low_and_retries_high_bitwise(
        tmp_path, monkeypatch, stub_lifter, shared):  # noqa: F811
    _install_metadata(monkeypatch)
    def models(strength):
        low = bypass_model(strength=.3)[0]
        high = low.clone() if shared else bypass_model(strength=strength)[0]
        if shared:
            high = bypass_model(high, strength=strength)[0]
        return dict(model=low, high=high)
    first, saved = run(tmp_path, **models(.7))
    replay, report = run(tmp_path, mode='reuse_only', high_seed=97, **models(.9))
    baseline, _ = run(high_seed=97, **models(.9))
    same(replay, baseline)
    assert saved['first_pass_draft']['saved_low']
    assert report['first_pass_draft']['reused_low']
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    assert report['noise']['high_seed'] == 97
    assert first['samples'].is_nested


def test_decline_codes_do_not_retain_private_error_details():
    assert native._decline_code(UnverifiedModelStack('private path/token payload')) == 'model_stack_unverified'
