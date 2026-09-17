"""VDN owns its block math; a dormant upstream override is not a conflict."""
import pytest

from h3_audio_t8_pkg import vdn_h3_advanced as vdn
from h3_audio_t8_pkg.vdn_attention_compat import prepare_vdn_attention_model
from test_vdn_attention_compat import model_fixture


def override(*args, **kwargs):
    raise AssertionError('VDN must not dispatch its learned branch through this override')


def configured_options(model):
    hooks = (lambda *a: None, lambda *a: None)
    options = model.model_options['transformer_options']
    options[vdn.OWNER_HOOKS_KEY] = hooks
    for index, hook in enumerate(hooks):
        model.set_model_patch_replace(hook, 'dit', 'double_block', index)
    return model.model_options['transformer_options']


def test_upstream_override_preserved_and_not_a_composition_conflict():
    source = model_fixture()
    source.model_options['transformer_options']['optimized_attention_override'] = override
    adapted, removed = prepare_vdn_attention_model(source)
    assert removed == 0 and adapted is source
    assert vdn._attention_conflicts(adapted) == []
    assert source.model_options['transformer_options']['optimized_attention_override'] is override


def test_runtime_override_does_not_block_owned_vdn_blocks():
    options = configured_options(model_fixture())
    options['optimized_attention_override'] = override
    vdn.validate_vdn_runtime_options(options)
    assert options['optimized_attention_override'] is override


def test_later_block_and_attention_hooks_are_preserved_with_advisories(caplog):
    model = model_fixture()
    options = configured_options(model)
    options['optimized_attention_override'] = override
    later = lambda *a: 'later-block'
    options['patches_replace']['dit'][('double_block', 0)] = later
    vdn.validate_vdn_runtime_options(options)
    assert options['patches_replace']['dit'][('double_block', 0)] is later
    assert later() == 'later-block'
    options['patches_replace']['dit'][('double_block', 0)] = options[vdn.OWNER_HOOKS_KEY][0]
    hook = lambda *a: 'later-attention'
    options['patches'] = {'attn1_patch': [hook]}
    vdn.validate_vdn_runtime_options(options)
    assert options['patches']['attn1_patch'] == [hook]
    assert hook() == 'later-attention'
    assert 'continuing' in caplog.text
    with pytest.raises(RuntimeError, match='ownership is missing'):
        vdn.validate_vdn_runtime_options({})
