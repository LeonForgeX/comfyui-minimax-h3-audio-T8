"""Pure recipe mutation guards; no pretrained inference or quality claim."""
from copy import deepcopy

import pytest

from tools.qualify_native_voice_gpu import VOICE_BINDING, without_voice_reference


def recipe():
    return {'7': dict(class_type='LoadAudio', inputs=dict(audio='reference.flac')),
            '8': dict(class_type='Conditioning', inputs={
                'ref_audios.ref_audio_0': ['7', 0], 'ref_images.ref_image_0': ['6', 0],
                'prompt': '<Subject 1> is the woman in <Picture 1>.' + VOICE_BINDING + '\n<d>[Mandarin]你终于回来了。</d>',
                'audio_mode': 'native', 'length': 73}),
            '9': dict(class_type='Sampler', inputs=dict(seed=20260918, steps=20)),
            '15': dict(class_type='CreateVideo', inputs=dict(audio=['14', 1]))}


def test_ablation_only_removes_recording_and_matching_text_binding():
    source = recipe()
    before = deepcopy(source)
    control = without_voice_reference(source)
    assert source == before
    assert '7' not in control and 'ref_audios.ref_audio_0' not in control['8']['inputs']
    assert control['8']['inputs']['prompt'] == source['8']['inputs']['prompt'].replace(VOICE_BINDING, '')
    assert control['9'] == source['9'] and control['15'] == source['15']
    assert control['8']['inputs']['ref_images.ref_image_0'] == ['6', 0]
    assert '<d>[Mandarin]你终于回来了。</d>' in control['8']['inputs']['prompt']


@pytest.mark.parametrize('fault', ['missing_reference', 'missing_binding', 'other_audio_tag', 'other_consumer'])
def test_unknown_recipe_cannot_be_silently_rewritten(fault):
    source = recipe()
    if fault == 'missing_reference':
        del source['8']['inputs']['ref_audios.ref_audio_0']
    elif fault == 'missing_binding':
        source['8']['inputs']['prompt'] = 'No known binding'
    elif fault == 'other_audio_tag':
        source['8']['inputs']['prompt'] += '<Audio 2>'
    else:
        source['15']['inputs']['audio'] = ['7', 0]
    before = deepcopy(source)
    with pytest.raises(ValueError):
        without_voice_reference(source)
    assert source == before
