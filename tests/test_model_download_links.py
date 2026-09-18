"""Download hints are metadata only; checkpoint choice and defaults stay stable."""
import json

from h3_audio_t8_pkg.nodes_meridian import MiniMaxH3MeridianConfigEXPT8
from h3_audio_t8_pkg.nodes_taeh3_preview import MiniMaxH3TAEH3SamplingPreviewEXPT8


def test_temporal_and_2d_download_hint_is_available_in_actual_core_schema():
    info = MiniMaxH3TAEH3SamplingPreviewEXPT8.GET_NODE_INFO_V1()
    assert 'https://huggingface.co/t8star/Taeh3-Comfy' in info['description']
    required = info['input']['required']
    checkpoint = required['checkpoint'][1]
    assert 'https://huggingface.co/t8star/Taeh3-Comfy' in checkpoint['tooltip']
    assert 'taeh3_2d_kijai.safetensors' in checkpoint['tooltip']
    assert required['enabled'][1]['default'] is True
    assert required['update_every_steps'][1]['default'] == 2
    assert required['min_interval_ms'][1]['default'] == 500
    assert required['max_resolution'][1]['default'] == 256
    assert required['frames'][1]['default'] == 12


def test_meridian_and_original_omega_share_mirror_but_not_format_or_license():
    info = MiniMaxH3MeridianConfigEXPT8.GET_NODE_INFO_V1()
    required = info['input']['required']
    assert 'https://huggingface.co/t8star/Meridian-Comfy' in info['description']
    for field in ('model_path', 'omega_checkpoint'):
        assert 'https://huggingface.co/t8star/Meridian-Comfy' in required[field][1]['tooltip']
        assert required[field][1]['default'] == ''
    omega = required['omega_checkpoint'][1]['tooltip']
    assert 'vggt_omega_1b_512.pt' in omega and '原始PT' in omega and 'FAIR' in omega
    assert '不重分发' not in json.dumps(info, ensure_ascii=False)
    assert required['cache_gib'][1]['default'] == 20.0
    assert set(info['input']['optional']) == {'model', 'video_vae'}
