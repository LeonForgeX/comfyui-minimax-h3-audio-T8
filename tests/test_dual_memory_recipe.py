from copy import deepcopy

import pytest

from h3_t8.prompt_tags import prepare_prompt
from tools.run_dual_model_pilot import (
    BUND_KOREAN_MV_GLOBAL_PROMPT,
    BUND_KOREAN_MV_LOCAL_PROMPTS,
    FLAT_CEL_GRID_GLOBAL_PROMPT,
    FLAT_CEL_GRID_LOCAL_PROMPTS,
    FLAT_STUDIO_GLOBAL_PROMPT,
    FLAT_STUDIO_LOCAL_PROMPTS,
    apply_relay_prompt_profile,
    attach_memory_chunks,
    attach_t8_memory,
)
from tools.audit_dual_model_pilot import validate_memory_backend


@pytest.mark.parametrize('heads,ffn', [(1, 1), (4, 1), (1, 2), (4, 2)])
def test_both_model_branches_receive_declared_patches_only(heads, ffn):
    graph = {'8': {'inputs': {'model_pass1': ['21', 0], 'model_pass2': ['22', 0]}}}
    attach_memory_chunks(graph, 'kj-memory', heads, ffn)
    for branch, source, h, f in [('model_pass1', '21', '24', '26'), ('model_pass2', '22', '25', '27')]:
        if heads > 1:
            assert graph[h]['inputs'] == {'model': [source, 0], 'head_chunks': 4}
            source = h
        if ffn > 1:
            assert graph[f]['inputs'] == {'model': [source, 0], 'chunks': 2, 'seq_threshold': 4096}
            source = f
        assert graph['8']['inputs'][branch] == [source, 0]
    assert len(graph) == 1 + 2 * (heads > 1) + 2 * (ffn > 1)


@pytest.mark.parametrize('backend,heads,ffn', [('sol', 4, 2), ('kj', 4, 1), ('kj-memory', 3, 2)])
def test_invalid_composition_fails_without_changing_graph(backend, heads, ffn):
    graph = {'8': {'inputs': {}}}
    original = deepcopy(graph)
    with pytest.raises(ValueError):
        attach_memory_chunks(graph, backend, heads, ffn)
    assert graph == original


def test_existing_node_not_replaced():
    graph = {'8': {'inputs': {}}, '27': {'inputs': {'existing': True}}}
    original = deepcopy(graph)
    with pytest.raises(ValueError, match='overwrite'):
        attach_memory_chunks(graph, 'kj-memory', 4, 2)
    assert graph == original


def test_flat_studio_profile_is_an_eight_second_fixed_light_seam_probe():
    graph = {
        '7': {
            'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced',
            'inputs': {'global_prompt': 'old', 'local_prompts': 'old'},
        }
    }
    apply_relay_prompt_profile(graph, 'flat-studio', 8)
    inputs = graph['7']['inputs']
    assert inputs == {
        'global_prompt': FLAT_STUDIO_GLOBAL_PROMPT,
        'local_prompts': FLAT_STUDIO_LOCAL_PROMPTS,
        'length': 193,
        'time_ranges': '0-15\n15-40\n40-75\n75-100',
    }
    assert all(
        phrase in inputs['global_prompt']
        for phrase in ('constant exposure', 'No windows', 'moving shadows', 'flicker')
    )
    assert 'continues through the segment boundary' in inputs['local_prompts']


def test_flat_studio_profile_rejects_non_continuation_probe():
    graph = {
        '7': {
            'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced',
            'inputs': {},
        }
    }
    with pytest.raises(ValueError, match='two-segment 8s'):
        apply_relay_prompt_profile(graph, 'flat-studio', 3)


def test_flat_cel_grid_profile_removes_lighting_as_a_visual_variable():
    graph = {
        '7': {
            'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced',
            'inputs': {'global_prompt': 'old', 'local_prompts': 'old'},
        }
    }
    apply_relay_prompt_profile(graph, 'flat-cel-grid', 8)
    inputs = graph['7']['inputs']
    assert inputs == {
        'global_prompt': FLAT_CEL_GRID_GLOBAL_PROMPT,
        'local_prompts': FLAT_CEL_GRID_LOCAL_PROMPTS,
        'length': 193,
        'time_ranges': '0-15\n15-40\n40-75\n75-100',
    }
    assert all(
        phrase in inputs['global_prompt']
        for phrase in ('Lighting is not depicted', 'no gradients', 'brightness breathing')
    )
    assert 'through the segment boundary without pausing' in inputs['local_prompts']
    assert 'no speech' in inputs['global_prompt']


def test_flat_cel_grid_profile_rejects_non_continuation_probe():
    graph = {
        '7': {
            'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced',
            'inputs': {},
        }
    }
    with pytest.raises(ValueError, match='two-segment 8s'):
        apply_relay_prompt_profile(graph, 'flat-cel-grid', 3)


def test_bund_korean_mv_profile_binds_reference_identity_lyrics_and_seam_crossing():
    graph = {
        '7': {
            'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced',
            'inputs': {'global_prompt': 'old', 'local_prompts': 'old'},
        }
    }
    apply_relay_prompt_profile(graph, 'bund-korean-mv-8s', 8)
    inputs = graph['7']['inputs']
    assert inputs == {
        'global_prompt': BUND_KOREAN_MV_GLOBAL_PROMPT,
        'local_prompts': BUND_KOREAN_MV_LOCAL_PROMPTS,
        'length': 193,
        'time_ranges': '0-43.75\n43.75-87.5\n87.5-100',
    }
    assert '已连接首帧参考图' in inputs['global_prompt']
    assert 'eight-second, two-segment' in inputs['global_prompt']
    assert '<d>[Korean] 아침 햇살 문을 열면</d>' in inputs['local_prompts']
    assert '<d>[Korean] 작은 새가 노래해요</d>' in inputs['local_prompts']
    assert '<d>[Korean] 초록 바람 <scenetrans></d>' in inputs['local_prompts']
    assert '连续穿过内部片段接缝' in inputs['local_prompts']
    combined = inputs['global_prompt'] + '\n' + inputs['local_prompts']
    normalized, warnings = prepare_prompt(
        combined, {'pictures': 0, 'videos': 0, 'audios': 0}, strict=True
    )
    assert normalized == combined
    assert warnings == []


def test_bund_korean_mv_profile_rejects_non_continuation_probe():
    graph = {'7': {'class_type': 'MiniMaxH3PromptRelayPlanT8Advanced', 'inputs': {}}}
    with pytest.raises(ValueError, match='two-segment 8s'):
        apply_relay_prompt_profile(graph, 'bund-korean-mv-8s', 3)


@pytest.mark.parametrize('heads', [1, 4])
def test_t8_memory_probe_uses_two_independent_native_paths(heads):
    graph = {
        '8': {'inputs': {}},
        '21': {'class_type': 'old-kj'},
        '22': {'class_type': 'old-kj'},
        '30': {'class_type': 'pass1-extra-lora'},
        '31': {'class_type': 'pass2-extra-lora'},
    }
    attach_t8_memory(graph, heads, 2)
    assert graph['21']['inputs'] == {'model': ['30', 0], 'head_chunks': heads}
    assert graph['22']['inputs'] == {'model': ['31', 0], 'head_chunks': heads}
    assert graph['24']['inputs'] == {
        'model': ['21', 0], 'chunks': 2, 'seq_threshold': 4096
    }
    assert graph['25']['inputs'] == {
        'model': ['22', 0], 'chunks': 2, 'seq_threshold': 4096
    }
    assert graph['8']['inputs']['model_pass1'] == ['24', 0]
    assert graph['8']['inputs']['model_pass2'] == ['25', 0]


@pytest.mark.parametrize('heads,ffn', [(2, 2), (4, 3)])
def test_t8_memory_probe_rejects_unreviewed_settings_without_mutation(heads, ffn):
    graph = {
        '8': {'inputs': {}}, '21': {}, '22': {}, '30': {}, '31': {}
    }
    original = deepcopy(graph)
    with pytest.raises(ValueError, match='explicit tested'):
        attach_t8_memory(graph, heads, ffn)
    assert graph == original


@pytest.mark.parametrize('case', [None, 'heads', 'ffn', 'count', 'source', 'grouping'])
def test_actual_composition_audit_rejects_missing_or_different_patches(case):
    backend = {'memory_composition': {'kind': 'kj_memory_sage', 'head_chunks': 4,
               'ffn_settings': [2, 4096], 'source_sha256s': ['fixture']},
               'completed_calls': {'sage:unbiased': 800},
               'head_grouping': 'inside_delegate; full-head Relay/EAV once per block'}
    terminal = {'memory_head_chunks': 4, 'memory_ffn_chunks': 2}
    if case == 'heads':
        backend['memory_composition']['head_chunks'] = 1
    elif case == 'ffn':
        backend['memory_composition']['ffn_settings'] = None
    elif case == 'count':
        backend['completed_calls']['sage:unbiased'] = 200
    elif case == 'source':
        backend['memory_composition']['source_sha256s'] = []
    elif case == 'grouping':
        backend['head_grouping'] = 'outside'
    if case:
        with pytest.raises(ValueError):
            validate_memory_backend(backend, terminal)
    else:
        validate_memory_backend(backend, terminal)
