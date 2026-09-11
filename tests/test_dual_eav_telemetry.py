"""Combined real owner/delegate telemetry; tiny CPU attention, not GPU qualification."""
import pytest
import torch
from comfy.ldm.modules import attention
from comfy.patcher_extension import WrapperExecutor

from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.long_video import patch_long_video_model
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
    PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
)
from test_prompt_relay_core_compat import bound_layout
from test_relay_kj_backend import kj, selector  # noqa: F401 -- fixture
from tools import run_dual_model_pilot as pilot


@pytest.mark.parametrize('enabled', [False, True])
def test_stock20_recipe_keeps_second_lora_and_bypasses_only_first(enabled):
    graph = {'1': {'class_type': 'UNETLoader'},
             '2': {'class_type': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced'},
             '3': {'class_type': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced'},
             '21': {'inputs': {'model': ['2', 0]}}, '22': {'inputs': {'model': ['3', 0]}},
             '8': {'inputs': {'coarse_steps': 4, 'eav_mode': 'disabled'}}}
    pilot.attach_eav_stock20(graph, enabled)
    assert graph['21']['inputs']['model'] == ['1' if enabled else '2', 0]
    assert graph['22']['inputs']['model'] == ['3', 0]
    assert graph['8']['inputs']['coarse_steps'] == (20 if enabled else 4)
    assert graph['8']['inputs']['eav_mode'] == ('apply_exp' if enabled else 'disabled')
    assert ('2' in graph) is (not enabled)


@pytest.mark.parametrize('outcome', ['success', 'abort', 'foreign'])
def test_combined_owner_records_its_actual_backend_after_execution(request, monkeypatch, outcome):
    installed_kj = request.getfixturevalue('kj')
    source, _ = selector(installed_kj)
    binding, layout = bound_layout('video_only_paper')
    source, _ = relay.patch_prompt_relay_model(source, binding, 32)
    backend = relay.prompt_relay_model_contract(source)['attention_backend']
    source = patch_long_video_model(source)
    source.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, {
        'schema': PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
        'global_plan_hash': 'fixture-global', 'projected_plan_hash': 'fixture-projected',
        'binding_hash': binding['binding_hash'], 'segment_index': 1,
    })
    model, runtime, _ = eav.build_eav_prompt_relay_long_video_model(
        source, torch.cat([torch.linspace(1., .05, 20), torch.zeros(1)]),
        segment_index=1, context_frames=22, mode='report_only', tau=4.,
        start_video_progress=0., end_video_progress=1., max_workspace_mib=32, g_hard_limit=1.5)
    # Isolate telemetry from the already-tested packed-motion/FETA equations.
    monkeypatch.setattr(eav, '_runtime_route', lambda **kwargs: {
        'task': 'LongVideoMotion', 'active': False, 'frames': 3, 'spatial_tokens': 4,
        'seq_len': layout.seq_len, 'audio_start': 4, 'audio_end': 6,
        'video_start': 6, 'video_end': layout.seq_len, 'sigma_video': .5, 'progress_video': .5,
    })
    q = torch.randn(1, 2, layout.seq_len, 8)
    options = model.model_options['transformer_options']
    if outcome == 'foreign':
        original = options['optimized_attention_override']
        def foreign(*args, **kwargs):
            pytest.fail('Foreign copied-marker function must not execute')
        foreign.__dict__.update(original.__dict__)
        options['optimized_attention_override'] = foreign

    def body(x, timestep, context, transformer_options, **kwargs):
        result = attention.attention_pytorch(q, q, q, 2, skip_reshape=True,
                                             transformer_options=transformer_options)
        if outcome == 'abort':
            raise RuntimeError('intentional downstream abort')
        return result

    executor = WrapperExecutor.new_executor(body, model.get_wrappers(
        'diffusion_model', eav.EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY))
    def run():
        return executor.execute([q], None, None, options, minimax_payload={'layout': layout},
                                **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding['binding_hash']})
    if outcome == 'success':
        run()
    else:
        with pytest.raises(RuntimeError):
            run()
    assert relay.PROMPT_RELAY_RUNTIME_KEY not in options and eav.EAV_RUNTIME_KEY not in options
    if outcome == 'foreign':
        assert backend.report()['completed_calls'] == {}
    else:
        assert backend.report()['completed_calls']
        assert runtime.config['composed_attention_backend'] == backend.report()
        if outcome == 'abort':
            assert 'intentional downstream abort' in runtime.snapshot(consume=False)['aborted']


@pytest.mark.parametrize('mutation', [None, 'nfe', 'measurements', 'backend', 'context', 'aborted'])
def test_independent_audit_rejects_eav_shortcuts(mutation):
    from tools.audit_dual_model_pilot import validate_eav_stage
    backend = {'kind': 'fixture', 'completed_calls': {'sage:unbiased': 1000}}
    audit = {'status': 'apply_exp_long_video_segment_verified', 'aborted': None,
             'model_forward_count': 20, 'attention_calls_per_active_forward': [50] * 20,
             'long_video': {'context_frames': 0, 'segment_index': 0},
             'config': {'mode': 'apply_exp', 'sampling_profile': 'stock20',
                        'sigma_contract': {'nfe': 20}, 'composed_attention_backend': backend}}
    report = {'completed_network_forwards': 20, 'schedule': {'mode': 'full_stock20'},
              'backend': backend, 'eav_audit': audit}
    if mutation == 'nfe':
        report['completed_network_forwards'] = 4
    elif mutation == 'measurements':
        audit['attention_calls_per_active_forward'] = [50] * 4
    elif mutation == 'backend':
        report['backend'] = {'kind': 'another'}
    elif mutation == 'context':
        audit['long_video']['context_frames'] = 22
    elif mutation == 'aborted':
        audit['aborted'] = 'error'
    if mutation:
        with pytest.raises(ValueError, match='audit mismatch'):
            validate_eav_stage(report)
    else:
        assert validate_eav_stage(report)['active_block_measurements'] == 1000
