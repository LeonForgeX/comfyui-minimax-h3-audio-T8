import pytest
import torch
from comfy.nested_tensor import NestedTensor
from h3_audio_t8_pkg import long_video_dual_model_runner as runner
from h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced import _write_effects_audit


from test_long_video_dual_model_runner import rig  # noqa: F401
from failed_dual_video_bridge_fixture import bridge_high_video_boundary


def case():
    video = torch.randn(1, 24, 7, 2, 4)
    audio = torch.randn(1, 32, 2, 20)
    vm, am = torch.ones_like(video), torch.ones_like(audio)
    am[..., :4] = 0
    am[..., 4:6] = .5
    vm[:, :, -1] = .4
    latent = {'samples': NestedTensor((video, audio)), 'noise_mask': NestedTensor((vm, am))}
    context = {'schema': 1, 'empty': False, 'video_tail': torch.randn(1, 24, 2, 2, 4),
        'metadata': {'chain_id': 'test', 'source_segment_index': 0, 'target_segment_index': 1, 'max_context_frames': 5}}
    return latent, context


def lock(latent, context):
    return runner.lock_high_video_prefix(latent, context, chain_id='test', segment_index=1, context_frames=5)


def bridge(latent, context):
    return bridge_high_video_boundary(
        latent, context, chain_id='test', segment_index=1, context_frames=5
    )


def test_prefix_is_prior_final_video_audio_and_other_cells_unchanged():
    latent, context = case()
    v, a = latent['samples'].unbind()
    vm, am = latent['noise_mask'].unbind()
    before = v.clone()
    result, report = lock(latent, context)
    rv, ra = result['samples'].unbind()
    rvm, ram = result['noise_mask'].unbind()
    assert torch.equal(rv[:, :, :2], context['video_tail'])
    assert torch.equal(rv[:, :, 2:], v[:, :, 2:])
    assert torch.equal(v, before)
    assert ra is a and ram is am
    assert torch.all(rvm[:, :, :2] == 0)
    assert torch.equal(rvm[:, :, 2:], vm[:, :, 2:])
    assert report['audio_touched'] is False


def test_absent_mask_only_locks_video_prefix():
    latent, context = case()
    latent.pop('noise_mask')
    result, _ = lock(latent, context)
    vm, am = result['noise_mask'].unbind()
    assert vm.shape == latent['samples'].unbind()[0].shape
    assert am.shape == latent['samples'].unbind()[1].shape
    assert torch.all(am == 1)


def test_ramp_mode_keeps_prefix_exact_and_releases_three_latent_cells_gradually():
    latent, context = case()
    video, audio = latent['samples'].unbind()
    _, original_audio_mask = latent['noise_mask'].unbind()
    output, report = runner.lock_high_video_prefix(
        latent, context, chain_id='test', segment_index=1, context_frames=5,
        mode='high_native_mask_ramp_exp',
    )
    out_video, out_audio = output['samples'].unbind()
    video_mask, audio_mask = output['noise_mask'].unbind()
    assert torch.equal(out_video[:, :, :2], context['video_tail'])
    assert torch.equal(out_video[:, :, 2:], video[:, :, 2:])
    assert out_audio is audio and audio_mask is original_audio_mask
    assert torch.all(video_mask[:, :, :2] == 0)
    for index, value in enumerate((.25, .5, .75), start=2):
        assert torch.all(video_mask[:, :, index] == value)
    assert torch.all(video_mask[:, :, 5] == 1)
    assert torch.all(video_mask[:, :, 6] == .4)
    assert report['mode'] == 'high_native_mask_ramp_exp'
    assert report['release_ramp_values'] == [.25, .5, .75]
    assert report['audio_touched'] is False


def test_ramp_mode_refuses_to_overwrite_an_existing_visual_mask_owner():
    latent, context = case()
    latent['noise_mask'].unbind()[0][:, :, 2] = .5
    with pytest.raises(ValueError, match='ramp is already owned'):
        runner.lock_high_video_prefix(
            latent, context, chain_id='test', segment_index=1, context_frames=5,
            mode='high_native_mask_ramp_exp',
        )


@pytest.mark.parametrize('frames,steps', [(22, 7), (39, 12)])
def test_supported_context_length_locks_exact_prefix_only(frames, steps):
    video = torch.randn(1, 24, 37, 2, 4)
    audio = torch.randn(1, 32, 2, 40)
    tail = torch.randn(1, 24, 12, 2, 4)
    context = {'schema': 1, 'empty': False, 'video_tail': tail,
               'metadata': {'chain_id': 'test', 'source_segment_index': 0,
                            'target_segment_index': 1, 'max_context_frames': 39}}
    output, report = runner.lock_high_video_prefix({'samples': NestedTensor((video, audio))},
        context, chain_id='test', segment_index=1, context_frames=frames)
    out_video, out_audio = output['samples'].unbind()
    mask_video, mask_audio = output['noise_mask'].unbind()
    assert torch.equal(out_video[:, :, :steps], tail[:, :, -steps:])
    assert torch.equal(out_video[:, :, steps:], video[:, :, steps:])
    assert out_audio is audio and torch.all(mask_audio == 1)
    assert torch.all(mask_video[:, :, :steps] == 0)
    assert torch.all(mask_video[:, :, steps:] == 1)
    assert report['context_steps'] == steps


def test_post_sampling_bridge_aligns_first_free_video_then_cosine_releases_audio_untouched():
    latent, context = case()
    video, audio = latent['samples'].unbind()
    video[:, :, :2] = context['video_tail']
    before = video.clone()
    noise_mask = latent['noise_mask']
    result, report = bridge(latent, context)
    bridged, result_audio = result['samples'].unbind()
    assert torch.equal(bridged[:, :, :2], context['video_tail'])
    assert torch.allclose(bridged[:, :, 2], context['video_tail'][:, :, -1], atol=1e-6, rtol=0)
    assert torch.equal(bridged[:, :, -1], before[:, :, -1])
    assert not torch.equal(bridged[:, :, 3:-1], before[:, :, 3:-1])
    assert torch.equal(video, before)
    assert result_audio is audio
    assert result['noise_mask'] is noise_mask
    assert report['bridge_tokens'] == 5
    assert report['bridge_frames'] == 17
    assert report['first_free_rms_before'] > 0
    assert report['first_free_rms_after'] < 1e-6
    assert report['audio_touched'] is False
    assert report['rgb_frames_blended'] is False


@pytest.mark.parametrize('fault', ['unlocked_prefix', 'nan', 'short_generated_region'])
def test_post_sampling_bridge_rejects_invalid_completed_output(fault):
    latent, context = case()
    video, _ = latent['samples'].unbind()
    video[:, :, :2] = context['video_tail']
    if fault == 'unlocked_prefix':
        video[:, :, 0] += 1
    elif fault == 'nan':
        video[:, :, 2, 0, 0] = float('nan')
    else:
        video, audio = latent['samples'].unbind()
        latent['samples'] = NestedTensor((video[:, :, :6], audio))
    with pytest.raises(ValueError):
        bridge(latent, context)


@pytest.mark.parametrize('fault', ['chain', 'geometry', 'nan', 'conflicting_mask', 'too_short'])
def test_invalid_context_or_other_visual_owner_rejected(fault):
    latent, context = case()
    if fault == 'chain':
        context['metadata']['chain_id'] = 'wrong'
    elif fault == 'geometry':
        context['video_tail'] = torch.zeros(1,24,2,4,8)
    elif fault == 'nan':
        context['video_tail'][0,0,0,0,0] = float('nan')
    elif fault == 'conflicting_mask':
        latent['noise_mask'].unbind()[0][:,:,:2] = .5
    else:
        v,a=latent['samples'].unbind()
        latent={'samples':NestedTensor((v[:,:,:2],a))}
    with pytest.raises(ValueError):
        lock(latent,context)


def test_unsupported_core_rejected(monkeypatch):
    from h3_audio_t8_pkg import native_masked_context_advanced as native

    def reject():
        raise RuntimeError('mask support missing')

    monkeypatch.setattr(native,'require_native_h3_av_mask_support',reject)
    with pytest.raises(RuntimeError,match='mask support missing'):
        lock(*case())


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('mode', ['high_native_mask_exp', 'high_native_mask_ramp_exp'])
def test_runner_applies_prefix_after_fresh_or_cached_reconcile(
    rig, tmp_path, monkeypatch, resume, mode  # noqa: F811
):
    engine, run, _, fail, _, second = rig
    engine.video_context_mode = mode
    first_result = run()
    _write_effects_audit(str(tmp_path/'candidates/segment_00000/candidate0/candidate.json'),
        {'contract_sha256':'job','segment_index':0,'candidate_id':'candidate0',
         'sampling_plan':first_result['sampling_report']})
    context = case()[1]
    context['video_tail'] = torch.full((1,24,2,4,8),9.)
    original = runner.sample_model_stage
    def forbidden_bridge(*args, **kwargs):
        pytest.fail('Failed post-sampling bridge must never run in delivery')
    monkeypatch.setattr(runner, 'bridge_high_video_boundary', forbidden_bridge, raising=False)
    observations=[]
    def sample(model, positive, latent, **kwargs):
        if model is second:
            v,a=latent['samples'].unbind()
            vm,am=latent['noise_mask'].unbind()
            observations.append((v.clone(),a.clone(),vm.clone(),am.clone()))
        result, report = original(model,positive,latent,**kwargs)
        if model is second:
            # The small fixture sampler adds a constant everywhere and does not
            # implement Core's native zero-mask restoration. Mirror the real
            # sampler contract; delivery must preserve its free video region.
            sampled_video, sampled_audio = result['samples'].unbind()
            input_video = latent['samples'].unbind()[0]
            sampled_video = sampled_video.clone()
            sampled_video[:, :, :2] = input_video[:, :, :2]
            result['samples'] = NestedTensor((sampled_video, sampled_audio))
        return result, report
    monkeypatch.setattr(runner,'sample_model_stage',sample)
    if resume:
        fail['high']=True
        with pytest.raises(RuntimeError,match='second pass failure'):
            run(1,'candidate1','candidate0',context,context_frames=5)
        fail['high']=False
    result=run(1,'candidate1','candidate0',context,context_frames=5)
    for v,a,vm,am in observations:
        assert torch.all(v[:,:,:2]==9) and torch.all(v[:,:,2:]==2)
        assert torch.all(vm[:,:,:2]==0)
        if mode == 'high_native_mask_ramp_exp':
            assert torch.all(vm[:, :, 2] == .25)
            assert torch.all(vm[:, :, 3] == .5)
            assert torch.all(vm[:, :, 4] == .75)
            assert torch.all(vm[:, :, 5:] == 1)
        else:
            assert torch.all(vm[:,:,2:]==1)
        assert torch.all(a==1) and torch.all(am==1)
    report=result['sampling_report']['dual_model']['second_pass']['video_context']
    assert report['applied'] and report['audio_touched'] is False
    assert report['mode'] == mode
    assert 'post_sampling_bridge' not in report
    assert torch.all(result['sampled']['samples'].unbind()[0][:, :, 2:] == 4)
    count=len(observations)
    cached=run(1,'candidate1','candidate0',context,context_frames=5)
    assert cached['sampling_report']['dual_model']['high_reused']
    assert len(observations)==count
    assert torch.equal(cached['sampled']['samples'].unbind()[0], result['sampled']['samples'].unbind()[0])


def test_changed_mode_cannot_reuse_reference_only_stages(rig):  # noqa: F811
    engine,run,calls,*_=rig
    run()
    count=len(calls)
    engine.video_context_mode='high_native_mask_exp'
    result=run()
    assert not result['sampling_report']['dual_model']['low_reused']
    assert not result['sampling_report']['dual_model']['high_reused']
    assert len([x for x in calls[count:] if x[0]=='sample'])==2
