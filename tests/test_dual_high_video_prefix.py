import pytest
import torch
from comfy.nested_tensor import NestedTensor
from h3_audio_t8_pkg import long_video_dual_model_runner as runner
from test_long_video_dual_model_runner import rig  # noqa: F401
from h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced import _write_effects_audit


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


@pytest.mark.parametrize('fault', ['chain', 'geometry', 'nan', 'conflicting_mask', 'too_short'])
def test_invalid_context_or_other_visual_owner_rejected(fault):
    latent, context = case()
    if fault == 'chain': context['metadata']['chain_id'] = 'wrong'
    elif fault == 'geometry': context['video_tail'] = torch.zeros(1,24,2,4,8)
    elif fault == 'nan': context['video_tail'][0,0,0,0,0] = float('nan')
    elif fault == 'conflicting_mask': latent['noise_mask'].unbind()[0][:,:,:2] = .5
    else:
        v,a=latent['samples'].unbind()
        latent={'samples':NestedTensor((v[:,:,:2],a))}
    with pytest.raises(ValueError): lock(latent,context)


def test_unsupported_core_rejected(monkeypatch):
    from h3_audio_t8_pkg import native_masked_context_advanced as native
    def reject(): raise RuntimeError('mask support missing')
    monkeypatch.setattr(native,'require_native_h3_av_mask_support',reject)
    with pytest.raises(RuntimeError,match='mask support missing'): lock(*case())


@pytest.mark.parametrize('resume', [False, True])
def test_runner_applies_prefix_after_fresh_or_cached_reconcile(rig, tmp_path, monkeypatch, resume):
    engine, run, _, fail, _, second = rig
    engine.video_context_mode = 'high_native_mask_exp'
    first_result = run()
    _write_effects_audit(str(tmp_path/'candidates/segment_00000/candidate0/candidate.json'),
        {'contract_sha256':'job','segment_index':0,'candidate_id':'candidate0',
         'sampling_plan':first_result['sampling_report']})
    context = case()[1]
    context['video_tail'] = torch.full((1,24,2,4,8),9.)
    original = runner.sample_model_stage
    observations=[]
    def sample(model, positive, latent, **kwargs):
        if model is second:
            v,a=latent['samples'].unbind()
            vm,am=latent['noise_mask'].unbind()
            observations.append((v.clone(),a.clone(),vm.clone(),am.clone()))
        return original(model,positive,latent,**kwargs)
    monkeypatch.setattr(runner,'sample_model_stage',sample)
    if resume:
        fail['high']=True
        with pytest.raises(RuntimeError,match='second pass failure'):
            run(1,'candidate1','candidate0',context,context_frames=5)
        fail['high']=False
    result=run(1,'candidate1','candidate0',context,context_frames=5)
    for v,a,vm,am in observations:
        assert torch.all(v[:,:,:2]==9) and torch.all(v[:,:,2:]==2)
        assert torch.all(vm[:,:,:2]==0) and torch.all(vm[:,:,2:]==1)
        assert torch.all(a==1) and torch.all(am==1)
    report=result['sampling_report']['dual_model']['second_pass']['video_context']
    assert report['applied'] and report['audio_touched'] is False
    count=len(observations)
    cached=run(1,'candidate1','candidate0',context,context_frames=5)
    assert cached['sampling_report']['dual_model']['high_reused']
    assert len(observations)==count


def test_changed_mode_cannot_reuse_reference_only_stages(rig):
    engine,run,calls,*_=rig
    run()
    count=len(calls)
    engine.video_context_mode='high_native_mask_exp'
    result=run()
    assert not result['sampling_report']['dual_model']['low_reused']
    assert not result['sampling_report']['dual_model']['high_reused']
    assert len([x for x in calls[count:] if x[0]=='sample'])==2
