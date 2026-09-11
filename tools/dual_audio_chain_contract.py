"""Read-back gates for native-audio 4+4 chains; not perceptual qualification."""
import torch


def audit_audio_stages(records, tensors):
    low, prepared, high = (tensors[k] for k in ('low_x0', 'high_input', 'high_output'))
    contract = records['high_output']['contract']
    policy = contract.get('audio_policy', {})
    a, b, c = (x['samples_audio'] for x in (low, prepared, high))
    if a.shape != b.shape or b.shape != c.shape or any(not torch.isfinite(x).all() for x in (a, b, c)):
        raise ValueError('Invalid stage audio geometry or values')
    mask = prepared.get('mask_audio')
    version = contract.get('audio_policy_version')
    if version not in (None, 2) or (version is None and policy):
        raise ValueError('Unknown audio policy version cannot qualify as historical evidence')
    if version is None:
        if (mask is None or torch.count_nonzero(mask)
                or not torch.equal(a, b) or not torch.equal(a, c)):
            raise ValueError('Historical locked audio contract changed')
        return {'mode': 'historical_first_pass_locked'}
    if (policy.get('effective_source') != 'legacy_policy'
            or policy.get('context_handoff') != 'coarse_unlocked_template_locked_v1'
            or any(records[k]['contract'].get('audio_policy') != policy for k in records)):
        raise ValueError('Chain audio policy is missing, stale, or inconsistent')
    if mask is None:
        mask = torch.ones_like(b)
    if mask.shape != b.shape or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('This native-audio chain audit requires a binary context mask')
    generated = mask == 1
    locked = ~generated
    if not generated.any() or not torch.equal(a[generated], b[generated]):
        raise ValueError('Generated audio was frozen or coarse audio discarded')
    delta = float((c[generated] - b[generated]).abs().max())
    locked_delta = float((c[locked] - b[locked]).abs().max()) if locked.any() else 0.
    if delta <= 1e-6 or locked_delta > 1e-6:
        raise ValueError('Generated audio did not evolve or locked context changed')
    if (records['low_x0']['report']['schedule']['coarse_audio_sigmas'][-1] <= 0
            or records['high_output']['report']['schedule']['refine_audio_sigmas'][-1] != 0
            or records['high_output']['report']['audio_delivery']['source'] != 'completed_second_pass_output'):
        raise ValueError('Partial coarse audio was not completed in pass2')
    return {'mode': 'joint_audio_with_locked_context', 'generated_delta': delta,
            'locked_delta': locked_delta, 'quality': 'human_review_required'}


def audit_saved_contexts(low, high, low_context, high_context):
    for context, video in ((low_context, low['samples_video']),
                           (high_context, high['samples_video'])):
        tail = context['video_tail']
        audio = context['audio_tail']
        expected_audio = high['samples_audio']
        if (tail.ndim != 5 or audio.ndim != 4 or tail.shape[2] < 1 or audio.shape[-1] < 1
                or tail.dtype != video.dtype or audio.dtype != expected_audio.dtype
                or not torch.equal(tail, video[:1, :, -tail.shape[2]:])
                or not torch.equal(audio, expected_audio[:1, :, :, -audio.shape[-1]:])):
            raise ValueError('Saved context differs from its video or completed high-pass audio')
    if not torch.equal(low_context['audio_tail'], high_context['audio_tail']):
        raise ValueError('Low/high contexts do not share the completed audio tail')
