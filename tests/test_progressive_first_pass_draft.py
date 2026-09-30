"""Registered public LOW replay executes real tiny H3/Euler, not trained lifter/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg.nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8 as Sampler
from h3_audio_t8_pkg.nodes_stage_residency import (
    MiniMaxH3FirstPassFingerprintT8 as Fingerprint, MiniMaxH3StageResidencyReleaseT8 as Release,
)
from h3_audio_t8_pkg.progressive_first_pass_draft import CAPABILITY, digest
from h3_audio_t8_pkg.sampling import native_flow_sigmas, setup_dual_clock_sampling
from test_progressive_sampling_runtime import tiny_model, conditioning, latent, stub_lifter  # noqa: F401
from test_progressive_stage_inputs import inputs
from test_progressive_continuation import accepted, capture  # noqa: F401


SCOPE = json.dumps({'chain_id': 'draft-test', 'material_pack_index': 2,
                    'parent_candidate_id': '', 'parent_revision': 0})


def run(root=None, *, model=None, high=None, task='t2va', scope=SCOPE, high_seed=None,
        mode='save_or_reuse', prepared=None, **changes):
    model = model or tiny_model()
    high = high or tiny_model()
    values = dict(model=model, model_hires=high, positive=conditioning(task), negative=conditioning(),
        av_latent=latent(), sampler=comfy.samplers.ksampler('euler'), sigmas=native_flow_sigmas(4, 12.),
        upscaler_model='test', seed=19, low_evaluations=2, task=task)
    if root is not None:
        values.update(draft_mode=mode, draft_directory=str(root.resolve()), draft_scope_json=scope)
    if high_seed is not None:
        values['high_seed'] = high_seed
    if prepared is not None:
        low, target = prepared
        values.update(positive=target[0], negative=target[0], av_latent=target[1],
            input_mode='prepared_pair_exp', positive_low=low[0], negative_low=low[0], av_latent_low=low[1])
    values.update(changes)
    output, report = Sampler.execute(**values).result
    return output, json.loads(report)


def same(left, right):
    for a, b in zip(left['samples'].unbind(), right['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_public_saved_and_replayed_default_video_audio_are_bitwise(tmp_path, stub_lifter):  # noqa: F811
    baseline, _ = run()
    first, report = run(tmp_path)
    replay, resumed = run(tmp_path)
    same(first, baseline)
    same(replay, baseline)
    assert report['first_pass_draft']['saved_low']
    assert resumed['first_pass_draft']['reused_low']
    assert resumed['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    assert resumed['counts']['callbacks'] == {'low': 0, 'high': 2}
    assert resumed['first_pass_draft']['historical_low_report']['actual_forwards'] == 2
    record = json.loads(next(tmp_path.glob('low-boundary-*.json')).read_text())
    assert record['capability'] == CAPABILITY
    assert record['cache_id'] == digest(record['contract'])
    assert record['contract']['scope']['material_pack_index'] == 2


def test_high_model_lora_and_independent_seed_reuse_low(tmp_path, stub_lifter):  # noqa: F811
    run(tmp_path)
    high = tiny_model()
    key, weight = next(iter(high.model.named_parameters()))
    high.add_patches({key: ('diff', (torch.full_like(weight, .02),))}, .7)
    changed, report = run(tmp_path, high=high, high_seed=97)
    fresh, _ = run(high=high, high_seed=97)
    same(changed, fresh)
    assert report['first_pass_draft']['reused_low']
    assert report['noise']['high_seed'] == report['noise']['high_sampler_seed'] == 97
    assert report['counts']['actual_forwards']['low'] == 0
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 1


@pytest.mark.parametrize('change', ['weight', 'lora', 'prompt', 'seed', 'scope', 'schedule', 'selected_id'])
def test_incompatible_low_refuses_reuse_before_lift(tmp_path, stub_lifter, change):  # noqa: F811
    run(tmp_path)
    model, values = tiny_model(), {}
    if change == 'weight':
        with torch.no_grad():
            next(model.model.parameters()).add_(.01)
    elif change == 'lora':
        key, weight = next(iter(model.model.named_parameters()))
        model.add_patches({key: ('diff', (torch.full_like(weight, .01),))}, .8)
    elif change == 'prompt':
        positive = conditioning()
        positive[0][0].add_(.1)
        values['positive'] = positive
    elif change == 'seed':
        values['seed'] = 20
    elif change == 'scope':
        values['scope'] = json.dumps({'chain_id': 'other', 'material_pack_index': 2})
    elif change == 'schedule':
        values['sigmas'] = native_flow_sigmas(4, 11.)
    else:
        values['draft_id'] = 'a' * 64
    count = len(stub_lifter)
    with pytest.raises(ValueError, match='compatible|incompatible|no completed receipt'):
        run(tmp_path, model=model, mode='reuse_only', **values)
    assert len(stub_lifter) == count


def test_selected_draft_names_changed_low_fields_before_sampling(tmp_path, stub_lifter):  # noqa: F811
    _, report = run(tmp_path)
    selected = report['first_pass_draft']['cache_id']
    count = len(stub_lifter)
    with pytest.raises(ValueError, match='changed LOW contract fields: scope, settings'):
        run(tmp_path, draft_id=selected, scope=json.dumps({'project': 'changed'}), seed=23)
    assert len(stub_lifter) == count
    _, replayed = run(tmp_path, draft_id=selected, mode='reuse_only')
    assert replayed['counts']['callbacks']['low'] == 0


@pytest.mark.parametrize('corruption', ['tensor', 'report', 'receipt_symlink'])
def test_corrupt_public_draft_never_falls_back_to_low(tmp_path, stub_lifter, corruption):  # noqa: F811
    run(tmp_path)
    path = next(tmp_path.glob('low-boundary-*.json'))
    record = json.loads(path.read_text())
    if corruption == 'tensor':
        tensor = tmp_path / record['tensor_file']
        tensor.write_bytes(tensor.read_bytes() + b'corruption')
    elif corruption == 'report':
        record['low_report']['callbacks'] = 0
        path.write_text(json.dumps(record))
    else:
        outside = tmp_path / 'outside.json'
        path.rename(outside)
        path.symlink_to(outside)
    count = len(stub_lifter)
    with pytest.raises(ValueError, match='integrity|completed LOW|receipt path'):
        run(tmp_path)
    assert len(stub_lifter) == count


def test_save_new_keeps_completed_low_immutable(tmp_path, stub_lifter):  # noqa: F811
    _, report = run(tmp_path, mode='save_new')
    path = next(tmp_path.glob('low-boundary-*.json'))
    before = path.read_bytes()
    count = len(stub_lifter)
    with pytest.raises(FileExistsError, match='new generation take or LOW seed'):
        run(tmp_path, mode='save_new')
    assert path.read_bytes() == before and len(stub_lifter) == count
    _, another = run(tmp_path, mode='save_new', seed=20)
    assert another['first_pass_draft']['cache_id'] != report['first_pass_draft']['cache_id']


def test_opaque_low_executes_but_never_publishes_portable_draft(tmp_path, stub_lifter):  # noqa: F811
    calls = []
    def model():
        value = tiny_model()
        def owner(apply, arguments):
            calls.append(arguments['timestep'])
            return apply(arguments['input'], arguments['timestep'], **arguments['c'])
        value.set_model_unet_function_wrapper(owner)
        return value
    output, report = run(tmp_path, model=model())
    assert all(bool(torch.isfinite(value).all()) for value in output['samples'].unbind())
    assert len(calls) == 2
    assert report['first_pass_draft']['portable_reuse'] is False
    assert report['first_pass_draft']['save_status'] == 'nonportable_not_saved'
    assert not list(tmp_path.glob('low-boundary-*.json'))
    with pytest.raises(ValueError, match='no portable'):
        run(tmp_path, model=model(), mode='reuse_only')
    assert len(calls) == 2


def test_completed_low_survives_failed_high_and_incomplete_low_does_not(tmp_path, stub_lifter, monkeypatch):  # noqa: F811
    original = comfy.utils.ProgressBar.update_absolute
    def cancel_high(self, value, *args, **kwargs):
        if value == 3:
            raise InterruptedError('HIGH cancelled')
        return original(self, value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(comfy.utils.ProgressBar, 'update_absolute', cancel_high)
        with pytest.raises(InterruptedError):
            run(tmp_path)
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 1
    _, report = run(tmp_path, mode='reuse_only')
    assert report['counts']['actual_forwards']['low'] == 0
    def cancel_low(self, value, *args, **kwargs):
        if value == 2:
            raise InterruptedError('LOW cancelled')
        return original(self, value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(comfy.utils.ProgressBar, 'update_absolute', cancel_low)
        with pytest.raises(InterruptedError):
            run(tmp_path / 'incomplete')
    assert not list((tmp_path / 'incomplete').glob('low-boundary-*.json'))


def test_public_draft_replays_in_a_fresh_process(tmp_path, stub_lifter):  # noqa: F811
    baseline, _ = run(tmp_path)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'T8_PUBLIC_DRAFT_CHILD': str(tmp_path)}
    code = ("from comfy.cli_args import args;args.cpu=True;import pytest;"
            "raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',"
            "'tests/test_progressive_first_pass_draft.py::test_public_draft_fresh_process_worker','--tb=short']))")
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    parts = load_file(str(tmp_path / 'child-output.safetensors'))
    for key, expected in zip(('video', 'audio'), baseline['samples'].unbind(), strict=True):
        torch.testing.assert_close(parts[key], expected, rtol=0, atol=0)
    report = json.loads((tmp_path / 'child-report.json').read_text())
    assert report['counts']['callbacks'] == {'low': 0, 'high': 2}
    assert report['noise']['high_seed'] == 20


@pytest.mark.skipif('T8_PUBLIC_DRAFT_CHILD' not in os.environ, reason='isolated public draft worker only')
def test_public_draft_fresh_process_worker(stub_lifter):  # noqa: F811
    root = Path(os.environ['T8_PUBLIC_DRAFT_CHILD'])
    output, report = run(root, mode='reuse_only')
    video, audio = output['samples'].unbind()
    save_file({'video': video, 'audio': audio}, str(root / 'child-output.safetensors'))
    (root / 'child-report.json').write_text(json.dumps(report))


@pytest.mark.parametrize('task,frames,audio_mode', [
    ('t2va', 5, 'native'), ('i2va', 22, 'native'), ('fl2va', 39, 'native'),
    ('ref2va', 22, 'native'), ('t2va', 22, 'lock_source'), ('ref2va', 22, 'remix_source'),
    ('ref2va', 22, 'reference_only')])
def test_public_prepared_context_audio_replays_native_low(tmp_path, accepted, stub_lifter, task, frames, audio_mode):  # noqa: F811
    pair = inputs(capture(accepted, context_frames=frames), task=task, audio_mode=audio_mode)
    def models():
        return dict(model=long_video.patch_long_video_model(tiny_model()),
                    high=long_video.patch_long_video_model(tiny_model()))
    baseline, _ = run(task=task, prepared=pair, **models())
    first, _ = run(tmp_path, task=task, prepared=pair, **models())
    replay, report = run(tmp_path, task=task, prepared=pair, **models())
    same(first, baseline)
    same(replay, baseline)
    assert report['first_pass_draft']['reused_low']
    assert report['prepared_stage_inputs']['audio_time_resampled'] is False
    assert report['counts']['actual_forwards']['low'] == 0


def test_public_legacy_fingerprint_tracks_real_lora_and_unknown_owner(stub_lifter):  # noqa: F811
    source = latent()
    def fingerprint(model):
        model, _, sigmas = setup_dual_clock_sampling(model, source, 8, 12., 3., 'dual_clock_euler', 'native_flow')
        value, = Fingerprint.execute(model=model, positive=conditioning(), av_latent=source,
            sigmas=sigmas[:5], seed=19, scope_json=SCOPE).result
        return json.loads(value)
    plain = fingerprint(long_video.patch_long_video_model(tiny_model()))
    assert plain['portable_reuse']
    assert plain['fingerprint'] == fingerprint(long_video.patch_long_video_model(tiny_model()))['fingerprint']
    model = tiny_model()
    key, weight = next(iter(model.model.named_parameters()))
    model.add_patches({key: ('diff', (torch.full_like(weight, .01),))}, .8)
    assert fingerprint(model)['fingerprint'] != plain['fingerprint']
    model = tiny_model()
    model.set_model_unet_function_wrapper(lambda apply, arguments: apply(arguments['input'], arguments['timestep'], **arguments['c']))
    opaque = fingerprint(model)
    assert opaque['portable_reuse'] is False and opaque['invalid_reason']
    model, _, sigmas = setup_dual_clock_sampling(tiny_model(), source, 8, 12., 3., 'dual_clock_euler', 'native_flow')
    model.get_model_object('model_sampling').foreign_owner = lambda: None
    value, = Fingerprint.execute(model=model, positive=conditioning(), av_latent=source,
        sigmas=sigmas[:5], seed=19, scope_json=SCOPE).result
    unauthenticated = json.loads(value)
    assert unauthenticated['portable_reuse'] is False
    assert unauthenticated['invalid_reason'] == 'Legacy sampling contains unbound execution state'


def test_public_finished_stage_release_preserves_unrelated_loaded_model(monkeypatch):
    from h3_audio_t8_pkg import long_video_dual_residency as residency
    finished, unrelated = tiny_model(), tiny_model()
    first = SimpleNamespace(model=finished, device=torch.device('cpu'), real_model=lambda: finished.model)
    keep = SimpleNamespace(model=unrelated, device=torch.device('cpu'), real_model=lambda: unrelated.model)
    monkeypatch.setattr(residency.mm, 'current_loaded_models', [first, keep])
    releases = []
    monkeypatch.setattr(residency.mm, 'free_memory', lambda amount, device, keep_loaded: releases.append(keep_loaded))
    text, = Release.execute(model=finished).result
    report = json.loads(text)
    assert releases == [[keep]]
    assert report['release']['unrelated_entries_preserved'] == 1
    assert report['numerical_sampling_changed'] is False


def test_internal_finished_stage_policy_preserves_default_outputs(stub_lifter):  # noqa: F811
    baseline, _ = run()
    released, report = run(stage_release='finished_stages')
    same(released, baseline)
    assert [x['stage'] for x in report['stage_residency']['stages']] == [
        'before_learned_lift', 'after_high_sampling']
