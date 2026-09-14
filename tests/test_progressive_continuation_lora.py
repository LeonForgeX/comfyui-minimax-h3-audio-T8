"""Native continuation with two independent real Core LoRAAdapter chains.

Tiny CPU H3, installed backend definitions, explicit VAE/lifter doubles.
Backend fallback is evidence of compatibility, not CUDA acceleration.
"""

import json

import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter

from test_progressive_continuation_runtime import run
from test_progressive_continuation import accepted, capture  # noqa: F401
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from test_progressive_first_segment import plan
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_relay_kj_backend import kj as kj  # noqa: F401

memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def stack(model, seed):
    key, weight = next((k, w) for k, w in model.model.named_parameters() if w.ndim == 2)
    expected = weight.detach().clone()
    rng = torch.Generator().manual_seed(seed)
    for strength, model_strength in ((.7, 1.), (.2, .9)):
        up = torch.randn(weight.shape[0], 2, generator=rng) * .03
        down = torch.randn(2, weight.shape[1], generator=rng) * .03
        adapter = LoRAAdapter(set(), (up, down, 2., None, None, None))
        assert model.add_patches({key: adapter}, strength_patch=strength, strength_model=model_strength) == [key]
        # alpha/rank=1. The second model-strength scaling makes order observable.
        expected = expected * model_strength + strength * (up @ down)
    return key, expected


@pytest.mark.parametrize('backend', ['pytorch', 'kj_memory', 'sol'])
@pytest.mark.parametrize('tst_mode', ['disabled', 'apply_exp'])
def test_independent_lora_chains_reach_native_continuation_forwards(
        accepted, stub_lifter, memory_nodes, installed_sol, backend, tst_mode):  # noqa: F811
    models, checks, snapshots = [], [], []
    for seed in (101, 202):
        model = tiny_model()
        if backend == 'kj_memory':
            lowmem, sage, _ = memory_nodes
            model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
            model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
        elif backend == 'sol':
            model = installed_sol.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]
        key, expected = stack(model, seed)
        models.append(model)
        checks.append((key, expected))
        snapshots.append({k: list(v) for k, v in model.patches.items()})
    assert not torch.equal(checks[0][1], checks[1][1])
    seen = [0, 0]
    handles = []
    for index, model in enumerate(models):
        def observe(module, args, index=index, model=model):
            key, expected = checks[index]
            actual = dict(model.model.named_parameters())[key]
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            seen[index] += 1
        handles.append(model.model.diffusion_model.register_forward_pre_hook(observe))
    try:
        output, text = run(capture(accepted), models[0], model_hires=models[1], prompt=None,
            clip=NativeLikeFakeClip(), prompt_relay_plan=plan(route='joint_av_exp'),
            eav_mode='apply_exp', eav_tau=.2, tst_mode=tst_mode)
    finally:
        for handle in handles:
            handle.remove()
    report = json.loads(text)
    assert seen == [4, 4]
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    for phase, model, snapshot in zip(('low', 'high'), models, snapshots, strict=True):
        assert model.patches == snapshot
        assert report['stage_models'][phase]['weight_patch_entries'] == 2
        assert report['prompt_relay'][phase]['completed_calls']['forward'] == 4
        assert report['eav'][phase]['verified_native_mask_forwards'] == 4
        if tst_mode != 'disabled':
            assert report['tst'][phase]['completed']
            assert [f['step_index'] for f in report['tst'][phase]['forwards']] == (
                [0, 1, 2, 3] if phase == 'low' else [4, 5, 6, 7])
            assert all(f['query_transform_calls'] == len(model.model.diffusion_model.blocks)
                       for f in report['tst'][phase]['forwards'])
        if backend != 'pytorch':
            calls = report['attention'][phase]['completed_calls']
            assert sum(calls.values()) > 0
            assert calls.get('sol:completed', 0) == 0
