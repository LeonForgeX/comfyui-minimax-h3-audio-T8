"""Opt-in real CUDA kernels on random tiny H3, with a labelled fake lifter.

Run only in a separate, explicitly GPU-enabled process under SerialProbeLease.
This is NOT trained H3/learned-upscaler/video quality or full-model VRAM evidence.
Ordinary CPU regression skips every case; no GPU allocations by this module.
"""

import gc
import ast
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import FunctionType, ModuleType

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('T8_RUN_PROGRESSIVE_TINY_CUDA') != '1',
                                reason='explicit isolated CUDA probe only')


def gpu_model():
    import comfy.latent_formats
    import comfy.model_base
    import comfy.model_patcher
    import comfy.ops
    import comfy.supported_models_base

    class Config(comfy.supported_models_base.BASE):
        latent_format = comfy.latent_formats.MiniMaxH3AV
        unet_extra_config = {}
        sampling_settings = {'shift': 12., 'audio_shift': 3.}
        custom_operations = comfy.ops.disable_weight_init

    config = Config(dict(hidden_size=256, num_layers=2, token_refiner_num_layers=0,
        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=512,
        text_dim=8, timestep_input_dim=16, time_embed_hidden_size=128, time_embed_dim=64,
        rope_inv_freq_len=16, dtype=torch.bfloat16))
    base = comfy.model_base.MiniMaxH3(config, device=torch.device('cpu'))
    generator = torch.Generator(device='cpu').manual_seed(43)
    with torch.no_grad():
        for p in base.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * .03)
        base.diffusion_model.rope.inv_freq.fill_(1.)
    return comfy.model_patcher.ModelPatcher(base, torch.device('cuda:0'), torch.device('cpu'))


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_kj_h3_definitions(path):
    """Original live code, without unrelated server route registration."""
    import comfy.model_management as mm
    import comfy.quant_ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy_api.latest import io
    from sageattention.core import get_cuda_arch_versions
    from h3_audio_t8_pkg.relay_kj_backend import _codes

    name = 'tiny_cuda_kj_h3_definitions'
    if name in sys.modules:
        return sys.modules[name]
    payload = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, mm=mm, _ck=comfy.quant_ops.ck, io=io,
        logging=logging, _cuda_archs=get_cuda_arch_versions(), _MiniMaxH3Model=MiniMaxH3Model)
    sys.modules[name] = module
    codes = tuple(_codes(compile(payload, str(path), 'exec', dont_inherit=True)))
    code = next(code for code in codes if code.co_name == 'minimax_sageattn_forward')
    module.minimax_sageattn_forward = FunctionType(code, vars(module), argdefs=(None, {}))
    declaration = next(node for node in ast.parse(payload).body if isinstance(node, ast.ClassDef)
        and node.name == 'MiniMaxH3MemoryEfficientSageAttentionPatch')
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(path), 'exec', dont_inherit=True), vars(module))

    def unadapted_kernel_is_not_allowed(*args, **kwargs):
        raise AssertionError('The progressive composer must own the real attention delegate')

    # This guard never returns an attention result. The runtime's scoped KJ
    # forward must dispatch the real installed Sage kernel through its owner.
    module._sageattn_int8_fp8_nhd = unadapted_kernel_is_not_allowed
    return module


def select_backend(model, backend, monkeypatch):
    from comfy.ldm.modules import attention
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend

    root = Path(attention.__file__).resolve().parents[3] / 'custom_nodes'
    if backend == 'core_sage':
        assert attention.SAGE_ATTENTION_IS_AVAILABLE
        set_h3_attention_backend(model, attention.attention_sage)
        return model
    if backend == 'kj_memory':
        # Execute the installed definitions, not CPU fixtures or plugin __init__.
        # Repository CPU tests put h3_t8 on sys.path. Installed plugins expect
        # bare `nodes` to resolve to Core, as in an actual ComfyUI process.
        monkeypatch.syspath_prepend(str(root.parent))
        lowmem = load_module('tiny_cuda_kj_minimax', root / 'ComfyUI-KJNodes/nodes/minimax_nodes.py')
        sage = load_kj_h3_definitions(root / 'ComfyUI-KJNodes/nodes/ltxv_nodes.py')
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
        return lowmem.MiniMaxChunkFeedForward.execute(model, 2, 256).result[0]
    if 'tiny_cuda_sol' not in sys.modules:
        package = ModuleType('tiny_cuda_sol')
        package.__path__ = [str(root / 'ComfyUI-sol-attn')]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    module = importlib.import_module('tiny_cuda_sol.nodes')
    return module.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]


def relay_pair(model):
    from h3_audio_t8_pkg import prompt_relay_advanced as relay

    embedding = torch.tensor([[[-1., -.7, -.4, -.1, .2, .5, .8, 1.],
                                [1., .8, .5, .2, -.1, -.4, -.7, -1.]]])
    layout = relay.build_packed_layout(2, 37, 8, 16, 207, keyframes=[], refs=[], frame_count=124)
    binding = relay._bind_layout_contract({
        'schema': relay.PROMPT_RELAY_PATCH_VERSION, 'plan_hash': 'tiny-cuda-two-events',
        'text_len': 2, 'query_route': 'joint_av_exp',
        'events': [{'text_key_start': i, 'text_key_end': i + 1, 'midpoint': float(i * 90),
                    'window': .1, 'sigma': 10.} for i in range(2)],
    }, layout, resolved_task='t2va', keyframes=[], refs=[])
    cond = [[embedding, {'minimax_token_tags': torch.ones(2, dtype=torch.long),
                         relay.PROMPT_RELAY_BINDING_KEY: binding}]]
    cond = relay._attach_binding_model_cond(cond, binding['binding_hash'])
    return relay.patch_prompt_relay_model(model, binding, 256)[0], cond


@pytest.mark.parametrize('backend', ['core_sage', 'kj_memory', 'sol'])
@pytest.mark.parametrize('relay', [False, True])
@pytest.mark.parametrize('eav_mode', ['apply_exp', 'disabled'])
@pytest.mark.parametrize('tst_mode', ['disabled', 'apply_exp'], ids=['tst_off', 'tst_on'])
@pytest.mark.parametrize('tst_interface', ['internal', 'model_nodes'])
def test_real_cuda_masked_eav_dual_models_with_lora(monkeypatch, record_property, backend, relay, eav_mode, tst_mode, tst_interface):
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.samplers
    from comfy.ldm.modules import attention
    from comfy.weight_adapter.lora import LoRAAdapter
    from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
    from h3_audio_t8_pkg.sampling import native_flow_sigmas

    assert os.environ.get('T8_PROGRESSIVE_GPU_LEASE_HELD') == '1'
    assert torch.cuda.device_count() == 1
    assert torch.cuda.mem_get_info(0)[0] > 4 * 1024**3
    # Trace the actual Core Sage entry, never substitute its kernel. If Core
    # catches a kernel error, attempts > completions makes this probe fail.
    sage_calls = {'attempted': 0, 'completed': 0, 'with_mask': 0}
    real_sage = attention.sageattn

    def traced_sage(*args, **kwargs):
        sage_calls['attempted'] += 1
        result = real_sage(*args, **kwargs)
        sage_calls['completed'] += 1
        sage_calls['with_mask'] += int(kwargs.get('attn_mask') is not None)
        return result

    monkeypatch.setattr(attention, 'sageattn', traced_sage)
    low, high = (select_backend(gpu_model(), backend, monkeypatch) for _ in range(2))
    positive = [[torch.zeros(1, 2, 8), {'minimax_token_tags': torch.ones(2, dtype=torch.long)}]]
    if relay:
        low, positive = relay_pair(low)
    adapters, lora_calls = [], []
    original_calculate = LoRAAdapter.calculate_weight

    def calculate(self, *args, **kwargs):
        result = original_calculate(self, *args, **kwargs)
        lora_calls.append(self)
        return result

    monkeypatch.setattr(LoRAAdapter, 'calculate_weight', calculate)
    for model in (low, high):
        key, p = next((k, p) for k, p in model.model.named_parameters() if p.ndim == 2)
        for strength in (.7, .2):
            adapter = LoRAAdapter(set(), (torch.full((p.shape[0], 2), .01),
                torch.full((2, p.shape[1]), .02), 2., None, None, None))
            assert model.add_patches({key: adapter}, strength_patch=strength)
            adapters.append(adapter)
    generator = torch.Generator().manual_seed(6)
    video = torch.randn((1, 24, 37, 8, 16), generator=generator) * .1
    audio = torch.randn((1, 32, 2, 207), generator=generator) * .1
    vm, am = torch.ones(1, 1, 37, 8, 16), torch.ones(1, 1, 2, 207)
    vm[:, :, :7] = 0.
    am[..., :36] = 0.
    av = {'samples': comfy.nested_tensor.NestedTensor((video, audio)),
          'noise_mask': comfy.nested_tensor.NestedTensor((vm, am))}
    monkeypatch.setattr(runtime, '_lifter_identity', lambda name, precision:
                        {'name': name, 'precision': precision, 'path': 'labelled-test-double', 'sha256': 'a' * 64})

    def fake_lift(v, a, plan, identity):
        result = torch.nn.functional.interpolate(v.float(),
            size=(v.shape[2], plan.target_height // 16, plan.target_width // 16),
            mode='trilinear', align_corners=False)
        return result, {'test_double': True, 'trained_lifter_executed': False}

    monkeypatch.setattr(runtime, '_lift_video', fake_lift)
    device_calls = []
    handles = [m.model.diffusion_model.register_forward_pre_hook(
        lambda module, args, phase=phase: device_calls.append((phase, [str(p.device) for p in args[0]])))
        for phase, m in (('low', low), ('high', high))]
    if tst_interface == 'model_nodes':
        from h3_audio_t8_pkg.nodes_tst import MiniMaxH3TSTModelEXPT8
        low, high = [MiniMaxH3TSTModelEXPT8.execute(model=m, sigmas=native_flow_sigmas(8, 12.),
            mode=tst_mode, tau=.2, max_workspace_mib=256).result[0] for m in (low, high)]
    try:
        output, text = runtime.sample_progressive_h3(low, positive,
            [[torch.zeros(1, 2, 8), {'minimax_token_tags': torch.ones(2, dtype=torch.long)}]],
            av, comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
            model_hires=high, upscaler_model='labelled-test-double', seed=6,
            low_evaluations=4, input_mode='initialized_av_exp', eav_mode=eav_mode, eav_tau=.2,
            reserve_vram_mib=2048, tst_mode=tst_mode if tst_interface == 'internal' else 'disabled')
        torch.cuda.synchronize(0)
        report = json.loads(text)
        record_property('report_json', text)
        record_property('gpu', torch.cuda.get_device_name(0))
        record_property('sage_actual_kernel_calls', json.dumps(sage_calls))
        record_property('qualification', 'random_tiny_H3_real_CUDA_fake_lifter_not_trained_media')
        assert device_calls == [(p, ['cuda:0', 'cuda:0']) for p in ('low', 'high') for _ in range(4)]
        assert all(any(a is called for called in lora_calls) for a in adapters)
        for actual, expected, mask in zip(output['samples'].unbind(), (video, audio), (vm, am)):
            selected = mask.expand_as(expected) == 0
            torch.testing.assert_close(actual.cpu()[selected], expected[selected], atol=2e-6, rtol=2e-6)
            assert torch.isfinite(actual).all()
        for phase in ('low', 'high'):
            if tst_mode == 'apply_exp':
                tst = report['tst'][phase]
                assert tst['completed']
                assert [f['step_index'] for f in tst['forwards']] == (
                    [0, 1, 2, 3] if phase == 'low' else [4, 5, 6, 7])
                assert all(f['query_transform_calls'] == 2 for f in tst['forwards'])
                assert tst['forwards'][0]['applied_layers'] > 0
            else:
                assert report['tst']['mode'] == 'disabled'
            if eav_mode == 'apply_exp':
                assert report['eav'][phase]['verified_native_mask_forwards'] == 4
            else:
                assert report['eav']['mode'] == 'disabled'
            assert report['stage_models'][phase]['weight_patch_entries'] == 2
            calls = report['attention'][phase].get('completed_calls', {})
            if backend == 'sol':
                if relay:
                    assert calls.get('sol:completed', 0) == 0
                    assert any(k.startswith('fallback:') and n > 0 for k, n in calls.items())
                else:
                    assert calls.get('sol:completed', 0) == 8
            elif backend == 'core_sage':
                assert sage_calls['completed'] > 0
                assert sage_calls['attempted'] == sage_calls['completed']
                if relay:
                    if attention.SAGE_ATTENTION_SUPPORTS_MASK:
                        assert sage_calls['with_mask'] > 0
                    else:
                        masked = report['attention'][phase]['scoped_masked_sage']
                        assert masked['completed_calls']['sage:biased'] > 0
                        assert not any(k.startswith('pytorch:') and n > 0 for k, n in masked['completed_calls'].items())
            else:
                assert sum(calls.values()) > 0
                assert not any(k.startswith(('fallback:', 'pytorch:')) and n > 0 for k, n in calls.items())
    finally:
        for handle in handles:
            handle.remove()
        mm.unload_all_models()  # Only this explicitly isolated CUDA probe process.
        del low, high
        gc.collect()
        mm.soft_empty_cache()
