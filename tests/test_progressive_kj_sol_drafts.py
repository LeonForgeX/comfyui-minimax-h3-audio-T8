"""Reproduce the active workflow's loader -> KJ Sage -> Sol -> KJ Preview chain.

Actual third-party factories/closures/hooks and Core Euler run on a tiny H3.
Sol GPU kernels are not run: its source adapter's kernel-contract boundary is
explicitly doubled for these CPU mechanics tests. A separate refusal test
proves that this double is not accepted by the production kernel verifier.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import comfy.samplers
import folder_paths
import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.progressive_first_pass_draft import low_model_identity
from h3_audio_t8_pkg import progressive_sol_triton_identity as sol_identity
from h3_audio_t8_pkg import progressive_first_pass_draft as draft_runtime
from test_progressive_lora_loader_drafts import loaded_model
from test_progressive_first_pass_draft import run, same
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_masking import memory_nodes, kj  # noqa: F401
from test_progressive_continuation import accepted, capture  # noqa: F401


def load(name, path, monkeypatch, *, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def third_party(tmp_path, monkeypatch):
    root = Path(folder_paths.__file__).parent / 'custom_nodes'
    preview_path = root / 'ComfyUI-KJNodes/nodes/preview_override_node.py'
    sol_path = root / 'ComfyUI-SolAttn_triton/__init__.py'
    if not preview_path.exists() or not sol_path.exists():
        pytest.skip('actual audited KJ Preview/Sol source checkout is required')
    package = ModuleType('test_kj_preview_nodes')
    package.__path__ = [str(preview_path.parent)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    preview = load(package.__name__ + '.preview_override_node', preview_path, monkeypatch)
    # A real tiny decoder file, loaded through KJ's real loader; no x0 callback
    # or preview function is replaced by the fixture.
    save_file({'decoder.1.weight': torch.full((3, 24, 3, 3), .001),
               'decoder.1.bias': torch.zeros(3)}, str(tmp_path / 'taeh3.safetensors'))
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'vae_approx', ([str(tmp_path)], {'.safetensors'}))
    log = ModuleType('test_sol_triton._autotune_log')
    log.set_verbose = lambda enabled: None
    monkeypatch.setitem(sys.modules, log.__name__, log)
    sol = load('test_sol_triton', sol_path, monkeypatch, package=True)
    # Actual selection/install code needs kernels present, but CPU always takes
    # Sol's existing dense branch. Never claim this exercises the GPU kernels.
    sol._sol_attn_kernel = sol._sol_attn_int8_kernel = lambda *a, **kw: pytest.fail('GPU kernel called on CPU')
    from comfy.ldm.minimax.model import PackedLayout
    original = PackedLayout.__init__
    yield preview, sol
    PackedLayout.__init__ = original
    for name in list(sys.modules):
        if name.startswith('test_sol_triton.') or name.startswith(package.__name__ + '.'):
            sys.modules.pop(name, None)


def stack(tmp_path, memory_nodes, third_party, *, sol=True, preview=True, strength=.3, high=False):
    kj_preview, sol_node = third_party
    model = tiny_model() if high else loaded_model(tmp_path, strength=strength, bypass_strength=.75)
    if not high:
        for extra in (.75, .6, .7):
            model = loaded_model(tmp_path, model=model, bypass=False, strength=extra)
        model = memory_nodes[1].MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
    if sol and not high:
        model = sol_node.SolAttnPatch.execute(model, tau=1.3, start_percent=.2, end_percent=.9,
            min_tokens=12800, int8_qk=True, sink_conditioning='exact_kv_and_rows', morton=False,
            morton_curve='2d_frame', dense_blocks='', verbose=False, use_tma=False,
            int8_pv=True).result[0]
    if preview:
        model.add_wrapper_with_key('outer_sample', 'kj_preview_override',
            kj_preview._PreviewOverrideWrapper(0, '161' if not high else '162', 80, True,
                                              240, 24, tiny_vae='taeh3.safetensors'))
    return model


@pytest.fixture
def cpu_kernel_contract(monkeypatch):
    monkeypatch.setattr(sol_identity, '_kernel_contract', lambda module: {'fixture': 'CPU; GPU kernel NOT_RUN'})


def test_kj_preview_alone_is_read_only_and_draft_identity_neutral(tmp_path, memory_nodes, third_party, stub_lifter):  # noqa: F811
    model = stack(tmp_path, memory_nodes, third_party, sol=False)
    without = model.clone()
    without.remove_wrappers_with_key('outer_sample', 'kj_preview_override')
    assert low_model_identity(model, comfy.samplers.ksampler('euler')) == low_model_identity(without, comfy.samplers.ksampler('euler'))
    result, saved = run(tmp_path / 'drafts', model=model)
    baseline, _ = run(model=without)
    same(result, baseline)
    assert saved['first_pass_draft']['saved_low']
    wrapper = model.wrappers['outer_sample']['kj_preview_override'][0]
    assert wrapper.frames  # actual decoder and callback executed


def test_entire_workflow_chain_retries_same_unaccepted_pack_without_low(tmp_path, memory_nodes, third_party,
        cpu_kernel_contract, stub_lifter, monkeypatch):  # noqa: F811
    def models(strength=.3):
        return dict(model=stack(tmp_path, memory_nodes, third_party, strength=strength),
                    high=stack(tmp_path, memory_nodes, third_party, high=True))
    root = tmp_path / 'drafts'
    from test_progressive_draft_nonportable_cost import count_identities
    captures = count_identities(monkeypatch)
    _, saved = run(root, **models())
    # Initial bind, after saving LOW, and final guard: each model once. No
    # duplicate full-weight scan immediately before the same LOW file write.
    assert len(captures) == 6
    replay, report = run(root, mode='reuse_only', high_seed=97, **models())
    baseline, _ = run(high_seed=97, **models())
    same(replay, baseline)
    assert saved['first_pass_draft']['saved_low']
    assert report['first_pass_draft']['reused_low']
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    receipt, = root.glob('low-boundary-*.json')
    scope = json.loads(receipt.read_text())['contract']['scope']
    assert scope['material_pack_index'] == 2 and not scope['parent_candidate_id']
    with pytest.raises(ValueError, match='compatible|incompatible|no completed receipt'):
        run(root, mode='reuse_only', **models(.8))


@pytest.mark.parametrize('stage', ['low', 'high'])
def test_post_write_guard_catches_weight_change_before_receipt_publication(tmp_path, memory_nodes,
        third_party, cpu_kernel_contract, stub_lifter, monkeypatch, stage):  # noqa: F811
    low = stack(tmp_path, memory_nodes, third_party)
    high = stack(tmp_path, memory_nodes, third_party, high=True)
    original = draft_runtime.save_file
    def mutate_during_write(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            selected = low if stage == 'low' else high
            # Change an original unpatched parameter. Mutating an already
            # applied LoRA tensor is not a change to the saved base+LoRA model;
            # Core deliberately reconstructs that base from its backup.
            _, parameter = next((name, p) for name, p in selected.model.named_parameters()
                                if name not in selected.backup and name not in selected.patches)
            parameter.add_(.01)
    monkeypatch.setattr(draft_runtime, 'save_file', mutate_during_write)
    root = tmp_path / 'drafts'
    with pytest.raises(ValueError, match='execution inputs changed'):
        run(root, model=low, high=high)
    assert not list(root.glob('low-boundary-*.json'))


def test_cpu_kernel_double_is_declined_without_test_injection(tmp_path, memory_nodes, third_party):
    model = stack(tmp_path, memory_nodes, third_party)
    assert low_model_identity(model, comfy.samplers.ksampler('euler'))['weights']['portable_cache_reuse'] is False


def test_full_chain_prepared_ref2va_six_low_two_high(tmp_path, memory_nodes, third_party,
        cpu_kernel_contract, stub_lifter, accepted):  # noqa: F811
    from test_progressive_stage_inputs import inputs
    from h3_audio_t8_pkg import long_video
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    # Match the active workflow's reference/context route and 6+2 schedule;
    # media/conditioning fixtures contain no uploaded private workflow data.
    source = capture(accepted, context_frames=22)
    pair = inputs(source, task='ref2va')
    def models():
        return dict(model=long_video.patch_long_video_model(stack(tmp_path, memory_nodes, third_party)),
                    high=long_video.patch_long_video_model(stack(tmp_path, memory_nodes, third_party, high=True)))
    values = dict(task='ref2va', prepared=pair, sigmas=native_flow_sigmas(8, 12.), low_evaluations=6)
    cache = tmp_path / 'six-low-drafts'
    _, saved = run(cache, **models(), **values)
    replay, report = run(cache, mode='reuse_only', high_seed=97, **models(), **values)
    baseline, _ = run(high_seed=97, **models(), **values)
    same(replay, baseline)
    assert saved['counts']['actual_forwards'] == {'low': 6, 'high': 2}
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    assert report['first_pass_draft']['reused_low']


@pytest.mark.parametrize('change', ['hook', 'forward', 'preview', 'preview_helper', 'compose', 'gate'])
def test_full_chain_does_not_whitelist_replacements(tmp_path, memory_nodes, third_party, cpu_kernel_contract, change):
    model = stack(tmp_path, memory_nodes, third_party)
    sampler = comfy.samplers.ksampler('euler')
    assert low_model_identity(model, sampler)['weights'].get('portable_cache_reuse') is not False
    def unknown(*args, **kwargs):
        pytest.fail('inspection executed an unknown function')
    block = model.model.diffusion_model.blocks[0]
    if change == 'hook':
        block.register_forward_pre_hook(unknown)
    elif change == 'forward':
        block.attn.forward = unknown
    elif change == 'preview':
        model.wrappers['outer_sample']['kj_preview_override'] = [unknown]
    elif change == 'preview_helper':
        third_party[0]._normalize_packed_x0 = unknown
    elif change == 'compose':
        model.object_patches['diffusion_model.blocks.0.attn.forward'] = unknown
    else:
        model.model_options['transformer_options']['sol_compose']['min_tokens'] += 1
    assert low_model_identity(model, sampler)['weights']['portable_cache_reuse'] is False
