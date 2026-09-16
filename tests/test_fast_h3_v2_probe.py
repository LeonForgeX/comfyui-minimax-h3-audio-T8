from pathlib import Path
import importlib.util
import sys

import pytest


def _tool():
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("v2_probe", tools / "run_fast_h3_v2_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ("trained_vsa_exp", "dense_compat_exp", "official_comfy_template_exp"))
@pytest.mark.parametrize("backend", ("pytorch", "kj_sage"))
@pytest.mark.parametrize("memory", ((1, 1), (4, 2)))
def test_source_bound_graph_routes_and_audit_dependencies(profile, backend, memory):
    tool = _tool()
    graph, reports = tool.build_graph(profile=profile, backend=backend, head_chunks=memory[0], ffn_chunks=memory[1])
    assert all(item["class_type"] != "MiniMaxH3LoRACompatibilityLoaderT8Advanced" for item in graph.values())
    assert graph["1"]["inputs"]["unet_name"] == tool.MODEL
    assert graph["19"]["inputs"]["sampled_av_latent"] == ["13", 0]
    assert graph["14"]["inputs"]["av_latent"] == ["19", 0]
    assert graph["16"]["inputs"]["codec"] == "h264"
    assert graph["10"]["inputs"]["profile"] == profile
    assert reports["runtime"] == "20"
    for item in graph.values():
        for value in item["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_unknown_recipe_is_not_silently_accepted():
    with pytest.raises(ValueError):
        _tool().build_graph(profile="old_four_step")


def test_production_graph_excludes_probe_only_nodes_and_preserves_reference_aspect():
    graph, reports = _tool().build_graph(instrument=False, width=512, height=768,
                                         first_frame="portrait.png")
    assert graph['13']['class_type'] == 'SamplerCustomAdvanced'
    assert graph['9']['inputs']['first_frame'] == ['23', 0]
    assert graph['9']['inputs']['task_type'] == 'I2VA'
    from h3_audio_t8_pkg.conditioning import resolve_task_type
    assert resolve_task_type(graph['9']['inputs']['task_type'], object(), None, False) == 'i2va'
    assert 'network' not in reports
    assert not any(node['class_type'].startswith('T8FastH3') for node in graph.values())


@pytest.mark.parametrize('first_frame', ['../outside.png', 'F:/outside.png', ''])
def test_reference_path_is_not_implicitly_imported(first_frame):
    with pytest.raises(ValueError, match='input-relative'):
        _tool().build_graph(first_frame=first_frame)


@pytest.mark.parametrize('change', [dict(width=0), dict(height=-32), dict(frames=72),
                                    dict(fps='25/1'), dict(audio_seconds='0.8')])
def test_media_contract_detects_wrong_canvas_frames_fps_or_audio(change):
    media = dict(width=832, height=480, frames=73, fps='24/1', strict_decode=True,
                 video_seconds=str(73/24), audio_seconds=str(73/24))
    media.update(change)
    with pytest.raises(RuntimeError):
        _tool().validate_media_contract(media, width=832, height=480, frames=73)


def test_exact_three_second_grid_media_contract_passes_without_quality_claim():
    _tool().validate_media_contract(dict(width=832, height=480, frames=73, fps='24/1',
        strict_decode=True, video_seconds=str(73/24), audio_seconds=str(73/24)),
        width=832, height=480, frames=73)


@pytest.mark.parametrize('change', [None, 'source', 'core', 'asset'])
def test_final_input_gate_covers_new_backend_observer_source_core_and_checkpoint(monkeypatch, change):
    tool = _tool()
    current = {'h3_t8/fast_h3_v2_advanced.py': 'frozen'}
    monkeypatch.setattr(tool.transport, 'source_snapshot', lambda: current.copy())
    monkeypatch.setattr(tool, 'verify_core_source', lambda _: {'revision': 'changed' if change == 'core' else 'pinned'})
    def identify(path):
        digest = path.name
        if change == 'asset' and path.name == 'checkpoint':
            digest = 'changed'
        return {'sha256': digest}
    monkeypatch.setattr(tool, 'file_identity', identify)
    expected = {'sources': {**current, 'tools/run_fast_h3_v2_probe.py': 'run_fast_h3_v2_probe.py',
        'tools/fast_h3_v2_backend_audit.py': 'fast_h3_v2_backend_audit.py'},
        'core': {'revision': 'pinned'}, 'assets': [{'path': 'checkpoint', 'sha256': 'checkpoint'}]}
    if change == 'source':
        expected['sources']['tools/fast_h3_v2_backend_audit.py'] = 'different'
    if change:
        with pytest.raises(RuntimeError, match='changed'):
            tool.verify_final_inputs(expected, Path('core'))
    else:
        tool.verify_final_inputs(expected, Path('core'))


@pytest.mark.parametrize('backend,plugin', [('pytorch', None), ('kj_sage', 'ComfyUI-KJNodes'),
                                           ('sol', 'ComfyUI-sol-attn')])
def test_custom_model_backend_does_not_change_global_encoder_attention(backend, plugin):
    tool = _tool()
    original = ['python', 'main.py', '--whitelist-custom-nodes', 'candidate']
    result = tool.isolated_backend_command(original, backend)
    assert original == ['python', 'main.py', '--whitelist-custom-nodes', 'candidate']
    assert result.count('--use-pytorch-cross-attention') == 1
    if plugin:
        assert result[result.index('--whitelist-custom-nodes')+1] == plugin
    assert tool.isolated_backend_command(result, 'pytorch').count('--use-pytorch-cross-attention') == 1


def test_owned_test_server_headroom_is_explicit_not_a_production_admission_gate():
    tool = _tool()
    original = ['python', 'main.py', '--whitelist-custom-nodes', 'candidate']
    assert '--vram-headroom' not in tool.isolated_backend_command(original, 'pytorch')
    result = tool.isolated_backend_command(original, 'kj_sage', 2)
    assert result[-2:] == ['--vram-headroom', '2']
    assert '--vram-headroom' not in original
    with pytest.raises(ValueError):
        tool.isolated_backend_command(original, 'pytorch', -1)
    with pytest.raises(ValueError, match='override'):
        tool.isolated_backend_command(result, 'pytorch', 2)


@pytest.mark.parametrize('outcome', ['pass', 'short', 'network_failure', 'nan'])
def test_isolated_v2_observer_counts_completed_calls_and_always_removes_clone_wrapper(monkeypatch, outcome):
    import json
    from types import SimpleNamespace
    import torch
    from comfy_extras import nodes_custom_sampler as native
    from tools import progressive_probe_extension as extension

    class Model:
        def __init__(self):
            self.wrappers = {}
        def clone(self):
            self.cloned = Model()
            return self.cloned
        def add_wrapper_with_key(self, role, key, value):
            self.wrappers[key] = value
        def remove_wrappers_with_key(self, role, key):
            self.wrappers.pop(key)

    model = Model()
    receipt = SimpleNamespace(profile='trained_vsa_exp')
    module = SimpleNamespace(capture_fast_h3_v2_owner=lambda model: receipt)
    parts = [torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 2)]
    monkeypatch.setattr(extension, 'project_module', lambda name: module if name == 'fast_h3_v2_advanced'
                        else SimpleNamespace(nested_av_parts=lambda value: parts))
    monkeypatch.setattr(native.RandomNoise, 'execute', lambda seed: SimpleNamespace(result=[seed]))
    monkeypatch.setattr(native.BasicGuider, 'execute', lambda clone, positive: SimpleNamespace(result=[clone]))
    def execute(noise, clone, sampler, sigmas, source):
        observer = next(iter(clone.wrappers.values()))
        def network(*args, **kwargs):
            if outcome == 'network_failure':
                raise RuntimeError('network_failed')
            return parts
        for _ in range(7 if outcome == 'short' else 8):
            observer(network, parts, torch.ones(1))
        if outcome == 'nan':
            parts[1].fill_(float('nan'))
        return SimpleNamespace(result=[source])
    monkeypatch.setattr(native.SamplerCustomAdvanced, 'execute', execute)
    if outcome == 'pass':
        raw = extension.FastH3V2SamplerProbe.execute(model, [], {}, object(), torch.ones(9), 7).result[1]
        assert json.loads(raw)['completed_network_forwards'] == 8
        assert json.loads(raw)['output_finite'] is True
    else:
        with pytest.raises(RuntimeError):
            extension.FastH3V2SamplerProbe.execute(model, [], {}, object(), torch.ones(9), 7)
    assert not model.wrappers and not model.cloned.wrappers
