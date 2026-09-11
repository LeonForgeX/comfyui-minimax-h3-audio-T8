import json

import pytest
from comfy_api.latest import InputImpl

from h3_audio_t8_pkg import nodes_topaz, topaz_contract, topaz_runtime
from tests.test_topaz_contract import runtime  # noqa: F401


def test_fixed_scale_geometry_remains_unchanged():
    width, height, report = topaz_contract.resolve_output_geometry(1024, 512, None, None, 2)
    assert (width, height) == (2048, 1024)
    assert report['ratio'] == '2'
    assert report['size_mode'] == 'scale'


def test_explicit_noninteger_ratio_preserves_aspect_and_ignores_fixed_selector():
    width, height, report = topaz_contract.resolve_output_geometry(1024, 512, 1536, 768, 4, 'target_dimensions')
    assert (width, height) == (1536, 768)
    assert report['ratio'] == '3/2'
    assert report['fixed_scale_selector_ignored'] is True


@pytest.mark.parametrize('width,height', [(1536, 800), (1537, 768), (None, 768),
    (0, 0), (True, 512), (512, 256), (5120, 2560), (9000, 4500)])
def test_invalid_custom_geometry_never_stretches_or_silently_rounds(width, height):
    with pytest.raises(ValueError):
        topaz_contract.resolve_output_geometry(1024, 512, width, height, 2, 'target_dimensions')


def test_fixed_mode_still_rejects_an_unmatched_explicit_size():
    with pytest.raises(ValueError):
        topaz_contract.resolve_output_geometry(1024, 512, 1536, 768, 2)


def test_dimensions_inputs_are_optional_and_appended_for_old_workflows():
    cls = nodes_topaz.MiniMaxH3TopazVideoEXPT8
    assert [item.id for item in cls.define_schema().inputs][:6] == [
        'topaz_runtime', 'source_video', 'model_id', 'scale', 'vram_fraction', 'parameters_json']
    optional = cls.GET_NODE_INFO_V1()['input'].get('optional', {})
    assert {'size_mode', 'target_width', 'target_height'} <= set(optional)


def test_public_node_forwards_requested_dimensions_without_decoding_source(runtime, monkeypatch, tmp_path):  # noqa: F811
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'not-decoded')
    video = InputImpl.VideoFromFile(str(source))
    monkeypatch.setattr(video, 'get_dimensions', lambda: pytest.fail('Parent decoded input'))
    monkeypatch.setattr(nodes_topaz.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    seen = {}
    def run(_runtime, _source, _job, **settings):
        seen.update(settings)
        return tmp_path / 'enhanced.mov', {'status': 'fixture_only'}
    monkeypatch.setattr(topaz_runtime, 'run_regular', run)
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
              'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    result = nodes_topaz.MiniMaxH3TopazVideoEXPT8.execute(handle, video, 'iris-3', '2x',
        size_mode='target_dimensions', target_width=1536, target_height=768)
    assert seen['size_mode'] == 'target_dimensions'
    assert (seen['width'], seen['height']) == (1536, 768)
    assert result.result[1] is video


def test_custom_evidence_binds_present_variants_without_claiming_selected_engine_scale(runtime):  # noqa: F811
    path = runtime.definitions / 'iris-3.json'
    definition = json.loads(path.read_text())
    definition.update(shortName='iris', version='3')
    path.write_text(json.dumps(definition))
    for scale in (1, 2):
        (runtime.data / f'iris-v3-fgnet-576x384-{scale}x-ox.tz3').write_bytes(b'fixture')
    evidence = topaz_runtime.dimension_model_evidence(runtime, 'iris-3')
    assert evidence['scale'] == 'engine_auto_dimensions'
    assert len(evidence['candidate_weights']) == 2
    assert evidence['missing_candidate_scales'] == [4]
    assert evidence['actual_engine_variant_verified'] is False


def test_custom_command_sizes_after_ai_and_records_native_output(runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'in.mp4'
    source.write_bytes(b'fixture')
    command = topaz_contract.regular_command(runtime, source, tmp_path / 'out.mov',
        'iris-3', 1536, 768, size_mode='target_dimensions')
    filters = command[command.index('-vf') + 1]
    assert filters.startswith('tvai_up=')
    assert filters.index('tvai_up=') < filters.index('showinfo@t8_topaz_native') < filters.index('scale=w=1536:h=768')
    assert 'flags=lanczos:threads=2' in filters
    assert command[command.index('-c:a') + 1] == 'copy'
    assert 'download=0' in filters


def test_fixed_command_has_no_added_resampling(runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'in.mp4'
    source.write_bytes(b'fixture')
    command = topaz_contract.regular_command(runtime, source, tmp_path / 'out.mov',
        'iris-3', 2048, 1024)
    assert ',scale=' not in command[command.index('-vf') + 1]


def test_native_size_trace_is_not_a_claim_of_exact_weight_variant():
    trace = '\n'.join(f'[showinfo@t8_topaz_native @ 123] n: {n} pts: {n} fmt:rgb48le s:2048x1024 i:P' for n in range(2))
    result = topaz_contract.native_dimension_evidence(trace, 2)
    assert (result['width'], result['height']) == (2048, 1024)
    assert result['exact_weight_variant_verified'] is False
    for broken in ('', trace.splitlines()[0], trace.replace('n: 1', 'n: 0'),
                   trace.replace('n: 1 pts: 1 fmt:rgb48le s:2048x1024', 'n: 1 pts: 1 fmt:rgb48le s:1024x512')):
        with pytest.raises(ValueError):
            topaz_contract.native_dimension_evidence(broken, 2)
