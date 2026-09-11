import json

import pytest
from comfy_api.latest import InputImpl

from h3_audio_t8_pkg import nodes_topaz as nodes
from h3_audio_t8_pkg import topaz_runtime
from tests.test_topaz_contract import runtime  # noqa: F401


def test_optional_nodes_have_no_license_checkbox_or_implicit_install():
    classes = nodes.TOPAZ_NODE_CLASSES
    assert len(classes) == 2
    for cls in classes:
        schema = cls.define_schema()
        assert schema.is_experimental and schema.is_output_node
        ids = [item.id for item in schema.inputs]
        assert not any('license' in key or 'download' in key for key in ids)
    defaults = {item.id: item.default for item in classes[0].define_schema().inputs}
    assert set(defaults.values()) == {''}


def test_empty_environment_cannot_treat_current_directory_as_install():
    with pytest.raises(ValueError, match='Topaz'):
        nodes.MiniMaxH3TopazEnvironmentEXPT8.execute('', '', '')


def test_file_video_is_forwarded_without_parent_decode(monkeypatch, runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'fixture-not-decoded')
    video = InputImpl.VideoFromFile(str(source))
    monkeypatch.setattr(video, 'get_dimensions', lambda: pytest.fail('Parent tried to decode media'))
    monkeypatch.setattr(nodes.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    observed = {}
    def run(env, path, job, **settings):
        observed.update(settings)
        assert path == source and env.install == runtime.install
        return tmp_path / 'result.mkv', {'status': 'fixture_only'}
    monkeypatch.setattr(topaz_runtime, 'run_regular', run)
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
        'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    result = nodes.MiniMaxH3TopazVideoEXPT8.execute(handle, video, 'iris-3', '2x', parameters_json='{"noise":0.3}')
    assert observed['width'] is None and observed['height'] is None
    assert observed['scale'] == 2 and observed['parameters'] == {'noise': .3}
    assert result.result[1] is video
    assert json.loads(result.result[3])['status'] == 'fixture_only'
