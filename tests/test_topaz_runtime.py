from copy import deepcopy
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg.topaz_runtime import (
    TOPAZ_STARTUP_FREE_RAM_BYTES,
    TaskProgress,
    discover_official_installation,
    discover_official_installations,
    model_evidence,
    record_topaz_startup_sample,
    validate_signatures,
)
from h3_audio_t8_pkg.dlss_fi_backend.resources import ResourceGuard
from tests.test_topaz_contract import runtime  # noqa: F401


def test_topaz_discovery_is_read_only_and_reports_complete_candidate(tmp_path):
    install = tmp_path / 'Topaz Video AI'
    models = tmp_path / 'models'
    install.mkdir()
    models.mkdir()
    for name in ('Topaz Video AI.exe', 'ffmpeg.exe', 'ffprobe.exe'):
        (install / name).write_bytes(b'fixture')
    (models / 'iris-3.json').write_text('{"backends": {}}', encoding='utf8')
    (models / 'iris-v3-fgnet-fp32-384x672-2x-rt809-10800-rt.tz').write_bytes(b'weights')
    rows = discover_official_installations(installs=[install], model_roots=[models])
    assert rows[0]['status'] == 'ready'
    assert rows[0]['model_definition_count'] == 1
    assert rows[0]['model_weight_count'] == 1
    assert rows[0]['inference_executed'] is False
    assert rows[0]['license_or_login_read'] is False
    assert discover_official_installation(installs=[install], model_roots=[models]) == rows[0]


@pytest.mark.parametrize('kind', ['valid', 'grouped', 'wrong_path', 'unsigned', 'other_signer', 'missing'])
def test_signature_receipts_bind_each_selected_executable(kind, tmp_path):
    paths = [tmp_path / name for name in ('app.exe', 'ffmpeg.exe', 'ffprobe.exe')]
    rows = [{'path': str(p), 'status': 'Valid', 'signer': 'CN=Topaz Labs LLC, O=Topaz Labs LLC'} for p in paths]
    if kind == 'valid':
        validate_signatures(rows, paths)
        return
    if kind == 'grouped':
        rows = {'path': [str(p) for p in paths], 'status': 'Valid Valid Valid', 'signer': ['Topaz'] * 3}
    elif kind == 'wrong_path':
        rows[1]['path'] = str(tmp_path / 'replacement.exe')
    elif kind == 'unsigned':
        rows[1]['status'] = 'NotSigned'
    elif kind == 'other_signer':
        rows[1]['signer'] = 'O=Other'
    else:
        rows.pop()
    with pytest.raises(RuntimeError, match='signatures'):
        validate_signatures(rows, paths)


@pytest.mark.parametrize('version', [3, '3'])
def test_model_discovery_never_claims_inference_or_other_scale(runtime, version):  # noqa: F811
    path = runtime.definitions / 'iris-3.json'
    definition = json.loads(path.read_text())
    definition.update(shortName='iris', version=version)
    path.write_text(json.dumps(definition))
    (runtime.data / 'iris-v3-fgnet-fp16-576x672-1x-ox.tz3').write_bytes(b'fake-only')
    before = deepcopy(definition)
    result = model_evidence(runtime, 'iris-3', 1)
    assert result['status'] == 'candidate_files_present_actual_engine_load_still_required'
    assert len(result['candidate_weights']) == 1
    assert json.loads(path.read_text()) == before
    with pytest.raises(RuntimeError, match='no installed 2x'):
        model_evidence(runtime, 'iris-3', 2)


@pytest.mark.parametrize(('model_id', 'scale', 'template', 'filename'), [
    ('rhea-1', 1, 'fgnet-fp16-[H]x[W]-4x-ox.tz', 'rhea-v1-fgnet-fp16-576x384-4x-ox.tz3'),
    ('rhea-1', 2, 'fgnet-fp16-[H]x[W]-4x-ox.tz', 'rhea-v1-fgnet-fp16-576x384-4x-ox.tz3'),
    ('nxl-1', 1, 'fp16-8x[H]x[W]-ox.tz', 'nxl-v1-fp16-8x576x416-ox.tz3'),
])
def test_model_evidence_follows_official_net_template(runtime, model_id, scale, template, filename):  # noqa: F811
    definition = {'shortName': model_id.split('-')[0], 'version': 1,
        'backends': {'tensorrt': {'scales': {str(scale): {'nets': [template]}}}}}
    (runtime.definitions / f'{model_id}.json').write_text(json.dumps(definition))
    (runtime.data / filename).write_bytes(b'official-template-fixture')
    result = model_evidence(runtime, model_id, scale)
    assert [Path(item['path']).name for item in result['candidate_weights']] == [filename]


def test_model_evidence_binds_shared_and_scale_specific_nets(runtime):  # noqa: F811
    definition = {'shortName': 'thf', 'version': 4, 'backends': {'tensorrt': {'scales': {'2': {
        'nets': ['fnet-fp16-[H]x[W]-ox.tz', 'gnet-fp16-[H]x[W]-2x-ox.tz']}}}}}
    (runtime.definitions / 'thf-4.json').write_text(json.dumps(definition))
    for name in ('thf-v4-fnet-fp16-576x384-ox.tz3', 'thf-v4-gnet-fp16-576x384-2x-ox.tz3'):
        (runtime.data / name).write_bytes(b'official-template-fixture')
    result = model_evidence(runtime, 'thf-4', 2)
    assert {Path(item['path']).name for item in result['candidate_weights']} == {
        'thf-v4-fnet-fp16-576x384-ox.tz3', 'thf-v4-gnet-fp16-576x384-2x-ox.tz3'}


def test_model_evidence_accepts_official_tensorrt_capability_and_runtime_slots(runtime):  # noqa: F811
    definition = {'shortName': 'iris', 'version': 3, 'backends': {'tensorrt': {'scales': {'2': {
        'nets': ['fgnet-fp32-[H]x[W]-[S]x-rt[C]-[R]-rt.tz']}}}}}
    (runtime.definitions / 'iris-3.json').write_text(json.dumps(definition))
    filename = 'iris-v3-fgnet-fp32-384x480-2x-rt809-10800-rt.tz'
    (runtime.data / filename).write_bytes(b'official-tensorrt-template-fixture')
    result = model_evidence(runtime, 'iris-3', 2)
    assert [Path(item['path']).name for item in result['candidate_weights']] == [filename]


def resource_sample(*, gpu_free=10 * 1024**3, ram_available=32 * 1024**3):
    return {'monotonic': 1.0, 'gpu_uuid': 'GPU-fixture',
        'gpu_total_bytes': 16 * 1024**3, 'gpu_used_bytes': 16 * 1024**3 - gpu_free,
        'gpu_free_bytes': gpu_free, 'ram_total_bytes': 64 * 1024**3,
        'ram_available_bytes': ram_available}


def test_topaz_startup_has_no_fixed_free_vram_gate(tmp_path):
    row = resource_sample(gpu_free=10 * 1024**3)
    reader = type('Reader', (), {'sample': lambda self: row})()
    with (tmp_path / 'resources.jsonl').open('x', encoding='utf8') as log:
        assert record_topaz_startup_sample(reader, ResourceGuard(), log) == row
    recorded = json.loads((tmp_path / 'resources.jsonl').read_text())
    assert recorded['phase'] == 'startup' and recorded['gpu_free_bytes'] == 10 * 1024**3


def test_topaz_startup_ram_error_is_specific_and_recorded(tmp_path):
    row = resource_sample(ram_available=TOPAZ_STARTUP_FREE_RAM_BYTES - 1)
    reader = type('Reader', (), {'sample': lambda self: row})()
    with (tmp_path / 'resources.jsonl').open('x', encoding='utf8') as log:
        with pytest.raises(RuntimeError, match='system RAM'):
            record_topaz_startup_sample(reader, ResourceGuard(), log)
    assert json.loads((tmp_path / 'resources.jsonl').read_text())['phase'] == 'startup'


def test_task_progress_reports_engine_frames_then_audit(tmp_path):
    events = []
    progress = TaskProgress(tmp_path, events.append, 'enhancement')
    progress.emit(0)
    (tmp_path / 'source_video_probe.stdout').write_text(json.dumps({'frames': [{}] * 100}))
    (tmp_path / 'enhancement.stdout').write_text('frame=50\nprogress=continue\n')
    progress.refresh()
    assert events[-1] == 47
    (tmp_path / 'output_video_probe.stdout').write_text('{}')
    progress.refresh()
    assert events[-1] == 92
    (tmp_path / 'strict_decode.stdout').write_text('')
    progress.refresh()
    assert events[-1] == 98
    (tmp_path / 'result.json').write_text('{}')
    progress.refresh()
    assert events[-1] == 100
