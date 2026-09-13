from copy import deepcopy
import json

import pytest

from h3_audio_t8_pkg.topaz_runtime import (
    TOPAZ_STARTUP_FREE_RAM_BYTES,
    model_evidence,
    record_topaz_startup_sample,
    validate_signatures,
)
from h3_audio_t8_pkg.dlss_fi_backend.resources import ResourceGuard
from tests.test_topaz_contract import runtime  # noqa: F401


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
