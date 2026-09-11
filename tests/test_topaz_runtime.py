from copy import deepcopy
import json

import pytest

from h3_audio_t8_pkg.topaz_runtime import model_evidence, validate_signatures
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
