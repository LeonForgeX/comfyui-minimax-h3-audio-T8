import json

from h3_audio_t8_pkg import topaz_runtime
from tests.test_topaz_contract import runtime  # noqa: F401


def prepare(installation):
    path = installation.definitions / 'iris-3.json'
    definition = json.loads(path.read_text())
    definition.update(shortName='iris', version='3', modelType=1)
    path.write_text(json.dumps(definition))
    (installation.data / 'iris-v3-fgnet-fp16-576x384-2x-ox.tz3').write_bytes(b'candidate-not-proof')


def test_catalog_distinguishes_scale_files_from_inference(runtime):  # noqa: F811
    prepare(runtime)
    catalog = topaz_runtime.model_catalog(runtime, ['noise', 'preblur', 'blend'])
    iris = next(row for row in catalog['models'] if row['id'] == 'iris-3')
    assert iris['scales']['1']['status'] == 'missing_candidate_weights'
    assert iris['scales']['2']['status'] == 'candidate_files_present_unverified'
    assert iris['scales']['4']['candidate_files'] == []
    assert iris['execution_verified'] is False
    assert catalog['inference_executed'] is False


def test_catalog_lists_only_runtime_and_model_accepted_parameters(runtime):  # noqa: F811
    prepare(runtime)
    iris = next(row for row in topaz_runtime.model_catalog(runtime, ['noise', 'blend'])['models']
                if row['id'] == 'iris-3')
    assert set(iris['parameters']) == {'noise', 'blend'}
    assert iris['parameters']['noise']['min'] == 0
    assert iris['parameters']['noise']['max'] == 1
    assert iris['parameters']['blend']['source'] == 'runtime_option'


def test_catalog_neuroserver_is_separate_not_regular_ready(runtime):  # noqa: F811
    row = next(row for row in topaz_runtime.model_catalog(runtime, [])['models']
               if row['id'] == 'slp-2.5')
    assert row['route'] == 'neuroserver'
    assert row['status'] == 'separate_neuroserver_qualification_required'
    assert row['execution_verified'] is False
    assert 'scales' not in row


def test_catalog_keeps_bad_definition_diagnostic_without_losing_valid_models(runtime):  # noqa: F811
    prepare(runtime)
    (runtime.definitions / 'broken.json').write_text('{')
    (runtime.definitions / 'metadata.json').write_text('{}')
    catalog = topaz_runtime.model_catalog(runtime, [])
    assert {row['id'] for row in catalog['models']} == {'iris-3', 'slp-2.5'}
    assert {row['id'] for row in catalog['unreadable_or_nonmodel_definitions']} == {'broken', 'metadata'}


def test_catalog_is_read_only_and_does_not_hash_weights(runtime, monkeypatch):  # noqa: F811
    prepare(runtime)
    before = {p: p.read_bytes() for root in (runtime.definitions, runtime.data) for p in root.iterdir()}
    monkeypatch.setattr(topaz_runtime, 'file_identity', lambda *_: (_ for _ in ()).throw(AssertionError('hashed large weight')))
    monkeypatch.setattr(topaz_runtime, '_run_readonly', lambda *_: (_ for _ in ()).throw(AssertionError('ran engine')))
    topaz_runtime.model_catalog(runtime, [])
    assert all(path.read_bytes() == data for path, data in before.items())


def test_catalog_does_not_offer_interpolation_or_auxiliary_models_as_upscale(runtime):  # noqa: F811
    prepare(runtime)
    for name, fields in [('chr-2', {'modelType': 2, 'changesFPS': 1}),
                         ('cpe-2', {'modelType': 4}), ('unknown-1', {})]:
        (runtime.definitions / (name + '.json')).write_text(json.dumps({'backends': {}, **fields}))
    rows = {row['id']: row for row in topaz_runtime.model_catalog(runtime, [])['models']}
    assert rows['chr-2']['route'] == 'tvai_fi'
    assert rows['chr-2']['status'] == 'frame_interpolation_definition_discovered'
    assert 'scales' not in rows['chr-2']
    for name in ('cpe-2', 'unknown-1'):
        assert rows[name]['route'] == 'not_offered_by_regular_upscale'
        assert 'scales' not in rows[name]
