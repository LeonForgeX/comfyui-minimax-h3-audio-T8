"""R1: caller declarations cannot authorize actual cache artifact reuse."""
from copy import deepcopy
import hashlib
import json

from h3_audio_t8_pkg.creator_segment_cache_advanced import compile_creator_segment_cache_plan
from h3_audio_t8_pkg.creator_workspace_advanced import _hash
from test_creator_segment_cache_advanced import _workspace


def compile_plan(workspace, model, loras=None, index=''):
    return compile_creator_segment_cache_plan(workspace, json.dumps(model),
        json.dumps({'loras': loras or []}), '{"steps":8}', '{}', index)


def cache_index(plan):
    return json.dumps({'entries': [{'cache_key': row['cache_key'],
        'semantic_hash': row['semantic_hash'], 'accepted': True,
        'artifact_manifest': {'source': 'caller-supplied-not-read'}}
        for row in plan['desired_entries']]})


def check_unverified(plan, report):
    assert plan['identity_assurance'] == 'caller_declarations_only'
    assert report['identity_assurance'] == 'caller_declarations_only'
    assert plan['verified_reuse_count'] == report['verified_reuse_count'] == 0
    assert not plan['execution_reuse_authorized']
    assert not plan['files_opened'] and not plan['files_mutated']


def test_same_filename_new_bytes_never_becomes_verified_hit(tmp_path):
    model = tmp_path / 'model.safetensors'
    model.write_bytes(b'first bytes')
    declaration = {'path': str(model)}
    before = compile_plan(_workspace(), declaration)[0]
    model.write_bytes(b'other bytes')
    after, _, count, invalid, _, report = compile_plan(_workspace(), declaration, index=cache_index(before))
    assert count == 3 and invalid == 0  # Declaration equality only; no invented detection.
    assert {row['status'] for row in after['desired_entries']} == {'declaration_match_unverified'}
    check_unverified(after, json.loads(report))


def test_supplied_changed_hash_invalidates_but_is_not_independently_verified(tmp_path):
    model = tmp_path / 'model.safetensors'
    model.write_bytes(b'first')
    declaration = {'path': str(model), 'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
    before = compile_plan(_workspace(), declaration)[0]
    model.write_bytes(b'second')
    declaration['sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
    after, _, count, invalid, _, report = compile_plan(_workspace(), declaration, index=cache_index(before))
    assert count == 0 and invalid == 3
    assert after['protected_accepted_count'] == 3 and not after['proposed_quarantine']
    check_unverified(after, json.loads(report))


def test_lora_strength_and_order_invalidate_all_affected_variants():
    loras = [{'path': 'a', 'strength': 1.0}, {'path': 'b', 'strength': .5}]
    initial = compile_plan(_workspace(), {'model': 'h3'}, loras)[0]
    for changed in (list(reversed(loras)), [dict(loras[0], strength=.8), loras[1]]):
        after, _, count, invalid, _, report = compile_plan(_workspace(), {'model': 'h3'}, changed, cache_index(initial))
        assert count == 0 and invalid == 3
        check_unverified(after, json.loads(report))


def test_reference_reorder_only_invalidates_affected_shot():
    workspace = _workspace()
    workspace['shots'][1]['media_roles'] = {'references': ['first.png', 'second.png']}
    workspace['workspace_hash'] = _hash({k: v for k, v in workspace.items() if k != 'workspace_hash'})
    before = compile_plan(workspace, {'model': 'h3'})[0]
    workspace['shots'][1]['media_roles']['references'].reverse()
    workspace['workspace_hash'] = _hash({k: v for k, v in workspace.items() if k != 'workspace_hash'})
    after, _, count, invalid, _, report = compile_plan(workspace, {'model': 'h3'}, index=cache_index(before))
    assert count == 2 and invalid == 1
    check_unverified(after, json.loads(report))


def test_queued_plan_is_snapshot_not_authority_for_changed_source(tmp_path):
    source = tmp_path / 'reference.png'
    source.write_bytes(b'old')
    workspace = _workspace()
    workspace['shots'][0]['media_roles'] = {'reference': str(source)}
    workspace['workspace_hash'] = _hash({k: v for k, v in workspace.items() if k != 'workspace_hash'})
    plan, *_, report = compile_plan(workspace, {'model': 'h3'})
    snapshot = deepcopy(plan)
    source.write_bytes(b'new')
    workspace['shots'][0]['media_roles']['reference'] = 'changed.png'
    assert plan == snapshot  # No reference alias to mutable queued inputs.
    check_unverified(plan, json.loads(report))
