from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.package_progressive_candidate import (
    EXPECTED_RELEASE_VERSION,
    EXPECTED_WORKFLOW_COUNT,
    PENDING_REVIEW_TOKENS,
    SELF_LIFT_WORKFLOWS,
    T8_MEMORY_REVIEW_BINDING,
    T8_MEMORY_WORKFLOW,
    validate_t8_memory_promotion,
    validate_t8_memory_workflow,
    recipe_identity,
)
from tools.verify_progressive_package import validate_registry, validate_workflows


ROOT = Path(__file__).resolve().parents[1]


def accepted_memory_workflow():
    workflow = json.loads((ROOT / T8_MEMORY_WORKFLOW).read_text(encoding='utf8'))
    workflow['extra']['acceptance_scope'] = 'Exact bound two-segment 8s sample accepted.'
    workflow['extra']['t8_bound_review'] = {
        'status': 'accepted_in_this_review_scope',
        'date': '2026-09-16',
        'scope': 'Exact h4+c2 two-segment 8s candidate accepted; not a universal claim.',
        'not_universal_quality_claim': True,
        **T8_MEMORY_REVIEW_BINDING,
    }
    return workflow


def test_release_gate_matches_current_declared_schema_and_workflow_counts():
    features = json.loads((ROOT / 'features.json').read_text(encoding='utf8'))
    assert EXPECTED_RELEASE_VERSION == '1.82.0'
    assert EXPECTED_WORKFLOW_COUNT == 255
    # The qualified v1.82.0 delivery remains exactly336 nodes. An unreviewed
    # V2 development suffix must not borrow this release gate's qualification.
    ids = features['nodes']
    validate_registry(ids[:336], ids[:336])
    if len(ids) > 336:
        assert ids[336:339] == ['MiniMaxH3FastH3V2SetupEXPT8',
            'MiniMaxH3FastH3V2RuntimeAuditEXPT8', 'MiniMaxH3FastH3V2DualModelLongVideoEXPT8']
        if len(ids) > 339:
            assert ids[339:] == ['SolAttnMiniMax']
        with pytest.raises(ValueError):
            validate_registry(ids, ids)
    names = list(SELF_LIFT_WORKFLOWS)
    names.extend(f'examples/workflows/synthetic/release-gate-{index:03d}.json'
                 for index in range(EXPECTED_WORKFLOW_COUNT - len(names)))
    validate_workflows(names, set(names))


def test_exact_accepted_h4c2_memory_workflow_passes_release_gate_in_memory():
    review = validate_t8_memory_workflow(accepted_memory_workflow())
    assert all(review[key] == value for key, value in T8_MEMORY_REVIEW_BINDING.items())


@pytest.mark.parametrize('mutation', ['review_id', 'media_sha256', 'head_chunks', 'ffn_chunks',
                                      'duration', 'context_mode', 'low_context_source', 'color_mode',
                                      'color_mode_motion'])
def test_memory_release_gate_rejects_unreviewed_or_different_recipe(mutation):
    workflow = accepted_memory_workflow()
    if mutation in T8_MEMORY_REVIEW_BINDING:
        workflow['extra']['t8_bound_review'][mutation] = 'different'
    elif mutation == 'head_chunks':
        next(node for node in workflow['nodes']
             if node['type'] == 'MiniMaxH3LowVRAMAttentionT8Advanced')['widgets_values'] = [1]
    elif mutation == 'ffn_chunks':
        next(node for node in workflow['nodes']
             if node['type'] == 'MiniMaxH3ChunkFeedForwardT8Advanced')['widgets_values'] = [1, 4096]
    else:
        node = next(node for node in workflow['nodes']
                    if node['type'] == 'MiniMaxH3DualModelLongVideoEXPT8')
        index, value = {
            'duration': (12, 24),
            'context_mode': (49, 'reference_only'),
            'low_context_source': (50, 'independent_low_x0'),
            'color_mode': (51, 'bounded_spatial_temporal_exp'),
            'color_mode_motion': (51, 'bounded_spatial_v2'),
        }[mutation]
        node['widgets_values'][index] = value
    with pytest.raises(ValueError):
        validate_t8_memory_workflow(workflow)


def test_current_accepted_memory_workflow_passes_candidate_gate():
    path = ROOT / T8_MEMORY_WORKFLOW
    text = path.read_text(encoding='utf8')
    assert not any(token.lower() in text.lower() for token in PENDING_REVIEW_TOKENS)
    result = validate_t8_memory_promotion({T8_MEMORY_WORKFLOW: __import__('hashlib').sha256(path.read_bytes()).hexdigest()})
    assert result['review']['media_sha256'] == T8_MEMORY_REVIEW_BINDING['media_sha256']


def test_pending_marker_still_blocks_promotion(tmp_path):
    path = tmp_path / T8_MEMORY_WORKFLOW
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(accepted_memory_workflow()) + ' human review pending', encoding='utf8')
    with pytest.raises(ValueError, match='pending human-review'):
        validate_t8_memory_promotion({T8_MEMORY_WORKFLOW: __import__('hashlib').sha256(path.read_bytes()).hexdigest()}, project=tmp_path)


@pytest.mark.parametrize('kind', ['seed', 'audio', 'lora', 'prompt', 'first_frame', 'link'])
def test_full_execution_recipe_cannot_borrow_accepted_review(kind):
    workflow = accepted_memory_workflow()
    node_type = {'seed': 'MiniMaxH3DualModelLongVideoEXPT8', 'audio': 'MiniMaxH3DualModelLongVideoEXPT8',
                 'lora': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced', 'prompt': 'MiniMaxH3PromptRelayPlanT8Advanced',
                 'first_frame': 'LoadImage'}
    if kind == 'link':
        workflow['links'][0][2] += 1
    else:
        node = next(n for n in workflow['nodes'] if n['type'] == node_type[kind])
        index = {'seed': 28, 'audio': 9, 'lora': 0, 'prompt': 0, 'first_frame': 0}[kind]
        node['widgets_values'][index] = 'different'
    with pytest.raises(ValueError):
        validate_t8_memory_workflow(workflow)


def test_registry_gate_rejects_reorder_and_missing_new_node():
    ids = json.loads((ROOT / 'features.json').read_text(encoding='utf8'))['nodes'][:336]
    reordered = deepcopy(ids)
    reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
    with pytest.raises(ValueError):
        validate_registry(reordered, reordered)
    with pytest.raises(ValueError):
        validate_registry(ids[:-1], ids[:-1])


def test_recipe_identity_preserves_large_integers_and_boolean_types():
    assert recipe_identity([8, 8.0, 2**63 + 1, True, .15]) == [8, 8, 2**63 + 1, True, .15]
    assert type(recipe_identity(True)) is bool
