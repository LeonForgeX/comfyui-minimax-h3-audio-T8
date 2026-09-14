"""Reproduce the real browser export shift after a metadata-enabled base_seed."""
from tools.api_to_frontend_workflow import convert


def test_metadata_seed_control_precedes_seed_policy_and_later_widgets():
    info = {'Example': {'input': {'required': {
        'base_seed': ['INT', {'default': 10, 'control_after_generate': True}],
        'seed_policy': [['increment', 'fixed'], {'default': 'increment'}],
        'steps': ['INT', {'default': 4}],
        'enabled': ['BOOLEAN', {'default': True}],
    }}, 'output': [], 'output_name': []}}
    graph = {'1': {'class_type': 'Example', 'inputs': {
        'base_seed': 10, 'seed_policy': 'increment', 'steps': 4, 'enabled': True}}}
    assert convert(graph, info, 'seed regression')['nodes'][0]['widgets_values'] == [
        10, 'fixed', 'increment', 4, True]


def test_explicit_false_does_not_add_control_to_legacy_seed_name():
    info = {'Example': {'input': {'required': {
        'seed': ['INT', {'default': 10, 'control_after_generate': False}],
        'steps': ['INT', {'default': 4}],
    }}, 'output': [], 'output_name': []}}
    graph = {'1': {'class_type': 'Example', 'inputs': {'seed': 10, 'steps': 4}}}
    assert convert(graph, info, 'seed regression')['nodes'][0]['widgets_values'] == [10, 4]
