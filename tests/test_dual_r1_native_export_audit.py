from copy import deepcopy

import pytest

from tools.audit_dual_r1_native_exports import audit_edits


def case():
    original = {'nodes': [{'id': 1, 'type': 'Text', 'widgets_values': ['old text'],
                          'inputs': [], 'outputs': []}], 'links': []}
    saved = deepcopy(original)
    saved['nodes'][0]['widgets_values'] = ['']
    api = {'1': {'class_type': 'Text', 'inputs': {'value': ''}}}
    info = {'Text': {'input': {'required': {'value': ['STRING', {'default': ''}]}}}}
    edits = [{'node': 1, 'type': 'Text', 'widget': 0, 'before': 'old text', 'after': ''}]
    return original, saved, api, info, edits


def test_clear_value_contract():
    assert audit_edits(*case())['all_other_execution_widgets_and_links_equal']


@pytest.mark.parametrize('drift', ['saved_stale', 'api_stale', 'api_missing', 'node_mode',
                                 'node_type', 'original_changed', 'missing_node'])
def test_reject_stale_or_unrelated_changes(drift):
    original, saved, api, info, edits = case()
    if drift == 'saved_stale':
        saved['nodes'][0]['widgets_values'] = ['old text']
    elif drift == 'api_stale':
        api['1']['inputs']['value'] = 'old text'
    elif drift == 'api_missing':
        # Empty is a real literal, not permission to drop a required widget.
        api['1']['inputs'] = {}
        info['Text']['input']['required']['value'][1]['default'] = 'old text'
    elif drift == 'node_mode':
        saved['nodes'][0]['mode'] = 4
    elif drift == 'node_type':
        saved['nodes'][0]['type'] = 'Other'
    elif drift == 'original_changed':
        original['nodes'][0]['widgets_values'] = ['changed']
    else:
        saved['nodes'] = []
    with pytest.raises(ValueError):
        audit_edits(original, saved, api, info, edits)
