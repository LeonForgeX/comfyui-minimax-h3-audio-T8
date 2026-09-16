"""Cold/hot source/measurement semantics, no model or CUDA allocation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_fast_h3_v2_thermal_pair import PROFILES, paired_graphs, read_resource_rows, resource_interval, stage_times


def test_pair_uses_identical_canvas_condition_and_reuses_native_recipes():
    old, new = paired_graphs().values()
    assert old['9'] == new['9']
    assert (old['9']['inputs']['width'], old['9']['inputs']['height'], old['9']['inputs']['length']) == (832, 480, 73)
    assert old['13']['inputs']['seed'] == new['13']['inputs']['seed'] == 2609032101
    assert old['10']['inputs']['steps'] == 8
    assert old['10']['inputs']['sampler_name'] == 'euler'
    assert old['10']['inputs']['shift_video'] == 12.0
    assert new['10']['inputs']['profile'] == 'trained_vsa_exp'
    assert new['10']['inputs']['min_tokens'] == 0
    assert '2' in old and '2' not in new and '3' not in new
    assert old['3']['class_type'] == 'PathchSageAttentionKJ'
    assert tuple(paired_graphs()) == PROFILES
    assert all(graph['13']['class_type'] == 'T8FastH3V2ThermalSamplerProbe' and
               graph['13']['inputs']['thermal_profile'] == profile
               for profile, graph in paired_graphs().items())


@pytest.mark.parametrize('change', ['cache', 'incomplete', 'missing', 'forwards'])
def test_any_cached_or_partial_timing_cannot_be_called_a_hot_generation(change):
    timing = dict(complete_uncached_graph=True, graph_cached_nodes=[], elapsed_seconds=8,
        node_intervals=[dict(node=n, seconds=1) for n in ('6', '9', '13', '14', '16')], timing_scope='dispatch overhead')
    network = dict(completed_network_forwards=8, sampler_seconds=1)
    if change == 'cache':
        timing['graph_cached_nodes'] = ['13']
    elif change == 'incomplete':
        timing['complete_uncached_graph'] = False
    elif change == 'missing':
        timing['node_intervals'].pop()
    else:
        network['completed_network_forwards'] = 4
    with pytest.raises(ValueError):
        stage_times(timing, network)


def test_resources_do_not_attribute_other_intervals_or_claim_exact_peaks():
    rows = [dict(monotonic=t, gpu_used_bytes=v, owned_process_rss_bytes=v,
                 ram_available_bytes=100-v) for t, v in [(1, 90), (2, 10), (3, 20), (4, 95)]]
    report = resource_interval(rows, 2, 3)
    assert report['maximum_observed_whole_card_bytes'] == 20
    assert report['maximum_observed_process_rss_bytes'] == 20
    assert 'not exact peaks' in report['scope']
    with pytest.raises(ValueError):
        resource_interval(rows, 2.5, 3)


def test_appending_resource_file_does_not_fake_or_drop_completed_bad_records(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"monotonic":1}\n{"unfinished":', encoding='utf8')
    assert read_resource_rows(path) == [dict(monotonic=1)]
    path.write_text('not-valid-json\n', encoding='utf8')
    with pytest.raises(ValueError):
        read_resource_rows(path)
