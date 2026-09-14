"""Probe configuration tests; actual live Core API audit remains separate."""

import pytest

from tools.run_progressive_native_chain import build_graph, asset_paths, startup_resources
from tools.progressive_native_probe_extension import NativeChainProbe
from h3_audio_t8_pkg.prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE, build_prompt_relay_plan


@pytest.mark.parametrize('backend', ['pytorch', 'kj-memory', 'sol'])
def test_graph_separates_models_preserves_timeline_and_modes(backend):
    graph = build_graph('probe', backend, True, 'apply_exp', 'report_only')
    assert graph['2']['inputs']['model'] == graph['3']['inputs']['model'] == ['1', 0]
    assert graph['21']['inputs']['model'] == ['2', 0]
    assert graph['22']['inputs']['model'] == ['3', 0]
    assert graph['8']['inputs']['eav_mode'] == 'apply_exp'
    assert graph['8']['inputs']['tst_mode'] == 'report_only'
    assert graph['8']['inputs']['prompt_relay_plan'] == ['11', 0]
    assert graph['11']['inputs']['query_route'] == 'joint_av_exp'
    plan = build_prompt_relay_plan(**graph['7']['inputs'])[0]
    assert plan['type'] == PROMPT_RELAY_PLAN_TYPE
    assert len(plan['events']) == 4


def test_plain_graph_has_no_unused_relay_nodes():
    graph = build_graph('probe', relay=False)
    assert not {'7', '11'}.intersection(graph)
    assert 'prompt_relay_plan' not in graph['8']['inputs']


def test_schema_uses_actual_relay_socket_and_real_model_inputs():
    schema = NativeChainProbe.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert inputs['prompt_relay_plan'].io_type == PROMPT_RELAY_PLAN_TYPE
    assert {'model', 'model_hires', 'clip', 'video_vae', 'audio_vae'} <= inputs.keys()


def test_missing_real_assets_fail_before_server_creation(tmp_path):
    with pytest.raises(FileNotFoundError):
        asset_paths(tmp_path)


@pytest.mark.parametrize('kwargs', [{'backend': 'unknown'}, {'eav': 'on'}, {'tst': 'on'}])
def test_unknown_modes_rejected(kwargs):
    with pytest.raises(ValueError):
        build_graph('probe', **kwargs)


def test_new_resource_observation_cannot_reuse_the_pre_hash_pass():
    samples = iter([{'free': 14000}, {'free': 9000}])
    class Reader:
        def sample(self):
            return next(samples)
    class Guard:
        def observe(self, sample, *, startup):
            assert startup is True
            return 'insufficient' if sample['free'] < 12000 else None
    reader, guard = Reader(), Guard()
    assert startup_resources(reader, guard) == {'free': 14000}
    with pytest.raises(RuntimeError, match='insufficient'):
        startup_resources(reader, guard)


@pytest.mark.parametrize('change,reason', [({}, None), ({'gpu_free_bytes': 9000*1024**2}, 'margin'),
    ({'gpu_uuid': 'another'}, 'device'), ({'monotonic': 1.}, 'clock')])
def test_real_guard_separates_cpu_hash_phase_but_rechecks_margin_and_device(change, reason):
    from tools.progressive_probe_control import ResourceGuard
    from test_progressive_probe_control import sample
    before = sample(time=1.)
    assert ResourceGuard().observe(before, startup=True) is None
    after = {**sample(time=121.), **change}
    class Reader:
        def sample(self):
            return after
    guard = ResourceGuard()
    if reason:
        with pytest.raises(RuntimeError, match=reason):
            startup_resources(Reader(), guard, prior_preflight=before)
    else:
        assert startup_resources(Reader(), guard, prior_preflight=before) == after
        # Runtime monitoring remains continuous and still rejects real gaps.
        assert guard.observe(sample(time=126.)) == 'resource_telemetry_invalid_or_stale'
