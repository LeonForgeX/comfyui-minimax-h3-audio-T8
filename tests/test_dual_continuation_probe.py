import json

import pytest

from tools.run_dual_model_continuation_probe import assert_interrupted, stage_snapshot, request_interrupt


def test_interrupt_accepts_empty_http200_and_checks_owned_server(monkeypatch):
    from types import SimpleNamespace
    import requests
    calls = []
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            assert self.trust_env is False
            calls.append(('post', url, kwargs))
            # No json method: the real endpoint has no JSON body.
            return SimpleNamespace(raise_for_status=lambda: calls.append('status'))
    monkeypatch.setattr(requests, 'Session', Session)
    request_interrupt(SimpleNamespace(url='http://127.0.0.1:8208',
        assert_port_owner=lambda: calls.append('owner')))
    assert calls[0] == 'owner' and calls[-1] == 'status'
    assert calls[1][1].endswith('/interrupt')


@pytest.mark.parametrize('sent,event,passed', [(True, 'execution_interrupted', True),
    (False, 'execution_interrupted', False), (True, 'execution_error', False),
    (True, 'execution_success', False)])
def test_expected_interrupt_does_not_mask_generation_errors(tmp_path, sent, event, passed):
    (tmp_path / 'events.jsonl').write_text(json.dumps({'type': event}) + '\n', encoding='utf-8')
    if passed:
        assert_interrupted(tmp_path, sent)
    else:
        with pytest.raises(RuntimeError, match='specifically requested Core interruption'):
            assert_interrupted(tmp_path, sent)


def test_resume_snapshot_tracks_exact_first_segment_bytes_only(tmp_path):
    for index in (0, 1):
        folder = tmp_path / f'chain/dual_stages/segment_{index:05d}/candidate'
        folder.mkdir(parents=True)
        for stage in ('low_x0', 'high_input', 'high_output'):
            (folder / (stage + '-a.json')).write_text('{}', encoding='utf-8')
            (folder / (stage + '-a.safetensors')).write_bytes(b'fixture')
    before = stage_snapshot(tmp_path)
    assert len(before) == 4
    assert all('segment_00000' in path and 'high_output' not in path for path in before)
    (tmp_path / next(iter(before))).write_bytes(b'changed')
    assert before != stage_snapshot(tmp_path)
