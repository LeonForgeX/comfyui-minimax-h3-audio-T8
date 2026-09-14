"""Private probe counters must not claim calls outside the actual forward."""
import sys
from types import SimpleNamespace

import pytest

from tools.progressive_observed_extension.observer import ForwardCallAudit


def kernel(value):
    if value == 'error':
        raise ValueError('kernel failure')
    return value


def forward(value):
    return kernel(value)


def test_forward_gate_and_snapshot():
    observer = ForwardCallAudit({'kernel': kernel}, forward)
    with observer:
        kernel(1)
        assert forward(2) == 2
        kernel(3)
        assert forward(4) == 4
    report = observer.report()
    assert observer.counts['kernel'] == dict(calls=2, successful_returns=2, empty_returns=0)
    assert [row['index'] for row in report['forwards']] == [0, 1]
    assert all(row['completed'] for row in report['forwards'])
    assert not report['profiler_still_owned'] and sys.getprofile() is None
    report['forwards'].clear()
    assert len(observer.forwards) == 2


def test_failure_is_not_a_successful_forward_and_restores_profiler():
    observer = ForwardCallAudit({'kernel': kernel}, forward)
    with pytest.raises(ValueError, match='kernel failure'), observer:
        forward('error')
    assert not observer.forwards[0]['completed']
    assert observer.counts['kernel'] == dict(calls=1, successful_returns=0, empty_returns=1)
    assert not observer.active and sys.getprofile() is None


def test_existing_profiler_is_not_replaced():
    def existing(frame, event, arg):
        pass
    observer = ForwardCallAudit({'kernel': kernel}, forward)
    sys.setprofile(existing)
    try:
        with pytest.raises(RuntimeError, match='existing profile'), observer:
            pass
        assert sys.getprofile() is existing
        assert not observer.active
    finally:
        sys.setprofile(None)


def minimax_mlp_chunked_forward(self, x):
    return x


def ffn_forward(value):
    module = SimpleNamespace(kj_seq_threshold=4, kj_num_chunks=2)
    return minimax_mlp_chunked_forward(module, value)


@pytest.mark.parametrize('length,expected', [(4, 0), (5, 1)])
def test_ffn_file_and_threshold(length, expected):
    observer = ForwardCallAudit({'kernel': kernel}, ffn_forward, __file__)
    with observer:
        ffn_forward(SimpleNamespace(shape=(length, 2)))
    row = observer.forwards[0]
    assert row['ffn_calls'] == 1 and row['ffn_chunk_eligible_calls'] == expected
    assert observer.counts['kernel']['calls'] == 0


def test_foreign_ffn_file_not_counted(tmp_path):
    observer = ForwardCallAudit({'kernel': kernel}, ffn_forward, tmp_path / 'foreign.py')
    with observer:
        ffn_forward(SimpleNamespace(shape=(5, 2)))
    assert observer.forwards[0]['ffn_calls'] == 0
