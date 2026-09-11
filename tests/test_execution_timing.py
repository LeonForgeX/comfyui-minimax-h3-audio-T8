import pytest

from h3_audio_t8_pkg.execution_timing import WallTimings


def test_values_arguments_and_failure_are_preserved():
    values = iter([10., 12., 20., 23.])
    timer = WallTimings(lambda: next(values))
    sentinel = object()
    assert timer.call('prepare', lambda *, value: value, value=sentinel) is sentinel
    error = RuntimeError('original')
    def fail():
        raise error
    with pytest.raises(RuntimeError) as caught:
        timer.call('sample', fail)
    assert caught.value is error
    report = timer.report()
    assert [(e['status'], e['seconds']) for e in report['events']] == [('completed', 2.), ('failed', 3.)]
    report['events'][0]['seconds'] = 999
    assert timer.report()['events'][0]['seconds'] == 2.


def test_new_invocation_does_not_import_historical_cache_time():
    original = WallTimings()
    original.call('sample', lambda: None)
    resumed = WallTimings()
    assert resumed.report()['events'] == []
    assert resumed.report()['historical_cache_time_counted_as_current'] is False
    assert resumed.report()['gpu_synchronization_added'] is False
