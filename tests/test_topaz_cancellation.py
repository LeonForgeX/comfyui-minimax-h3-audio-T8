from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg import topaz_runtime
from h3_audio_t8_pkg.dlss_fi_backend import process, resources
from tests.test_topaz_contract import runtime  # noqa: F401


@pytest.mark.parametrize('remaining', [0, 1])
def test_original_cancel_returns_only_after_confirmed_owned_cleanup(monkeypatch, runtime, tmp_path, remaining):  # noqa: F811
    class ComfyCancellation(Exception):
        pass

    cancellation = ComfyCancellation('fixture cancellation')
    monkeypatch.setattr(topaz_runtime, 'audit_installation', lambda _: {'executables': [], 'tvai_up_options': []})
    monkeypatch.setattr(topaz_runtime, 'model_evidence', lambda *a: {'candidate_weights': []})
    monkeypatch.setattr(resources, 'SerialProbeLease', lambda _: nullcontext())
    monkeypatch.setattr(resources, 'NvmlResourceReader', lambda: nullcontext(SimpleNamespace(sample=lambda: {})))
    monkeypatch.setattr(resources, 'ResourceGuard', lambda: SimpleNamespace(observe=lambda *a, **kw: None))

    def isolated(*args, check, **kwargs):
        try:
            check()
        except ComfyCancellation:
            raise process.IsolatedTaskError({'status': 'controller_failed' if not remaining else 'cleanup_failed',
                'active_after_cleanup': remaining})
        pytest.fail('Cancellation callback was not called')

    monkeypatch.setattr(process, 'run_isolated', isolated)
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture')
    def interrupt():
        raise cancellation

    expected = ComfyCancellation if not remaining else process.IsolatedTaskError
    with pytest.raises(expected) as error:
        topaz_runtime.run_regular(runtime, source, tmp_path / 'task', model_id='iris-3',
            width=None, height=None, scale=1, lease_path=tmp_path / 'lease', interrupt=interrupt)
    if not remaining:
        assert error.value is cancellation
    assert (tmp_path / 'task/process.json').is_file()
