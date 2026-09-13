"""Publication must rebind a worker result to the original source and new master."""
import pytest

from h3_audio_t8_pkg import topaz_runtime as runtime
from h3_audio_t8_pkg.topaz_media import file_identity


def fixture(tmp_path):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    job = tmp_path / 'job'
    job.mkdir()
    output = job / 'enhanced.mov'
    output.write_bytes(b'output')
    spec = {'source': file_identity(source), 'settings': {'output_profile': 'lossless_master'}}
    report = {'status': 'media_audit_pass_human_pending',
        'source': spec['source'], 'output': file_identity(output)}
    return source, job, output, spec, report


def test_publication_rechecks_source_and_output(tmp_path):
    _, job, output, spec, report = fixture(tmp_path)
    assert runtime.validate_publication(job, spec, report) == output


def test_main10_publication_uses_delivery_container(tmp_path):
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'source')
    job = tmp_path / 'job'
    job.mkdir()
    output = job / 'enhanced.mp4'
    output.write_bytes(b'output')
    spec = {'source': file_identity(source), 'settings': {'output_profile': 'delivery_hevc_main10'}}
    report = {'status': 'media_audit_pass_human_pending', 'source': spec['source'],
        'output': file_identity(output)}
    assert runtime.validate_publication(job, spec, report) == output


@pytest.mark.parametrize('kind', ['source_bytes', 'output_bytes', 'foreign_path', 'report_source', 'pending_name'])
def test_changed_or_foreign_publication_is_rejected(tmp_path, kind):
    source, job, output, spec, report = fixture(tmp_path)
    if kind == 'source_bytes':
        source.write_bytes(b'replaced')
    elif kind == 'output_bytes':
        output.write_bytes(b'replaced')
    elif kind in ('foreign_path', 'pending_name'):
        other = (tmp_path / 'foreign.mov') if kind == 'foreign_path' else job / 'enhanced.pending.mov'
        other.write_bytes(b'output')
        report['output'] = file_identity(other)
    else:
        report['source'] = {**spec['source'], 'sha256': '0' * 64}
    with pytest.raises((ValueError, RuntimeError), match='(?i)source|output|publication|master'):
        runtime.validate_publication(job, spec, report)
