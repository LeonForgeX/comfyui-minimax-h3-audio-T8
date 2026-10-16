"""A reviewed second source must not admit unknown files or stale live code."""
import hashlib
from types import ModuleType

import pytest

from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from h3_audio_t8_pkg.progressive_source_audit import SourceAudit


ORIGINAL = 'def observe(value, enabled=True):\n    return value\n'
REVISED = '# reviewed display-only change\n' + ORIGINAL


def digest(source):
    return hashlib.sha256((source.rstrip() + '\n').encode()).hexdigest()


def module(tmp_path, source):
    path = tmp_path / 'observer.py'
    path.write_text(source, encoding='utf-8')
    result = ModuleType('reviewed_observer')
    result.__file__ = str(path)
    exec(compile(source, str(path), 'exec', dont_inherit=True), vars(result))
    return result, path


@pytest.mark.parametrize('source', [ORIGINAL, REVISED])
def test_two_exact_reviewed_sources_keep_live_callable_checks(tmp_path, source):
    loaded, _ = module(tmp_path, source)
    audit = SourceAudit(loaded, digest(ORIGINAL), alternate_digests=(digest(REVISED),))
    audit.top_functions()
    assert audit.digest == digest(source)
    if source == REVISED:
        with pytest.raises(UnverifiedModelStack, match='source revision'):
            SourceAudit(loaded, digest(ORIGINAL))


@pytest.mark.parametrize('mutation', ['unknown_file', 'stale_live_code', 'live_function', 'defaults'])
def test_reviewed_alternate_does_not_exempt_source_or_execution_mutation(tmp_path, mutation):
    loaded, path = module(tmp_path, ORIGINAL)
    if mutation == 'unknown_file':
        path.write_text(ORIGINAL + '# not reviewed\n', encoding='utf-8')
    elif mutation == 'stale_live_code':
        # Even another allowed file cannot stand in for the already loaded code.
        path.write_text(REVISED, encoding='utf-8')
    elif mutation == 'live_function':
        loaded.observe = lambda value, enabled=True: value
    else:
        loaded.observe.__defaults__ = (False,)
    with pytest.raises(UnverifiedModelStack):
        SourceAudit(loaded, digest(ORIGINAL), alternate_digests=(digest(REVISED),)).top_functions()
