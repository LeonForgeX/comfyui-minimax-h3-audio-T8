import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("registry_release_change", ROOT / "tools/registry_release_change.py")
CHANGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHANGE)
OLD, NEW = "a" * 40, "b" * 40


@pytest.mark.parametrize("previous,current,expected", [
    ("1.85.0", "1.85.0", False), ("1.85.0", "1.85.1", True), ("1.85.0", "1.86.0", True),
])
def test_only_actual_version_changes_republish(monkeypatch, previous, current, expected):
    monkeypatch.setattr(CHANGE, 'version_at', lambda root, sha: previous if sha == OLD else current)
    assert CHANGE.should_publish(ROOT, 'push', OLD, NEW)['publish'] is expected


def test_manual_dispatch_still_requires_real_publish_and_activation(monkeypatch):
    monkeypatch.setattr(CHANGE, 'version_at', lambda *args: pytest.fail('Unneeded comparison'))
    assert CHANGE.should_publish(ROOT, 'workflow_dispatch', '', NEW)['publish'] is True


def test_initial_push_can_publish(monkeypatch):
    monkeypatch.setattr(CHANGE, 'version_at', lambda *args: '1.85.0')
    assert CHANGE.should_publish(ROOT, 'push', '0' * 40, NEW)['publish'] is True


def test_unknown_event_is_not_silently_skipped():
    with pytest.raises(ValueError, match='event'):
        CHANGE.should_publish(ROOT, 'pull_request', OLD, NEW)


def test_invalid_git_revision_is_rejected_before_shell_execution():
    with pytest.raises(ValueError, match='SHA'):
        CHANGE.version_at(ROOT, '--invalid-revision')


def test_missing_git_base_is_error_not_false_success(monkeypatch):
    def fail(*args):
        raise ValueError('Missing base')
    monkeypatch.setattr(CHANGE, 'version_at', fail)
    with pytest.raises(ValueError, match='Missing base'):
        CHANGE.should_publish(ROOT, 'push', OLD, NEW)


def test_workflow_guards_both_publication_and_activation_not_version_integrity():
    text = (ROOT / '.github/workflows/publish_action.yml').read_text(encoding='utf-8')
    assert 'tools/registry_release_change.py' in text
    assert text.count("if: steps.release-change.outputs.publish == 'true'") == 2
    assert 'tools/prepare_release_version.py --check' in text
    assert 'tools/verify_registry_publish.py' in text
