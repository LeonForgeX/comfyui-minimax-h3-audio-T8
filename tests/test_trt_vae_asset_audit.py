import hashlib

import pytest

from tools.audit_trt_vae_environment import contained_file, hash_file


def test_data_digest(tmp_path):
    path = tmp_path / "weights.data"
    content = b"trt-weights" * 1000
    path.write_bytes(content)
    assert hash_file(path) == hashlib.sha256(content).hexdigest()


def test_local_external_data_allowed(tmp_path):
    path = tmp_path / "weights.data"
    path.touch()
    assert contained_file(tmp_path, "weights.data") == path.resolve()


@pytest.mark.parametrize("name", ["../escape.data", "missing.data"])
def test_missing_or_traversal_data_rejected(tmp_path, name):
    (tmp_path.parent / "escape.data").touch()
    with pytest.raises(ValueError):
        contained_file(tmp_path, name)


def test_absolute_external_data_rejected_even_inside_root(tmp_path):
    path = tmp_path / "weights.data"
    path.touch()
    with pytest.raises(ValueError, match="Absolute"):
        contained_file(tmp_path, str(path.resolve()))
