"""Real safetensors IO, atomic publication and corruption/cancel contracts."""

import json
from pathlib import Path
import struct

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg.meridian_checkpoint_io import (
    assemble_checkpoint,
    atomic_json,
    canonical_identity,
    conversion_lock,
    file_sha,
    part_header,
)


@pytest.fixture
def parts(tmp_path):
    tensors = [
        dict(
            a=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            b=torch.ones(5, dtype=torch.bfloat16),
        ),
        dict(
            c=torch.arange(8, dtype=torch.int8),
            d=torch.tensor([1, 2, 3], dtype=torch.uint8),
        ),
    ]
    result = []
    for i, values in enumerate(tensors):
        path = tmp_path / f"part-{i}.safetensors"
        save_file(values, path, metadata={"layer": str(i)})
        result.append(dict(path=str(path), sha256=file_sha(path)))
    return result, {k: v for group in tensors for k, v in group.items()}


def test_streamed_checkpoint_is_native_and_preserves_tensor_bits(parts, tmp_path):
    source, wanted = parts
    dest = tmp_path / "converted.safetensors"
    progress = []
    report = assemble_checkpoint(
        source,
        dest,
        {"format": "pt", "dmd_merged": "true"},
        progress=lambda *row: progress.append(row),
    )
    assert report["sha256"] == file_sha(dest)
    assert report["tensor_count"] == 4 and report["bytes"] == dest.stat().st_size
    loaded = load_file(dest)
    assert set(loaded) == set(wanted)
    assert all(torch.equal(value, loaded[name]) for name, value in wanted.items())
    with safe_open(dest, framework="pt") as handle:
        assert handle.metadata() == {"format": "pt", "dmd_merged": "true"}
    assert [row[:2] for row in progress] == [(1, 2), (2, 2)]
    assert not list(tmp_path.glob("*.partial-*"))


def test_destination_never_overwritten(parts, tmp_path):
    dest = tmp_path / "already.safetensors"
    dest.write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        assemble_checkpoint(parts[0], dest, {})
    assert dest.read_bytes() == b"user data"


def test_racing_destination_is_not_overwritten(parts, tmp_path):
    dest = tmp_path / "raced.safetensors"

    def race(i, count, name):
        if i == count:
            dest.write_bytes(b"other writer")

    with pytest.raises(FileExistsError):
        assemble_checkpoint(parts[0], dest, {}, progress=race)
    assert dest.read_bytes() == b"other writer"
    assert not list(tmp_path.glob("*.partial-*"))


@pytest.mark.parametrize("when", ["before", "after_last_part"])
def test_cancel_never_publishes_preserves_parts(parts, tmp_path, when):
    source, _ = parts
    cancelled = [when == "before"]

    def progress(i, count, name):
        if i == count:
            cancelled[0] = True

    with pytest.raises(InterruptedError):
        assemble_checkpoint(
            source,
            tmp_path / "no.safetensors",
            {},
            cancelled=lambda: cancelled[0],
            progress=progress,
        )
    assert not (tmp_path / "no.safetensors").exists()
    assert all(file_sha(p["path"]) == p["sha256"] for p in source)
    assert not list(tmp_path.glob("*.partial-*"))


def test_same_size_payload_mutation_refused(parts, tmp_path):
    source, _ = parts
    path = Path(source[0]["path"])
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA mismatch"):
        assemble_checkpoint(source, tmp_path / "no.safetensors", {})
    assert not (tmp_path / "no.safetensors").exists()
    assert not list(tmp_path.glob("*.partial-*"))


def test_duplicate_target_rejected(parts, tmp_path):
    source, _ = parts
    with pytest.raises(ValueError, match="Duplicate output"):
        assemble_checkpoint([source[0], source[0]], tmp_path / "no.safetensors", {})


@pytest.mark.parametrize(
    "change", ["dtype", "span", "shape_bool", "trailing", "truncated", "overlap"]
)
def test_corrupt_header_or_payload_refused(tmp_path, change):
    header = {"x": dict(dtype="F32", shape=[2], data_offsets=[0, 8])}
    payload = b"\0" * 8
    if change == "dtype":
        header["x"]["dtype"] = "F64"
    if change == "span":
        header["x"]["data_offsets"] = [0, 7]
    if change == "shape_bool":
        header["x"]["shape"] = [True, 2]
    if change == "trailing":
        payload += b"x"
    if change == "truncated":
        payload = payload[:-1]
    if change == "overlap":
        header["y"] = dict(header["x"])
    encoded = json.dumps(header).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    with pytest.raises(ValueError):
        part_header(path)


def test_duplicate_header_keys_refused(tmp_path):
    header = b'{"x":{},"x":{}}'
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="Duplicate"):
        part_header(path)


def test_atomic_json_and_identity(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, dict(a=1, b=2))
    with pytest.raises(FileExistsError):
        atomic_json(path, dict(a=5))
    assert json.loads(path.read_text()) == dict(a=1, b=2)
    atomic_json(path, dict(a=7), replace_existing=True)
    assert json.loads(path.read_text()) == dict(a=7)
    assert canonical_identity(dict(a=1, b=2)) == canonical_identity(dict(b=2, a=1))
    assert not list(tmp_path.glob("*.partial-*"))


def test_owned_lock_serializes_and_releases(tmp_path):
    path = tmp_path / "convert.lock"
    with conversion_lock(path):
        with pytest.raises(RuntimeError, match="another process"):
            with conversion_lock(path):
                pass
    with conversion_lock(path):
        pass
