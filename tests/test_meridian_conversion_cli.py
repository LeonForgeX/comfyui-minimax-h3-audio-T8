"""Pinned Hub metadata binding without network requests or large test assets."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.meridian_checkpoint_io import file_sha


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "tools/convert_meridian_convrot.py"
    spec = importlib.util.spec_from_file_location("meridian_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources(tmp_path, cli):
    (tmp_path / "transformer").mkdir()
    cfg = dict(
        hidden_size=5376,
        num_layers=50,
        num_refiner_layers=2,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_dim=14336,
        time_embed_dim=2688,
        freq_dim=256,
        time_embed_hidden_dim=5376,
        text_dim=5120,
        in_channels=24,
        audio_in_channels=32,
        rope_freq_dim=16,
        rope_theta=10000.0,
        patch_size=[1, 2, 2],
    )
    index = dict(
        metadata=dict(total_size=66280430080),
        weight_map={
            f"placeholder-{i}": f"diffusion_pytorch_model-{i:05d}-of-00014.safetensors"
            for i in range(1, 15)
        },
    )
    siblings = []
    for name, obj in ((cli.CONFIG, cfg), (cli.INDEX, index)):
        raw = json.dumps(obj).encode()
        (tmp_path / name).write_bytes(raw)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        siblings.append(
            SimpleNamespace(rfilename=name, size=len(raw), blob_id=blob, lfs=None)
        )
    for name in [cli.DMD] + ["transformer/" + s for s in index["weight_map"].values()]:
        siblings.append(
            SimpleNamespace(
                rfilename=name,
                size=100,
                lfs=SimpleNamespace(
                    size=100, sha256=cli.DMD_SHA if name == cli.DMD else "b" * 64
                ),
            )
        )
    return tmp_path, SimpleNamespace(id=cli.REPO, sha=cli.REVISION, siblings=siblings)


def test_manifest_contains_expected_digests_not_claimed_local_verification(
    cli, sources
):
    root, info = sources
    manifest = cli.build_manifest(root, info)
    assert len(manifest["files"]) == 17
    assert manifest["revision"] == cli.REVISION
    index = next(f for f in manifest["files"] if f["path"] == cli.INDEX)
    assert index["sha256"] == file_sha(root / cli.INDEX)
    assert not (root / cli.DMD).exists()  # metadata request must not download weights


@pytest.mark.parametrize(
    "problem", ["revision", "repo", "blob", "dmd", "duplicate", "lfs_size"]
)
def test_manifest_rejects_wrong_pinned_metadata(cli, sources, problem):
    root, info = sources
    if problem == "revision":
        info.sha = "c" * 40
    elif problem == "repo":
        info.id = "elsewhere/model"
    elif problem == "blob":
        info.siblings[0].blob_id = "a" * 40
    elif problem == "dmd":
        info.siblings[2].lfs.sha256 = "b" * 64
    elif problem == "duplicate":
        info.siblings.append(info.siblings[0])
    else:
        info.siblings[2].lfs.size = 90
    with pytest.raises(ValueError):
        cli.build_manifest(root, info)


def test_file_hash_interrupts_before_reading_whole_model(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"x" * 20_000_000)
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) == 2

    with pytest.raises(InterruptedError):
        file_sha(path, cancelled=cancelled)
    assert len(calls) == 2
