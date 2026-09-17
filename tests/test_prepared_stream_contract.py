"""Ordered manifest identities and independent teacher/video seed semantics."""

from copy import deepcopy

import pytest

from h3_audio_t8_pkg.prepared_stream_contract import (
    bind_stream_video_seeds,
    validate_stream_requests,
)


@pytest.fixture
def items(tmp_path):
    return [
        dict(
            request_index=i,
            audio_seed=8301 + i,
            text_features=str(tmp_path / f"text-{i}.safetensors"),
            milestones=str(tmp_path / f"teacher-{i}.safetensors"),
        )
        for i in range(4)
    ]


def test_all_requests_bound_without_mutation_and_seed_wrap(items):
    before = deepcopy(items)
    paths = [item[key] for item in items for key in ("text_features", "milestones")]
    assert validate_stream_requests(items, asset_paths=paths) is items
    bound = bind_stream_video_seeds(items, 2**64 - 2)
    assert [item["video_seed"] for item in bound] == [2**64 - 2, 2**64 - 1, 0, 1]
    assert [item["audio_seed"] for item in bound] == [8301, 8302, 8303, 8304]
    assert items == before


@pytest.mark.parametrize("value", [True, -1, 2**64, 0.0, "8301", None])
def test_seed_types_and_bounds(items, value):
    with pytest.raises(ValueError):
        bind_stream_video_seeds(items, value)
    items[-1]["audio_seed"] = value
    with pytest.raises(ValueError):
        validate_stream_requests(items)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "tuple",
        "duplicate",
        "skip",
        "bool_index",
        "injected_seed",
        "injected_worker",
        "missing",
        "relative",
        "traversal",
    ],
)
def test_invalid_order_or_fields(items, fault):
    if fault == "empty":
        items = []
    elif fault == "tuple":
        items = tuple(items)
    elif fault == "duplicate":
        items[2]["request_index"] = 1
    elif fault == "skip":
        items[2]["request_index"] = 3
    elif fault == "bool_index":
        items[1]["request_index"] = True
    elif fault == "injected_seed":
        items[0]["video_seed"] = 1
    elif fault == "injected_worker":
        items[0]["worker"] = "external.py"
    elif fault == "missing":
        del items[-1]["milestones"]
    elif fault == "relative":
        items[0]["text_features"] = "relative.safetensors"
    elif fault == "traversal":
        items[0]["text_features"] += "/../outside.safetensors"
    with pytest.raises(ValueError):
        validate_stream_requests(items)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("key", ["text_features", "milestones"])
def test_each_request_file_needs_identity(items, index, key):
    paths = [item[k] for item in items for k in ("text_features", "milestones")]
    paths.remove(items[index][key])
    with pytest.raises(ValueError, match="asset identity"):
        validate_stream_requests(items, asset_paths=paths)


@pytest.fixture
def bundle(items, tmp_path):
    from h3_audio_t8_pkg.prepared_backend.resource_guard import file_identity

    files = {
        name: str(tmp_path / (name + ".fixture"))
        for name in ("download_receipt", "video_vae", "audio_vae")
    }
    for item in items:
        for key in ("text_features", "milestones"):
            files[f"{key}{item['request_index']}"] = item[key]
    from pathlib import Path

    directories = [
        tmp_path / "base/FL2VA/transformer",
        tmp_path / "adapter",
        tmp_path / "teacher",
    ]
    trees = []
    for directory in directories:
        directory.mkdir(parents=True)
        member = directory / "fixture.bin"
        files[str(member)] = str(member)
        trees.append(dict(path=str(directory), files=[str(member)]))
    for path in files.values():
        Path(path).write_bytes(b"CPU fixture, not model weights")
    source = str(tmp_path / "source")
    return dict(
        schema="t8_prepared_generation_bundle_v1",
        kind="tao_stream",
        generation=dict(
            schema="t8-taomate-prepared-stream-v1",
            source=source,
            source_revision="a" * 40,
            base=str(tmp_path / "base"),
            adapter=str(tmp_path / "adapter"),
            teacher=str(tmp_path / "teacher"),
            download_receipt=files["download_receipt"],
            stream_requests=items,
        ),
        decode=dict(
            schema="t8-taomate-stream-decode-v1",
            core=str(tmp_path / "core"),
            source=source,
            video_vae=files["video_vae"],
            audio_vae=files["audio_vae"],
            request_count=len(items),
        ),
        assets=[file_identity(path) for path in files.values()],
        trees=trees,
        source_revisions=[
            dict(path=source, revision="a" * 40),
            dict(path=str(tmp_path / "core"), revision="b" * 40),
        ],
    )


def test_public_generation_contract_binds_all_requests(bundle):
    from h3_audio_t8_pkg.prepared_generation_contract import (
        validate_bundle,
        generation_request,
    )

    assert validate_bundle(bundle) is bundle
    before = deepcopy(bundle)
    request = generation_request(bundle, 123, "fixtureGPU", {})
    assert [item["video_seed"] for item in request["stream_requests"]] == [
        123,
        124,
        125,
        126,
    ]
    assert [item["audio_seed"] for item in request["stream_requests"]] == [
        8301,
        8302,
        8303,
        8304,
    ]
    assert request["schema"] == "t8-taomate-prepared-stream-v1"
    assert bundle == before


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "bool_count",
        "source",
        "teacher_tree",
        "source_pin",
        "text_identity",
        "video_vae_identity",
        "audio_vae_identity",
        "injected_worker",
        "injected_seed",
    ],
)
def test_public_stream_bundle_rejects_invalid_binding(bundle, fault):
    from h3_audio_t8_pkg.prepared_generation_contract import validate_bundle

    if fault == "count":
        bundle["decode"]["request_count"] = 3
    elif fault == "bool_count":
        bundle["decode"]["request_count"] = True
    elif fault == "source":
        bundle["decode"]["source"] += "-changed"
    elif fault == "teacher_tree":
        bundle["trees"].pop()
    elif fault == "source_pin":
        bundle["source_revisions"][0]["revision"] = "c" * 40
    elif fault.endswith("identity"):
        key = (
            "text_features"
            if fault == "text_identity"
            else fault.removesuffix("_identity")
        )
        path = (
            bundle["generation"]["stream_requests"][2][key]
            if key == "text_features"
            else bundle["decode"][key]
        )
        bundle["assets"] = [
            asset for asset in bundle["assets"] if asset["path"] != path
        ]
    elif fault == "injected_worker":
        bundle["generation"]["worker"] = "external.py"
    else:
        bundle["generation"]["stream_requests"][0]["video_seed"] = 1
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_stream_route_uses_packaged_workers_and_original_routes_survive():
    from h3_audio_t8_pkg.prepared_generation_contract import BACKEND
    from h3_audio_t8_pkg.prepared_generation_runtime import ROUTES

    assert set(ROUTES) == {"tao5s", "tao_stream", "ltx_refine"}
    assert ROUTES["tao5s"][0][0] == "taomate_video_worker.py"
    assert ROUTES["ltx_refine"][0][0] == "ltx_prepared_refinement_worker.py"
    for stage in ROUTES["tao_stream"]:
        assert (BACKEND / stage[0]).is_file() and 0 < stage[3] <= 7200


def test_preparation_tool_inventories_every_ordered_input(bundle, monkeypatch):
    from tools import prepare_generation_bundle as tool
    from h3_audio_t8_pkg.prepared_generation_contract import validate_bundle

    pins = {item["path"]: item for item in bundle["source_revisions"]}
    monkeypatch.setattr(tool, "source_pin", lambda path: pins[str(path)])
    prepared = tool.prepare("tao_stream", bundle["generation"], bundle["decode"])
    validate_bundle(prepared)
    assert prepared["generation"] == bundle["generation"]
    assert prepared["decode"] == bundle["decode"]
    assert {item["path"] for item in prepared["assets"]} == {
        item["path"] for item in bundle["assets"]
    }
    with pytest.raises(ValueError, match="per-request"):
        tool.prepare("tao_stream", bundle["generation"], bundle["decode"], audio_seed=0)
