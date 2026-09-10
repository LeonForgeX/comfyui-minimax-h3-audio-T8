import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tools.trt_encoder_condition_probe import SavedReferenceEncoder, condition_recipe, digest, evidence_identity, load_reference
from tools.run_progressive_pilot import RESEARCH, instrument_recipe


def fixture_evidence(root):
    from safetensors.torch import save_file
    pixels = torch.arange(512 * 1024 * 3, dtype=torch.int32).remainder(256).byte().reshape(1, 3, 1, 512, 1024)
    save_file({"image": pixels}, root / "source-rgb8.safetensors")
    for name, value in (("native", 1.), ("trt", 2.)):
        save_file({"image": torch.full((1, 24, 1, 32, 64), value)}, root / (name + "-latents.safetensors"))
    for name in ("request.json", "result.json"):
        (root / name).write_text("{}")
    audit = {"status": "full_0p5mp_encoder_evidence_audited_not_downstream_quality_qualification",
             "source_pixels_exact": True, "pixel_normalization_exact": True, "actual_trt_calls": 90,
             "files": {p.name: digest(p) for p in root.iterdir()}}
    path = root / "independent-encoder-audit.json"
    path.write_text(json.dumps(audit))
    return digest(path)


@pytest.mark.parametrize("backend,expected", [("native", 1.), ("trt", 2.)])
def test_saved_reference_exact_actual_project_resize_and_encode(tmp_path, backend, expected):
    from h3_audio_t8_pkg.core import resize_image
    sha = fixture_evidence(tmp_path)
    vae = SimpleNamespace(decode=lambda value: value, property_marker="original")
    delegate, image = load_reference(vae, tmp_path, backend, sha, allowed_root=tmp_path)
    with pytest.raises(ValueError, match="consumed"):
        delegate.report()
    prepared = resize_image(image, 1024, 512, "disabled")
    result = delegate.encode(prepared)
    assert torch.equal(prepared, image) and bool((result == expected).all())
    assert delegate.property_marker == "original" and delegate.decode(image) is image
    assert delegate.report()["actual_encode_calls"] == 1
    result.zero_()
    assert bool((delegate.latent == expected).all())
    with pytest.raises(ValueError, match="exactly one"):
        delegate.encode(image)


def test_saved_reference_rejects_changed_pixels_and_dtypes():
    image = torch.zeros(1, 512, 1024, 3)
    delegate = SavedReferenceEncoder(object(), image, torch.zeros(1, 24, 1, 32, 64), {}, "trt")
    for invalid in (image + 0.001, image.half(), image[:, :, :512]):
        with pytest.raises(ValueError, match="unchanged"):
            delegate.encode(invalid)
    assert delegate.calls == 0


def test_saved_reference_rejects_nonfinite_latent():
    with pytest.raises(ValueError, match="Nonfinite"):
        SavedReferenceEncoder(object(), torch.zeros(1, 512, 1024, 3), torch.full((1, 24, 1, 32, 64), float("nan")), {}, "native")


def test_evidence_changed_or_wrong_location_is_rejected(tmp_path):
    fixture_evidence(tmp_path)
    with pytest.raises(ValueError, match="research"):
        evidence_identity(tmp_path, allowed_root=tmp_path / "different")
    (tmp_path / "result.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        evidence_identity(tmp_path, allowed_root=tmp_path)


def test_saved_reference_rejects_changed_audit(tmp_path):
    fixture_evidence(tmp_path)
    with pytest.raises(ValueError, match="audit identity"):
        load_reference(object(), tmp_path, "native", "0" * 64, allowed_root=tmp_path)


def test_recipe_changes_only_reference_path_and_keeps_native_final_decode():
    graph = instrument_recipe(json.loads((RESEARCH / "pilot-api-drafts/I2VA_native8.prompt.json").read_text()))
    before = deepcopy(graph)
    native = condition_recipe(graph, "evidence", "native", "a" * 64)
    trt = condition_recipe(graph, "evidence", "trt", "a" * 64)
    assert graph == before
    for node in ("1", "2", "3", "4", "5", "7", "10", "11", "12", "90", "91"):
        assert native[node] == graph[node] == trt[node]
    assert native["11"]["inputs"]["video_vae"] == ["1", 0]
    assert native["6"]["inputs"]["first_frame"] == ["107", 1]
    assert native["100"]["inputs"]["positive"] == ["108", 0]
    trt["107"]["inputs"]["backend"] = "native"
    trt["18"] = native["18"]
    assert trt == native
    for node in native.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in native


def test_recipe_refuses_other_tasks():
    graph = instrument_recipe(json.loads((RESEARCH / "pilot-api-drafts/T2VA_native8.prompt.json").read_text()))
    with pytest.raises(ValueError, match="fixed I2VA"):
        condition_recipe(graph, "evidence", "trt", "a" * 64)
